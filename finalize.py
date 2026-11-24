"""
finalize.py — Robust recovery + final assembly of cars_data.json.

WHY THIS EXISTS
---------------
A straight re-run of run_pipeline.py got throttled: ~81/100 detail pages
returned HTTP 202 (a soft rate-limit) on the second pass, which would have
discarded the 1000+ images already downloaded on disk. This script avoids
that by:

  1. Re-fetching the LISTING once (1 request) to get the 100 newest cars and
     their thumbnail URLs (which encode each car's image-folder prefix).
  2. Reusing any mileage already captured in the current cars_data.json
     (from whichever ads' detail fetch previously succeeded).
  3. Fetching detail pages SLOWLY (2s polite delay) with explicit HTTP 202
     retry+backoff, ONLY for ads still missing mileage. This is the one piece
     of data not recoverable from disk or the listing.
  4. Rebuilding each ad's image list from the FILES ALREADY ON DISK
     (images/<cid>_NN.jpg) — no re-download. Source URLs are reconstructed
     from the listing thumbnail's cidx prefix.
  5. Re-scoring every image with the local Classical-CV IQA scorer.

Output: a clean cars_data.json with all 100 ads, brand fixed (from namemmt),
mileage where the site lists it, and every on-disk image linked + scored.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import asdict

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except ImportError:
    from requests.packages.urllib3.util.retry import Retry  # type: ignore

from classify import classify_car
from image_quality import assess_image

BASE = "https://www.taladrod.com/w40"
LISTING_URL = f"{BASE}/isch/schc.aspx?fno:new"
DETAIL_URL_TPL = f"{BASE}/iCar/CarDet.aspx?cid={{cid}}"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    })
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET", "HEAD"]),
                  raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def fetch_with_202_retry(session: requests.Session, url: str,
                         referer: str | None = None,
                         max_tries: int = 6, base_wait: float = 5.0):
    """GET url; on HTTP 202 (soft throttle) wait with growing backoff and retry.

    Returns the response text on HTTP 200, or None on persistent failure.
    """
    headers = {"Referer": referer} if referer else {}
    for attempt in range(1, max_tries + 1):
        try:
            r = session.get(url, headers=headers, timeout=30, allow_redirects=True)
            if r.status_code == 200:
                return r.text
            if r.status_code == 202:
                wait = base_wait * attempt + random.uniform(0, 2)
                print(f"    [202 throttle] waiting {wait:.1f}s "
                      f"(attempt {attempt}/{max_tries})", flush=True)
                time.sleep(wait)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * attempt)
                continue
            print(f"    [skip] HTTP {r.status_code} on {url}")
            return None
        except requests.RequestException as e:
            print(f"    [retry] {type(e).__name__} (attempt {attempt})", flush=True)
            time.sleep(2 * attempt)
    return None


def cidx_prefix(thumbnail_url: str) -> str | None:
    m = re.search(r"(https://imgc2\.taladrod\.com/c/cidx/\d+/\d+/\d+)",
                  thumbnail_url or "")
    return m.group(1) if m else None


def images_on_disk(images_dir: str, cid: str):
    """Sorted [(idx, filepath)] for images/<cid>_NN.jpg on disk."""
    out = []
    if not os.path.isdir(images_dir):
        return out
    pat = re.compile(rf"^{re.escape(cid)}_(\d+)\.jpg$")
    for fn in os.listdir(images_dir):
        m = pat.match(fn)
        if m:
            out.append((int(m.group(1)), os.path.join(images_dir, fn)))
    out.sort()
    return out


def parse_mileage(html: str):
    m = re.search(r"ไมล์\s*([\d,]+)\s*กม", html)
    if not m:
        return None, None
    raw = m.group(1)
    digits = raw.replace(",", "")
    return raw, (int(digits) if digits.isdigit() else None)


def extract_gallery(html: str, own_prefix: str | None):
    """Gallery image URLs for this car, filtered to its own cidx prefix."""
    raw = re.findall(
        r"""src=["'](https://imgc2\.taladrod\.com/c/cidx/[^"']+?_\d+\.jpg[^"']*)["']""",
        html)
    groups: dict[str, list[str]] = {}
    for u in raw:
        gp = re.match(
            r"(https://imgc2\.taladrod\.com/c/cidx/\d+/\d+/\d+)_\d+\.jpg", u)
        if gp:
            groups.setdefault(gp.group(1), []).append(u)
    if own_prefix and own_prefix in groups:
        picked = groups[own_prefix]
    elif groups:
        picked = max(groups.values(), key=len)
    else:
        picked = []
    seen, out = set(), []
    for u in picked:
        clean = re.sub(r"\?.*$", "", u)
        if clean not in seen:
            seen.add(clean)
            out.append(u)
    out.sort(key=lambda u: int(re.search(r"_(\d+)\.jpg", u).group(1))
             if re.search(r"_(\d+)\.jpg", u) else 999)
    return out


def download_missing(session, images_dir, cid, gallery_urls, own_prefix, existing_idx):
    """Download any gallery image not already on disk."""
    for u in gallery_urls:
        gm = re.search(r"_(\d+)\.jpg", u)
        if not gm:
            continue
        n = int(gm.group(1))
        if n in existing_idx:
            continue
        dest = os.path.join(images_dir, f"{cid}_{n:02d}.jpg")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        try:
            rr = session.get(u, headers={"Referer": DETAIL_URL_TPL.format(cid=cid)},
                             timeout=30)
            if rr.status_code == 200 and rr.content:
                # quick image validity check
                if rr.content.startswith(b"\xff\xd8") or rr.content.startswith(b"\x89PNG"):
                    with open(dest, "wb") as f:
                        f.write(rr.content)
                    print(f"    [img] downloaded {os.path.basename(dest)}", flush=True)
                    time.sleep(0.8 + random.uniform(-0.2, 0.2))
        except requests.RequestException:
            pass


def main():
    images_dir = "images"
    out_path = "cars_data.json"
    sess = make_session()

    # --- 1. listing ---
    print("[1] fetching listing...", flush=True)
    html = fetch_with_202_retry(sess, LISTING_URL)
    if not html:
        raise SystemExit("FATAL: could not fetch listing page")
    m = re.search(r"SchDataJSON\s*=\s*(\{.*?\})\s*;", html, re.DOTALL)
    cars_all = json.loads(m.group(1))["cars"]
    cars = [c for c in cars_all if c.get("issold", "N") != "Y"][:100]
    print(f"    {len(cars)} cars selected (newest, sold skipped)", flush=True)

    # --- 2. reuse any mileage already in current JSON ---
    existing = {}
    if os.path.exists(out_path):
        for r in json.load(open(out_path, encoding="utf-8")):
            existing[r["ad_id"]] = r
        print(f"    reusing mileage for {sum(1 for r in existing.values() if r.get('mileage_km'))} ads from current JSON", flush=True)

    # --- 3. build records ---
    records = []
    fetched_detail = 0
    for i, c in enumerate(cars, 1):
        cid = c["cid"]
        namemmt = (c.get("namemmt") or "").strip()
        brand = (c.get("amake") or "").strip() or (namemmt.split()[0] if namemmt else "")
        model = (c.get("amodel") or c.get("model") or "").strip()
        if not model and namemmt:
            parts = namemmt.split()
            model = parts[1] if len(parts) > 1 else ""
        year = (c.get("yr4") or "").strip()
        price_raw = c.get("prc") or c.get("sbaht") or ""
        price_baht = int(price_raw.replace(",", "")) if price_raw.replace(",", "").isdigit() else 0
        thumb = c.get("img", "")
        prefix = cidx_prefix(thumb)
        cls = classify_car(c)

        mileage = existing.get(cid, {}).get("mileage")
        mileage_km = existing.get(cid, {}).get("mileage_km")
        gallery_urls = []

        if mileage_km is None:
            # Need a detail fetch for mileage (and to confirm gallery).
            print(f"[{i}/100] {cid}  {brand} {model} {year} — fetching detail...", flush=True)
            dhtml = fetch_with_202_retry(sess, DETAIL_URL_TPL.format(cid=cid),
                                         referer=LISTING_URL)
            time.sleep(2.0 + random.uniform(-0.3, 0.3))  # polite delay
            if dhtml:
                mileage, mileage_km = parse_mileage(dhtml)
                gallery_urls = extract_gallery(dhtml, prefix)
                fetched_detail += 1
            else:
                print("    [warn] detail fetch failed; will rely on disk images", flush=True)
        else:
            print(f"[{i}/100] {cid}  {brand} {model} {year} — mileage cached ({mileage_km} km)", flush=True)

        # --- 4. images: prefer disk, download any missing ---
        disk = images_on_disk(images_dir, cid)
        existing_idx = set(idx for idx, _ in disk)
        if gallery_urls and len(disk) < len(gallery_urls):
            download_missing(sess, images_dir, cid, gallery_urls, prefix, existing_idx)
            disk = images_on_disk(images_dir, cid)

        # --- 5. score + assemble image entries ---
        images = []
        for idx, fpath in disk:
            src = f"{prefix}_{idx}.jpg" if prefix else ""
            entry = {"local_path": f"images/{os.path.basename(fpath)}",
                     "source_url": src}
            try:
                res = assess_image(fpath)
                entry["quality_score"] = res.score
                entry["quality"] = {
                    "sharpness": res.sharpness, "brightness": res.brightness,
                    "contrast": res.contrast, "colorfulness": res.colorfulness,
                    "saturation": res.saturation, "clipping_penalty": res.clipping_penalty,
                    "width": res.width, "height": res.height, "note": res.note,
                }
            except Exception as e:
                entry["quality_score"] = None
                entry["quality"] = {"error": f"{type(e).__name__}: {e}"}
            images.append(entry)

        records.append({
            "ad_id": cid,
            "brand": brand,
            "model": model,
            "year": year,
            "model_name": f"{brand} {model} {year}".strip(),
            "namemmt_full": namemmt,
            "category": cls.category,
            "category_en": cls.category_en,
            "price": price_raw,
            "price_baht": price_baht,
            "mileage": mileage,
            "mileage_km": mileage_km,
            "ad_source_url": DETAIL_URL_TPL.format(cid=cid),
            "thumbnail_url": thumb,
            "title": (c.get("title") or "").strip(),
            "image_count": len(images),
            "images": images,
        })

    # --- write ---
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    total_imgs = sum(len(r["images"]) for r in records)
    scored = [im["quality_score"] for r in records for im in r["images"]
              if im.get("quality_score") is not None]
    has_mile = sum(1 for r in records if r["mileage_km"])
    empty_brand = sum(1 for r in records if not r["brand"])
    empty_img = sum(1 for r in records if not r["images"])
    print("\n=== FINALIZE SUMMARY ===")
    print(f"ads                  : {len(records)}")
    print(f"detail fetches made  : {fetched_detail} (rest reused/cached)")
    print(f"images linked+scored : {total_imgs} (scored: {len(scored)})")
    print(f"avg quality score    : {sum(scored)/len(scored):.2f}" if scored else "avg quality score    : n/a")
    print(f"ads with mileage     : {has_mile}/{len(records)}")
    print(f"ads with empty brand : {empty_brand}")
    print(f"ads with empty images: {empty_img}")
    print(f"output               : {out_path}")


if __name__ == "__main__":
    main()
