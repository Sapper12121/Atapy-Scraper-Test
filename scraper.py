"""
scraper.py — Web scraper for taladrod.com used-car listings.

============================================================================
WHAT IT DOES
============================================================================
Starting URL (per the assignment):
    https://www.taladrod.com/w40/isch/schc.aspx?fno:new

Step 1 — LISTING: fetch the listing page. taladrod embeds the whole result
          set as a JavaScript variable `SchDataJSON` inside the HTML
          (`<script>var SchDataJSON={"cars":[...],"ncar":"209",...}</script>`).
          We parse that JSON (no JS engine needed) and read the car list.
          Pagination is handled by a loop that looks for a "next page" link
          in the HTML and follows it; for `fno:new` the site returns ALL new
          listings on page 1, so the loop terminates after one fetch
          (data exhausted) — which is exactly the stop condition required.

Step 2 — DETAIL:  for each chosen car (up to max_ads, in display order — see
          SELECTION below), fetch the detail page
          /w40/iCar/CarDet.aspx?cid=<cid> (it 302-redirects to the lowercase
          cardet.aspx form; we follow redirects) and extract:
            - mileage  (regex on Thai "ไมล์ <number> กม.")
            - full gallery image URLs (every imgc2.taladrod.com/.../c_N.jpg)

Step 3 — IMAGES: download every gallery image to a local folder, recording
          both the local path and the original source URL.

SELECTION METHOD (required by the assignment to be stated in code):
    We take the FIRST `max_ads` cars in the order the website lists them
    (newest listings first, because fno:new = "cars posted today") and skip 
    any that are already marked sold (`issold:"Y"`). 

STABILITY (1.3):
    - HTTP retries with exponential backoff via urllib3 Retry (3 attempts,
      backoff factor 0.6, retry on 429/500/502/503/504 and connection errors)
    - A configurable delay between requests (default 1.0s) plus +/-0.3s jitter
      so the site is not hammered and is less likely to block us
    - A browser-like User-Agent and a Referer header on detail requests
    - Pagination loop that stops when no new cars are returned
    - Each downloaded image is verified to be a non-empty, decodable image

This module is importable (see run_pipeline.py) and also runnable directly.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 v2
    from urllib3.util.retry import Retry
except ImportError:  # urllib3 v1 fallback
    from requests.packages.urllib3.util.retry import Retry  # type: ignore


BASE = "https://www.taladrod.com/w40"
LISTING_URL = f"{BASE}/isch/schc.aspx?fno:new"
DETAIL_URL_TPL = f"{BASE}/iCar/CarDet.aspx?cid={{cid}}"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CarRecord:
    ad_id: str
    brand: str
    model: str
    year: str
    price: str                 # raw price string from site, e.g. "550,000"
    price_baht: int            # numeric price in Baht
    mileage: Optional[str]     # raw mileage string e.g. "72,800" or None
    mileage_km: Optional[int]  # numeric km or None
    ad_source_url: str
    thumbnail_url: str
    title: str
    gallery_image_urls: list = field(default_factory=list)
    images: list = field(default_factory=list)  # filled by downloader
    category: Optional[str] = None
    category_en: Optional[str] = None


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class TaladRodScraper:
    def __init__(
        self,
        max_ads: int = 100,
        rate_limit: float = 1.0,
        rate_jitter: float = 0.3,
        max_retries: int = 3,
        backoff_factor: float = 0.6,
        request_timeout: int = 30,
        images_dir: str = "images",
        verbose: bool = True,
    ):
        self.max_ads = max_ads
        self.rate_limit = rate_limit
        self.rate_jitter = rate_jitter
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.images_dir = images_dir
        self.verbose = verbose

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": UA,
            "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
            "Accept": ("text/html,application/xhtml+xml,application/xml;"
                       "q=0.9,image/avif,image/webp,*/*;q=0.8"),
        })
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(202, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        os.makedirs(self.images_dir, exist_ok=True)

    # --- polite sleep -------------------------------------------------------
    def _sleep(self):
        if self.rate_limit > 0:
            time.sleep(max(0.0, self.rate_limit + random.uniform(-self.rate_jitter, self.rate_jitter)))

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    # --- raw GET with retry already on the adapter, but add a final guard ---
    def _get(self, url: str, referer: Optional[str] = None, binary: bool = False):
        headers = {}
        if referer:
            headers["Referer"] = referer
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, headers=headers,
                                        timeout=self.request_timeout, allow_redirects=True)
                if resp.status_code == 200:
                    return resp.content if binary else resp.text
                # Non-200: retry on soft-throttle / server errors.
                # NOTE: taladrod returns HTTP 202 "Accepted" (instead of 429) when
                # it wants us to slow down — the body is NOT the image, so we must
                # retry after a pause, not skip it. A retry almost always returns
                # the real 200 image. 202/429 get an extra cooldown because they
                # are explicit throttle signals.
                if resp.status_code in (202, 429, 500, 502, 503, 504):
                    extra = 1.5 if resp.status_code in (202, 429) else 0.0
                    self._log(f"    [retry] HTTP {resp.status_code} on {url} (attempt {attempt})")
                    time.sleep(self.backoff_sleep(attempt) + extra)
                    continue
                self._log(f"    [skip] HTTP {resp.status_code} on {url}")
                return None
            except requests.RequestException as e:
                last_exc = e
                self._log(f"    [retry] {type(e).__name__} on {url} (attempt {attempt})")
                time.sleep(self.backoff_sleep(attempt))
        if last_exc:
            self._log(f"    [fail] giving up on {url}: {last_exc}")
        return None

    def backoff_sleep(self, attempt: int) -> float:
        return 0.6 * (2 ** (attempt - 1)) + random.uniform(0, 0.4)

    # -----------------------------------------------------------------------
    # STEP 1 — listing (with pagination loop)
    # -----------------------------------------------------------------------
    def fetch_listing_cars(self) -> list[dict]:
        """Return the list of raw car dicts from the listing, paginated.

        We loop over pages: page 1 is LISTING_URL; subsequent pages come from
        a 'next page' link in the HTML if present. The loop stops when a page
        yields no new car ids, when we already have >= max_ads unique cars,
        or when we hit a safety cap of 20 pages.
        """
        cars_by_cid: dict[str, dict] = {}
        url = LISTING_URL
        pages = 0
        while url and pages < 20 and len(cars_by_cid) < self.max_ads:
            pages += 1
            self._log(f"[listing] page {pages}: {url}")
            html = self._get(url)
            self._sleep()
            if not html:
                break
            page_cars = self._parse_listing_json(html)
            new = 0
            for c in page_cars:
                cid = c.get("cid")
                if cid and cid not in cars_by_cid:
                    cars_by_cid[cid] = c
                    new += 1
            self._log(f"           +{new} new cars (total unique: {len(cars_by_cid)})")
            if new == 0:
                break  # data exhausted — no new cars on this page
            # Look for a "next page" link to continue pagination.
            url = self._find_next_page(html, url)
        return list(cars_by_cid.values())

    @staticmethod
    def _parse_listing_json(html: str) -> list[dict]:
        """Extract SchDataJSON["cars"] from the listing HTML."""
        m = re.search(r"SchDataJSON\s*=\s*(\{.*?\})\s*;", html, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(1))
            return data.get("cars", []) or []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _find_next_page(html: str, current_url: str) -> Optional[str]:
        """Look for a 'next page' anchor in the listing HTML.

        taladrod's fno:new returns all cars on page 1 and has no next link,
        so this returns None and the pagination loop stops after page 1.
        If a real next-page link exists for other filters, we follow it.
        """
        # Common patterns: <a ... href="...&prg=N"> or rel="next"
        for pat in (
            r'<a[^>]+rel=["\']next["\'][^>]+href=["\']([^"\']+)',
            r'<a[^>]+href=["\']([^"\']*schc\.aspx[^"\']*prg=\d+[^"\']*)',
            r'<a[^>]+href=["\']([^"\']*schc\.aspx[^"\']*&?p\d*=\d+[^"\']*)',
        ):
            m = re.search(pat, html, re.IGNORECASE)
            if m:
                return urljoin(current_url, m.group(1))
        return None

    # -----------------------------------------------------------------------
    # STEP 2 — detail page
    # -----------------------------------------------------------------------
    def fetch_detail(self, cid: str, car: Optional[dict] = None) -> dict:
        """Fetch a detail page and return {mileage, mileage_km, gallery_urls}.

        `car` is the listing dict; we use its `img` (thumbnail URL) to identify
        THIS car's image-folder prefix so we only collect its own gallery and
        not "similar cars" thumbnails that also appear on the page.
        Returns an empty dict on hard failure so the caller can still record
        the listing-level data.
        """
        url = DETAIL_URL_TPL.format(cid=cid)
        referer = LISTING_URL
        html = self._get(url, referer=referer)
        self._sleep()
        if not html:
            return {}
        return self._parse_detail(html, cid, car)

    @staticmethod
    def _parse_detail(html: str, cid: str, car: Optional[dict] = None) -> dict:
        # Mileage: Thai "ไมล์ 72,800 กม." (also handle without space variants)
        mileage_raw = None
        m = re.search(r"ไมล์\s*([\d,]+)\s*กม", html)
        if m:
            mileage_raw = m.group(1)
        mileage_km = None
        if mileage_raw:
            digits = mileage_raw.replace(",", "")
            if digits.isdigit():
                mileage_km = int(digits)

        # This car's image-folder prefix, taken from the listing thumbnail
        # (e.g. https://imgc2.taladrod.com/c/cidx/014/106/66). The detail page
        # also shows "similar cars" thumbnails with DIFFERENT prefixes, so we
        # filter to this prefix to keep only this car's own gallery.
        own_prefix = None
        if car:
            thumb = car.get("img", "") or ""
            mp = re.search(r"(https://imgc2\.taladrod\.com/c/cidx/\d+/\d+/\d+)", thumb)
            if mp:
                own_prefix = mp.group(1)

        # Gallery images: every imgc2.taladrod.com/c/cidx/.../cN.jpg (full size).
        # Match BOTH single- and double-quoted src attributes — taladrod uses
        # both inconsistently across detail-page templates.
        raw = re.findall(
            r"""src=["'](https://imgc2\.taladrod\.com/c/cidx/[^"']+?_\d+\.jpg[^"']*)["']""",
            html)

        # Group by cidx prefix so we can pick this car's gallery even if the
        # listing thumbnail was empty (fallback: the largest sequential group).
        groups: dict[str, list[str]] = {}
        for u in raw:
            gp = re.match(r"(https://imgc2\.taladrod\.com/c/cidx/\d+/\d+/\d+)_\d+\.jpg", u)
            if gp:
                groups.setdefault(gp.group(1), []).append(u)

        if own_prefix and own_prefix in groups:
            picked = groups[own_prefix]
        elif groups:
            picked = max(groups.values(), key=len)
        else:
            picked = []

        # Dedupe by clean URL (drop cache-busting ?v= for the dedupe key) and
        # sort by the _N index so images are in gallery order.
        seen = set()
        gallery = []
        for u in picked:
            clean = re.sub(r"\?.*$", "", u)
            if clean not in seen:
                seen.add(clean)
                gallery.append(u)
        gallery.sort(key=lambda u: int(re.search(r"_(\d+)\.jpg", u).group(1))
                     if re.search(r"_(\d+)\.jpg", u) else 999)

        return {
            "mileage": mileage_raw,
            "mileage_km": mileage_km,
            "gallery_image_urls": gallery,
        }

    # -----------------------------------------------------------------------
    # STEP 3 — image download
    # -----------------------------------------------------------------------
    def download_image(self, url: str, ad_id: str, idx: int) -> Optional[dict]:
        """Download one image. Returns {local_path, source_url} or None."""
        ext = ".jpg"
        fname = f"{ad_id}_{idx:02d}{ext}"
        dest = os.path.join(self.images_dir, fname)
        # Skip if already downloaded (idempotent re-runs)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return {"local_path": dest, "source_url": url}
        data = self._get(url, referer=DETAIL_URL_TPL.format(cid=ad_id), binary=True)
        self._sleep()
        if not data:
            return None
        # Verify it is a decodable image (not an HTML error page).
        try:
            import cv2
            import numpy as np
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                self._log(f"    [skip] not an image: {url}")
                return None
        except Exception:
            # Fallback: trust magic bytes if cv2 import failed for some reason
            if not (data.startswith(b"\xff\xd8") or data.startswith(b"\x89PNG")):
                return None
        with open(dest, "wb") as f:
            f.write(data)
        return {"local_path": dest, "source_url": url}

    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------
    def run(self) -> list[CarRecord]:
        from classify import classify_car  # local import to keep module decoupled

        self._log(f"=== TaladRod scraper starting (max_ads={self.max_ads}) ===")
        raw_cars = self.fetch_listing_cars()
        self._log(f"[listing] {len(raw_cars)} unique cars collected from listing pages")

        # SELECTION: first max_ads in display order, skipping sold cars.
        chosen = []
        for c in raw_cars:
            if c.get("issold", "N") == "Y":
                continue
            chosen.append(c)
            if len(chosen) >= self.max_ads:
                break
        self._log(f"[select] {len(chosen)} cars selected (newest first, sold skipped)")

        records: list[CarRecord] = []
        for i, c in enumerate(chosen, 1):
            cid = c.get("cid", "")

            # Brand: the listing JSON populates `amake` only sporadically
            # (~1 car in ~200); `namemmt` is always present and begins with
            # the brand name (e.g. "BMW SERIES 3 ..."), so fall back to its
            # first token. `model` is reliably populated for every listing.
            namemmt = (c.get("namemmt") or "").strip()
            brand = (c.get("amake") or "").strip()
            if not brand and namemmt:
                brand = namemmt.split()[0]
            model = (c.get("amodel") or c.get("model") or "").strip()
            if not model and namemmt:
                # second token is usually the model when amodel/model missing
                parts = namemmt.split()
                model = parts[1] if len(parts) > 1 else ""

            self._log(f"[{i}/{len(chosen)}] cid={cid}  {brand} {model} {c.get('yr4','')}")

            detail = self.fetch_detail(cid, car=c)

            # Build the record from listing data + detail data.
            price_raw = c.get("prc") or c.get("sbaht") or ""
            price_baht = int(price_raw.replace(",", "")) if price_raw.replace(",", "").isdigit() else 0
            rec = CarRecord(
                ad_id=cid,
                brand=brand,
                model=model,
                year=(c.get("yr4") or "").strip(),
                price=price_raw,
                price_baht=price_baht,
                mileage=detail.get("mileage"),
                mileage_km=detail.get("mileage_km"),
                ad_source_url=DETAIL_URL_TPL.format(cid=cid),
                thumbnail_url=c.get("img", ""),
                title=(c.get("title") or "").strip(),
                gallery_image_urls=detail.get("gallery_image_urls", []),
            )

            # Classify car type.
            cls = classify_car(c)
            rec.category = cls.category
            rec.category_en = cls.category_en

            # Download gallery images.
            imgs = []
            for idx, img_url in enumerate(rec.gallery_image_urls, 1):
                got = self.download_image(img_url, cid, idx)
                if got:
                    imgs.append(got)
                    self._log(f"    [img] {idx}/{len(rec.gallery_image_urls)} -> {os.path.basename(got['local_path'])}")
            rec.images = imgs
            records.append(rec)

        self._log(f"=== Done: {len(records)} ads, "
                  f"{sum(len(r.images) for r in records)} images downloaded ===")
        return records


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="taladrod.com used-car scraper")
    ap.add_argument("--max-ads", type=int, default=100,
                    help="max number of ads to scrape per run (default 100)")
    ap.add_argument("--images-dir", default="images",
                    help="folder to save downloaded images (default 'images')")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="seconds to sleep between requests (default 1.0)")
    ap.add_argument("--out", default="cars_data.json",
                    help="output JSON file (default cars_data.json)")
    args = ap.parse_args()

    scraper = TaladRodScraper(max_ads=args.max_ads, rate_limit=args.rate,
                              images_dir=args.images_dir)
    records = scraper.run()

    # Minimal JSON dump here (run_pipeline.py does the richer version with
    # image quality scores). This CLI path is mainly a smoke test.
    out = []
    for r in records:
        out.append({
            "ad_id": r.ad_id, "brand": r.brand, "model": r.model, "year": r.year,
            "price": r.price, "price_baht": r.price_baht,
            "mileage": r.mileage, "mileage_km": r.mileage_km,
            "category": r.category, "category_en": r.category_en,
            "ad_source_url": r.ad_source_url,
            "images": r.images,
        })
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} ({len(out)} ads)")
