"""
run_pipeline.py — End-to-end pipeline for the taladrod.com assignment.

Pipeline:
  1. Scrape  (scraper.TaladRodScraper.run)
       listing page -> detail pages -> download all gallery images
  2. Score   (image_quality.assess_image)
       run the local Classical-CV IQA scorer on every downloaded image
  3. Export  -> cars_data.json  (the deliverable data file)

The output JSON schema (per the assignment, Part 4):
  [
    {
      "ad_id": "2934539",
      "brand": "MITSUBISHI",
      "model": "TRITON 18-23",
      "year": "2023",
      "model_name": "MITSUBISHI TRITON 18-23 2.4 GLS 4WD DOUBLE C",
      "category": "กระบะ",            <-- Thai car-type label
      "category_en": "Pickup",
      "price": "550,000",
      "price_baht": 550000,
      "mileage": "72,800",
      "mileage_km": 72800,
      "ad_source_url": "https://www.taladrod.com/w40/iCar/CarDet.aspx?cid=2934539",
      "thumbnail_url": "...",
      "title": "...",
      "image_count": 9,
      "images": [
        {
          "local_path": "images/2934539_01.jpg",
          "source_url": "https://imgc2.taladrod.com/c/cidx/014/106/66_1.jpg?v=...",
          "quality_score": 8.5,           <-- 0-10, local AI score
          "quality": {
             "sharpness": 9.7, "brightness": 9.3, "contrast": 8.1,
             "colorfulness": 6.9, "saturation": 8.6, "clipping_penalty": 1.0,
             "width": 600, "height": 337, "note": "balanced"
          }
        },
        ...
      ]
    },
    ...
  ]

Run:
  python run_pipeline.py                  # scrape 100 ads (default)
  python run_pipeline.py --max-ads 20     # smaller smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

from scraper import TaladRodScraper, CarRecord
from image_quality import assess_image


def build_output_record(rec: CarRecord, images_dir_rel: str) -> dict:
    """Attach quality scores to each image and build the final JSON dict."""
    scored_images = []
    for img in rec.images:
        local_path = img["local_path"]
        entry = {
            "local_path": os.path.relpath(local_path),  # relative to CWD/output dir
            "source_url": img["source_url"],
        }
        try:
            res = assess_image(local_path)
            entry["quality_score"] = res.score
            entry["quality"] = {
                "sharpness": res.sharpness,
                "brightness": res.brightness,
                "contrast": res.contrast,
                "colorfulness": res.colorfulness,
                "saturation": res.saturation,
                "clipping_penalty": res.clipping_penalty,
                "width": res.width,
                "height": res.height,
                "note": res.note,
            }
        except Exception as e:
            entry["quality_score"] = None
            entry["quality"] = {"error": f"{type(e).__name__}: {e}"}
        scored_images.append(entry)

    return {
        "ad_id": rec.ad_id,
        "brand": rec.brand,
        "model": rec.model,
        "year": rec.year,
        "model_name": f"{rec.brand} {rec.model} {rec.year}".strip(),
        "namemmt_full": f"{rec.brand} {rec.model}".strip(),
        "category": rec.category,
        "category_en": rec.category_en,
        "price": rec.price,
        "price_baht": rec.price_baht,
        "mileage": rec.mileage,
        "mileage_km": rec.mileage_km,
        "ad_source_url": rec.ad_source_url,
        "thumbnail_url": rec.thumbnail_url,
        "title": rec.title,
        "image_count": len(scored_images),
        "images": scored_images,
    }


def main():
    ap = argparse.ArgumentParser(description="taladrod.com scrape + IQA pipeline")
    ap.add_argument("--max-ads", type=int, default=100,
                    help="max ads per run (assignment cap: 100)")
    ap.add_argument("--images-dir", default="images",
                    help="folder for downloaded images")
    ap.add_argument("--out", default="cars_data.json",
                    help="output JSON file")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="seconds between HTTP requests")
    ap.add_argument("--no-scrape", action="store_true",
                    help="skip scraping; re-score an existing image folder "
                         "using cars_data.json's image list (not used by default)")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    images_dir_abs = args.images_dir if os.path.isabs(args.images_dir) \
        else os.path.join(out_dir, args.images_dir)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    scraper = TaladRodScraper(
        max_ads=args.max_ads,
        rate_limit=args.rate,
        images_dir=images_dir_abs,
    )
    records = scraper.run()

    output = [build_output_record(r, args.images_dir) for r in records]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Summary
    total_imgs = sum(len(r["images"]) for r in output)
    scored = [im["quality_score"] for r in output for im in r["images"]
              if im.get("quality_score") is not None]
    avg = sum(scored) / len(scored) if scored else 0.0
    cats = {}
    for r in output:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print("\n=== SUMMARY ===")
    print(f"ads scraped       : {len(output)}")
    print(f"images downloaded : {total_imgs}")
    print(f"images scored     : {len(scored)}")
    print(f"avg quality score : {avg:.2f} / 10")
    print(f"category breakdown: {cats}")
    print(f"output JSON       : {args.out}")
    print(f"elapsed           : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
