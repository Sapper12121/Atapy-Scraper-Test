"""verify.py — post-run verification of cars_data.json + images/."""
import json
import os
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "cars_data.json"), encoding="utf-8"))
print("=== ADS:", len(d), "===")

req = ["ad_id", "model", "category", "ad_source_url", "images"]
missing = {f: sum(1 for r in d if not r.get(f)) for f in req}
print("missing/empty required fields:", missing)

total_imgs = imgs_no_score = imgs_missing_file = imgs_no_src = 0
for r in d:
    for im in r["images"]:
        total_imgs += 1
        if im.get("quality_score") is None:
            imgs_no_score += 1
        if not im.get("source_url"):
            imgs_no_src += 1
        if not os.path.exists(os.path.join(HERE, im["local_path"])):
            imgs_missing_file += 1
print(f"total images            : {total_imgs}")
print(f"images without score    : {imgs_no_score}")
print(f"images without source   : {imgs_no_src}")
print(f"images w/ missing file  : {imgs_missing_file}")

cats = collections.Counter(r["category"] for r in d)
print("category breakdown      :", dict(cats))

mile = sum(1 for r in d if r.get("mileage_km"))
print(f"ads with mileage        : {mile}/{len(d)}")

text = json.dumps(d, ensure_ascii=False)
placeholders = ["XXX", "TODO", "待补充", "【", "undefined", "NaN"]
hits = {p: text.count(p) for p in placeholders if text.count(p)}
print("placeholder hits        :", hits if hits else "clean")

empty_brand = sum(1 for r in d if not r.get("brand"))
empty_model = sum(1 for r in d if not r.get("model"))
print(f"empty brand / model     : {empty_brand} / {empty_model}")

scores = [im["quality_score"] for r in d for im in r["images"]
          if im.get("quality_score") is not None]
print(f"score range             : {min(scores):.1f} - {max(scores):.1f}, "
      f"avg {sum(scores)/len(scores):.2f}")

# Show 3 sample records (compact)
print("\n=== SAMPLE RECORDS ===")
for r in d[:3]:
    print(f"  ad_id={r['ad_id']} | {r['model_name']} | {r['category']} | "
          f"price={r['price']} | mileage={r['mileage']} | imgs={r['image_count']} "
          f"| top_score={max(im['quality_score'] for im in r['images'])}")
