# taladrod.com Used-Car Scraper + Local Image Quality Assessment

Solution to the **AI Data Engineer (Data Collection & Analysis)** assignment:
scrape used-car listings from [taladrod.com](https://www.taladrod.com/w40/isch/schc.aspx?fno:new),
categorize each car, download every listing photo, and score each photo's
quality with a **fully local** image-quality model (no external LLM / Vision
API).

---

## 1. Deliverables (Part 4 of the assignment)

| Deliverable | Location |
| --- | --- |
| **Source code** | `scraper.py`, `classify.py`, `image_quality.py`, `run_pipeline.py`, `finalize.py`, `verify.py` |
| **Image folder** | `images/`  (every photo downloaded from every ad) |
| **JSON data file** | `cars_data.json` |

Each ad in `cars_data.json` contains at minimum: `ad_id`, `model`/`model_name`,
`category` (Thai car-type label), `ad_source_url`, and an `images` list where
each image has `local_path`, `source_url`, and `quality_score`.

---

## 2. How to run

```bash
pip install -r requirements.txt

# full run: scrape up to 100 ads, download all photos, score them, write JSON
python run_pipeline.py --max-ads 100 --images-dir images --out cars_data.json

# smaller smoke run
python run_pipeline.py --max-ads 20
```

Options:

| flag | default | meaning |
| --- | --- | --- |
| `--max-ads` | `100` | max ads per run (assignment cap is 100) |
| `--images-dir` | `images` | folder for downloaded photos |
| `--out` | `cars_data.json` | output JSON file |
| `--rate` | `1.0` | seconds to sleep between HTTP requests (politeness) |

Re-runs are idempotent: an image already on disk is not re-downloaded, so you
can resume after an interruption by just running again.

---

## 3. Architecture

```
run_pipeline.py  ── orchestrates ──┐
                                   ├── scraper.py        (data collection)
                                   ├── classify.py       (car-type categorization)
                                   └── image_quality.py  (local AI image scoring)
                                   └──> cars_data.json + images/
```

* `scraper.py` — `TaladRodScraper` class: HTTP session with retry/backoff,
  listing-page parsing, detail-page parsing, image download.
* `classify.py` — rule-based car-type classifier returning a Thai label.
* `image_quality.py` — Hybrid Classical-CV IQA scorer returning 0–10.
* `run_pipeline.py` — runs the three above in order and writes the JSON.
* `finalize.py` — recovery helper: if a long run is interrupted/throttled
  mid-way, it rebuilds `cars_data.json` from the listing JSON already on disk
  and re-fetches only the missing detail pages, so a partial run is salvageable
  instead of being thrown away.
* `verify.py` — post-run sanity check: confirms the required fields are
  present, every image file exists on disk, every image has a quality score,
  and reports the category breakdown and score range.

---

## 4. Part 1 — Data collection (scraper.py)

### 4.1 Starting point & fields collected
The listing page `schc.aspx?fno:new` embeds the entire result set as a
JavaScript variable `SchDataJSON = {"cars":[...], "ncar":"209", ...}` directly
in the HTML. We parse that JSON (no headless browser needed) and read, per ad:

* `ad_id`  ← `cid`
* `brand`  ← `amake`
* `model`  ← `amodel` / `model`
* `year`   ← `yr4`
* `price`  ← `prc` / `sbaht`
* `thumbnail_url` ← `img`
* `ad_source_url` ← `https://www.taladrod.com/w40/iCar/CarDet.aspx?cid=<cid>`

The detail page (`CarDet.aspx?cid=<cid>`, which 302-redirects to the lowercase
`cardet.aspx` form — we follow redirects) supplies the two fields the listing
JSON omits or only puts in the title:

* **mileage** — regex on the Thai label `ไมล์ <number> กม.` (e.g. `72,800`)
* **gallery images** — every `imgc2.taladrod.com/c/cidx/.../cN.jpg` `src`
  found on the detail page (the full-size variants; the listing only carries
  the `_1T3.jpg` thumbnail).

### 4.2 Image collection (1.2)
Every gallery image on the detail page is downloaded to `images/<ad_id>_<NN>.jpg`,
and **both** the local path and the original source URL are recorded per image
in the JSON. Ads commonly have 5–10+ photos; there is no per-ad cap.

### 4.3 Stability (1.3)
* **Retry** — `urllib3.Retry` on the requests `Session`: 3 attempts, backoff
  factor 0.6, retries on 429/500/502/503/504 and connection errors. A second
  manual retry loop wraps each request for robustness.
* **Rate limiting** — configurable sleep (`--rate`, default 1.0s) with
  ±0.3s jitter between every request, plus a browser-like `User-Agent` and a
  `Referer` header on detail/image requests, to avoid being blocked.
* **Pagination** — `fetch_listing_cars()` loops pages: it parses the cars on
  each page, looks for a "next page" link, and stops when a page returns **no
  new car ids** (data exhausted) or when `max_ads` unique cars are collected
  or a 20-page safety cap is hit. For `fno:new` the site returns *all* new
  listings on page 1, so the loop correctly terminates after one fetch.

### 4.4 Selection method (required to be stated)
We take the **first `max_ads` cars in the order the website displays them**
(`fno:new` = newest listings first) and skip any car already marked sold
(`issold:"Y"`). This is the "เรียงตามที่เว็บแสดงผล" option the assignment
explicitly allows.

---

## 5. Part 2 — Car-type categorization (classify.py)

Each listing's JSON carries an `abody` field (body style, e.g. `"DOUBLE CAB"`)
but it is frequently empty, so we combine three signals in priority order:

1. `abody` (when present)
2. `namemmt` (full brand+model+trim string)
3. `amodel` / `model`

…and match against a curated keyword table of body styles and well-known Thai
market model names. Output is a **Thai category label** as requested:

`เก๋ง` (Sedan) · `กระบะ` (Pickup) · `รถตู้` (Van/MPV) · `SUV` · `แฮทช์แบ็ค` (Hatchback) · `คูเป้` (Coupe) · `อื่นๆ` (Other)

The classifier is deterministic, fast, and fully explainable (`matched_by`
records which field/rule decided each classification). At <=100 ads/run a
rule-based approach is more appropriate than training an ML model.

---

## 6. Part 3 — Image quality assessment (image_quality.py)

### Chosen approach: **Hybrid Classical CV IQA** (local, no external API)

**Why this and not NIMA / MUSIQ / CLIP-IQA?** The assignment forbids external
LLM/Vision APIs (GPT-4V, Claude Vision, Gemini Vision). Deep-learning IQA
models (NIMA, MUSIQ, HyperIQA, MANIQA) and CLIP-IQA are *local* and therefore
allowed, but they require PyTorch + pretrained weight files (tens–hundreds of
MB) downloaded from the internet, which makes the pipeline slow and fragile
when batch-scoring hundreds of photos and ties it to a weight download that
may fail. Classical CV metrics, by contrast, run instantly with OpenCV /
scikit-image, need **no weights**, and every sub-score is explainable and maps
to a factor a human reviewer cares about for a listing photo. This also
matches the assignment's **Hybrid** row: we compute several complementary CV
metrics and combine them.

### Sub-metrics (each normalized to 0–10, 10 = best)
| metric | what it measures | method |
| --- | --- | --- |
| `sharpness` | blur / edge crispness | variance of the Laplacian |
| `brightness` | exposure (too dark / too bright) | mean luminance vs ideal ~128 |
| `contrast` | tonal range / dynamic range | std-dev of luminance |
| `colorfulness` | colour richness | Hasler & Süsstrunk (2003) metric |
| `saturation` | vividness vs washed-out | mean saturation in HSV |
| `clipping_penalty` | blown highlights / crushed shadows | ratio of pure-black/white pixels |

### Final score
A weighted blend of the five sub-scores (sharpness 0.40, brightness 0.25,
contrast 0.15, colorfulness 0.12, saturation 0.08) multiplied by the clipping
penalty, clamped to **0.0–10.0** and rounded to one decimal (e.g. `8.5`).
Sharpness is weighted highest because a blurry photo is the single most
disqualifying flaw for a car listing. Each image's full sub-score breakdown
and a short `note` (e.g. `"low: sharpness"`) are kept in the JSON so a low
score is always diagnosable.

---

## 7. Output JSON schema

```jsonc
[
  {
    "ad_id": "2934539",
    "brand": "MITSUBISHI",
    "model": "TRITON 18-23",
    "year": "2023",
    "model_name": "MITSUBISHI TRITON 18-23 2023",
    "category": "กระบะ",
    "category_en": "Pickup",
    "price": "550,000",
    "price_baht": 550000,
    "mileage": "72,800",
    "mileage_km": 72800,
    "ad_source_url": "https://www.taladrod.com/w40/iCar/CarDet.aspx?cid=2934539",
    "thumbnail_url": "https://imgc2.taladrod.com/c/cidx/014/106/66_1T3.jpg?v=...",
    "title": "รถบ้าน เจ้าของขายเอง ...",
    "image_count": 9,
    "images": [
      {
        "local_path": "images/2934539_01.jpg",
        "source_url": "https://imgc2.taladrod.com/c/cidx/014/106/66_1.jpg?v=...",
        "quality_score": 8.5,
        "quality": {
          "sharpness": 9.7, "brightness": 9.3, "contrast": 8.1,
          "colorfulness": 6.9, "saturation": 8.6, "clipping_penalty": 1.0,
          "width": 600, "height": 337, "note": "balanced"
        }
      }
    ]
  }
]
```

---

## 8. Notes & assumptions

* **No headless browser is used.** The listing data is embedded as JSON in
  the HTML, so `requests` + regex parsing is sufficient and far faster/cheaper
  than driving a browser.
* **Rate limiting is intentionally polite** (default 1 req/s) — scraping 100
  ads with ~8 photos each takes roughly 15–20 minutes. Lower `--rate` to go
  faster at your own risk of being throttled.
* **The IQA model is intentionally classical**, per the reasoning in §6. If
  the team later wants a human-judgement-correlated score, the same interface
  (`assess_image(path) -> QualityResult`) lets you swap in a local NIMA/MUSIQ
  model behind the same weighted fusion without touching the rest of the code.
* All text (Thai category labels, etc.) is preserved as UTF-8 in the JSON.
