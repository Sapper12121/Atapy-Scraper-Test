"""
classify.py — Car type classification for taladrod.com listings.

============================================================================
APPROACH
============================================================================
Each listing JSON object carries an `abody` field (body style, e.g.
"DOUBLE CAB") but it is frequently empty on taladrod.com. We therefore
combine three signals in priority order and fall back to a keyword map over
the model / full-name text:

  1. `abody`  (when present)  — the site's own body-style label.
  2. `amodel` / `model`      — the model name (e.g. "TRITON", "ALPHARD").
  3. `namemmt`               — the full brand+model+trim string.

We match against a curated keyword table of body styles and well-known Thai
market model names, and emit a Thai category label as requested by the
assignment (เก๋ง / กระบะ / รถตู้ / SUV / แฮทช์แบ็ค / คูเป้ / อื่นๆ).

The classifier is deterministic, fast, and easy to audit/extend. It is
intentionally rule-based rather than ML-based: the volume here (<=100 ads/run)
does not justify training a model, and rules are fully explainable to the
data team.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Thai category constants
CAT_SEDAN = "เก๋ง"
CAT_PICKUP = "กระบะ"
CAT_VAN = "รถตู้"
CAT_SUV = "SUV"
CAT_HATCHBACK = "แฮทช์แบ็ค"
CAT_COUPE = "คูเป้"
CAT_EV = "รถไฟฟ้า"
CAT_OTHER = "อื่นๆ"


# Each rule: (priority, regex, category, english_label).
# Lower priority number = checked first (higher precedence).
# Regexes are case-insensitive and word-boundary aware where it matters.
_RULES = [
    # --- Pickup: body label first, then well-known pickup model names ---
    (1, r"\b(double\s*cab|single\s*cab|cab\s*4|cab\b|pickup)\b", CAT_PICKUP, "Pickup"),
    (1, r"\b(triton|hilux|hilux\s*revo|revo|navara|d-max|dmax|isuzu\s*d|max|ranger|mighty|tiggr?er\s*8|colt\s*l)\b", CAT_PICKUP, "Pickup"),

    # --- Van / MPV ---
    (2, r"\b(van|mpv|minivan|commuter)\b", CAT_VAN, "Van/MPV"),
    (2, r"\b(alphard|voxy|noah|hiace|starex|elgrand|odyssey|previa|serena|grandbird|mazda\s*8|sienta|staria|delica|emgrand\s*9)\b", CAT_VAN, "Van/MPV"),

    # --- SUV / Crossover ---
    (3, r"\b(suv|crossover)\b", CAT_SUV, "SUV"),
    (3, r"\b(fortuner|everest|mu-?x|pajero|cx-?3|cx-?5|cx-?8|cx-?9|cx-?30|cx-?50|hr-?v|cr-?v|br-?v|eco\s*sport|rush|x-?trail|terra|captiva|tiggo|cs\d+|atoz|mg\s*zs|mg\s*hs|hs\b|zs\b|byd\s*atto|netta\s*v|gac\s*gn\d|harrier|fortuner|ruvi\d?)\b", CAT_SUV, "SUV"),

    # --- Hatchback ---
    (4, r"\b(hatchback|hb\b)\b", CAT_HATCHBACK, "Hatchback"),
    (4, r"\b(jazz|mazda2|yaris|swift|civic\s*hatch|i10|picanto|spark|mirage|igo|vios\s*hatch|celerio|alto|figo|march|colt|note|fabia|polo)\b", CAT_HATCHBACK, "Hatchback"),

    # --- Coupe ---
    (5, r"\b(coupe|cabriolet|convertible|roadster)\b", CAT_COUPE, "Coupe"),

    # --- Sedan (explicit) ---
    (6, r"\b(sedan|saloon)\b", CAT_SEDAN, "Sedan"),
    # Well-known sedan model defaults
    (6, r"\b(civic|corolla|altis|vios|city|accord|camry|mazda3|mazda6|bmw\s*\d|mercedes|c-class|e-class|a-class|altis|sylphy|teana|creta|avante|cerato|c5\b|catlca|model\s*3)\b", CAT_SEDAN, "Sedan"),
]


@dataclass
class Classification:
    category: str       # Thai label
    category_en: str    # English label
    matched_by: str     # which field/rule decided it


def _match(text: str):
    """Return (category, category_en, matched_rule) for the first matching rule."""
    if not text:
        return None
    t = text.lower()
    for prio, pattern, cat, cat_en in _RULES:
        if re.search(pattern, t):
            return cat, cat_en, pattern
    return None


def classify_car(car: dict) -> Classification:
    """Classify a single listing dict.

    `car` is one element of SchDataJSON["cars"]. Uses abody -> namemmt ->
    amodel/model in priority order; falls back to 'Sedan' (เก๋ง) because the
    overwhelming majority of used-car listings on taladrod.com are sedans,
    then 'Other' if even that guess is unsupported.
    """
    abody = (car.get("abody") or "").strip()
    namemmt = (car.get("namemmt") or "").strip()
    amodel = (car.get("amodel") or car.get("model") or "").strip()

    # Priority 1: explicit body label
    for field_name, field_val in (("abody", abody), ("namemmt", namemmt), ("amodel", amodel)):
        m = _match(field_val)
        if m:
            cat, cat_en, rule = m
            return Classification(category=cat, category_en=cat_en,
                                  matched_by=f"{field_name}:{rule}")

    # Default fallback: sedan is the most common body style on the site.
    return Classification(category=CAT_SEDAN, category_en="Sedan",
                          matched_by="default")


if __name__ == "__main__":
    # Smoke test on representative samples.
    samples = [
        {"abody": "DOUBLE CAB", "amodel": "TRITON 18-23", "namemmt": "MITSUBISHI TRITON 2.4 GLS 4WD DOUBLE CAB"},
        {"abody": "", "amodel": "ALPHARD", "namemmt": "TOYOTA ALPHARD 2.5 HYBRID"},
        {"abody": "", "amodel": "MARCH", "namemmt": "NISSAN MARCH 1.2 EL"},
        {"abody": "", "amodel": "CX-8", "namemmt": "MAZDA CX-8 2.5 SP EXCLUSIVE"},
        {"abody": "", "amodel": "HILUX REVO", "namemmt": "TOYOTA HILUX REVO 2.4 4WD"},
        {"abody": "", "amodel": "JAZZ", "namemmt": "HONDA JAZZ 1.5"},
        {"abody": "", "amodel": "CIVIC", "namemmt": "HONDA CIVIC 1.8 EL"},
        {"abody": "", "amodel": "BYD ATTO 3", "namemmt": "BYD ATTO 3 ELECTRIC"},
    ]
    for s in samples:
        r = classify_car(s)
        print(f"{r.category:8s} ({r.category_en:10s})  <-  {s['amodel']:18s} matched_by={r.matched_by}")
