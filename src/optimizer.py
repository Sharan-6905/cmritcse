"""Recycling optimizer: deterministic maths over the items the agent identified.

Compares two scenarios for the same set of items:
  as-is  - items go in the bins as they are (contaminated recyclables get rejected)
  fixed  - the user follows the recommended prep steps (rinse, separate lid, tear off greasy part)
and reports how much material is recovered (kept out of general waste) and the CO2 avoided.
Weights default to typical per-item values and are estimates unless the user stated them.
"""
from pydantic import BaseModel

from . import knowledge

DRY_RECYCLABLES = {"cardboard", "glass", "metal", "paper", "plastic"}
SALVAGE_WHEN_HEAVY = {"cardboard": 0.5, "paper": 0.5}  # tear off the clean part
TYPICAL_WEIGHT_G = {"cardboard": 150, "glass": 300, "metal": 20, "paper": 20, "plastic": 25,
                    "trash": 50, "organic": 100, "battery": 25, "e-waste": 300, "textile": 250, "hazardous": 300}


class PlanItem(BaseModel):
    name: str
    category: str          # cardboard, glass, metal, paper, plastic, trash, organic, battery, e-waste, textile, hazardous
    contamination: str     # none, light (food residue, liquid), heavy (grease-soaked, wet, mixed materials)
    weight_g: float = 0    # 0 = unknown, use the typical weight


def _recovered_fraction(category: str, contamination: str, fixed: bool) -> float:
    if category == "trash":
        return 0.0
    if category not in DRY_RECYCLABLES or contamination == "none":
        return 1.0
    if contamination == "light":
        return 1.0 if fixed else 0.0
    return SALVAGE_WHEN_HEAVY.get(category, 0.0) if fixed else 0.0


def _action(item: PlanItem) -> str | None:
    if item.category not in DRY_RECYCLABLES or item.contamination == "none":
        return None
    if item.contamination == "light":
        return f"{item.name}: rinse off food/liquid residue and drain it before binning"
    if item.category in SALVAGE_WHEN_HEAVY:
        return f"{item.name}: tear off the clean part to recycle; the soiled part goes in general waste"
    return f"{item.name}: heavily contaminated, cannot be recycled; put in general waste"


def optimize(items: list[PlanItem]) -> dict:
    rows, bins, as_is_g, fixed_g, total_g, co2_as_is, co2_fixed = [], {}, 0.0, 0.0, 0.0, 0.0, 0.0
    for it in items:
        cat = it.category.lower().strip()
        it = it.model_copy(update={"category": cat, "contamination": it.contamination.lower().strip()})
        weight = it.weight_g if it.weight_g > 0 else TYPICAL_WEIGHT_G.get(cat, 50)
        a, f = _recovered_fraction(cat, it.contamination, False), _recovered_fraction(cat, it.contamination, True)
        factor = knowledge.CO2_PER_KG.get(cat, 0.0)
        total_g += weight; as_is_g += weight * a; fixed_g += weight * f
        co2_as_is += factor * weight / 1000 * a; co2_fixed += factor * weight / 1000 * f
        bin_name = knowledge.GUIDES.get(cat, {}).get("bin", "General waste")
        bins.setdefault(bin_name, []).append(it.name)
        rows.append({"item": it.name, "category": cat, "contamination": it.contamination, "weight_g": round(weight),
                     "weight_source": "stated/estimated" if it.weight_g > 0 else "typical value",
                     "recovered_as_is": round(a, 2), "recovered_after_prep": round(f, 2)})
    pct = lambda x: round(100 * x / total_g, 1) if total_g else 0.0
    return {
        "bins": bins,
        "items": rows,
        "total_weight_g": round(total_g),
        "recovered_share_as_is_pct": pct(as_is_g),
        "recovered_share_after_prep_pct": pct(fixed_g),
        "improvement_pct_points": round(pct(fixed_g) - pct(as_is_g), 1),
        "co2_avoided_kg_as_is": round(co2_as_is, 3),
        "co2_avoided_kg_after_prep": round(co2_fixed, 3),
        "prep_actions": [a for a in map(_action, items) if a],
        "note": "Weights are typical values unless the user stated them; CO2 figures are rough estimates.",
    }
