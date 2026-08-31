"""
SPIRO ML — GuidanceEngine
Maps every detected waste item to one of 7 disposal streams and
generates actionable disposal guidance.

7 Disposal Streams
------------------
organic         — compost / green bin
recoverable     — recycling bin
non_recoverable — general waste / black bin
hazardous       — hazardous waste collection point
sanitary        — sealed sanitary / clinical bin
e_waste         — e-waste collection point
reject          — unknown / manual inspection

All 109 SPIRO class IDs are mapped here. No class may be unmapped.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

# =============================================================================
# Stream definitions
# =============================================================================

STREAMS: Dict[str, Dict[str, str]] = {
    "organic": {
        "label":       "Organic Waste",
        "description": "Biodegradable food, garden, and natural waste",
        "bin":         "Green / compost bin",
        "color":       "#4CAF50",
        "icon":        "leaf",
    },
    "recoverable": {
        "label":       "Recoverable Waste",
        "description": "Clean recyclable materials — plastic, glass, metal, paper",
        "bin":         "Blue / recycling bin",
        "color":       "#2196F3",
        "icon":        "recycle",
    },
    "non_recoverable": {
        "label":       "Non-Recoverable Waste",
        "description": "Contaminated, composite, or residual waste",
        "bin":         "Black / general waste bin",
        "color":       "#FF9800",
        "icon":        "trash",
    },
    "hazardous": {
        "label":       "Hazardous Waste",
        "description": "Dangerous chemicals, flammables, aerosols",
        "bin":         "Hazardous waste collection point",
        "color":       "#F44336",
        "icon":        "warning",
    },
    "sanitary": {
        "label":       "Sanitary Waste",
        "description": "Medical, hygiene, and bodily waste",
        "bin":         "Sealed sanitary / clinical bin",
        "color":       "#9C27B0",
        "icon":        "medical",
    },
    "e_waste": {
        "label":       "E-Waste",
        "description": "Electronics and batteries",
        "bin":         "E-waste collection point",
        "color":       "#607D8B",
        "icon":        "electrical",
    },
    "reject": {
        "label":       "Reject / Unknown",
        "description": "Cannot be classified — manual inspection required",
        "bin":         "Set aside for manual inspection",
        "color":       "#9E9E9E",
        "icon":        "question",
    },
}

# =============================================================================
# Master class-ID → stream mapping  (all 109 IDs, no gaps)
# =============================================================================

_CLASS_TO_STREAM: Dict[int, str] = {
    # ── PLASTIC (recoverable unless contaminated/non-recyclable form) ─────────
    0:  "recoverable",   # plastic_bottle
    1:  "recoverable",   # plastic_bottle_cap
    2:  "recoverable",   # plastic_bag
    3:  "recoverable",   # plastic_film
    4:  "recoverable",   # plastic_straw
    5:  "recoverable",   # plastic_cup
    6:  "recoverable",   # plastic_lid
    7:  "recoverable",   # plastic_cutlery_fork
    8:  "recoverable",   # plastic_cutlery_knife
    9:  "recoverable",   # plastic_cutlery_spoon
    10: "recoverable",   # plastic_container
    11: "recoverable",   # plastic_tray
    12: "recoverable",   # plastic_packaging
    13: "recoverable",   # plastic_wrap
    14: "recoverable",   # plastic_tube
    15: "non_recoverable",  # plastic_rope (mixed/contaminated)
    16: "non_recoverable",  # plastic_toy (mixed materials)
    17: "non_recoverable",  # plastic_hanger
    18: "recoverable",   # plastic_bucket
    19: "non_recoverable",  # polystyrene_cup (not accepted most recycling)
    20: "non_recoverable",  # polystyrene_container
    21: "non_recoverable",  # polystyrene_packaging
    # ── GLASS ─────────────────────────────────────────────────────────────────
    22: "recoverable",   # glass_bottle
    23: "recoverable",   # glass_jar
    24: "non_recoverable",  # glass_fragment (broken — safety risk in recycling)
    25: "recoverable",   # glass_cup
    26: "non_recoverable",  # glass_window_fragment (different glass type)
    # ── METAL ─────────────────────────────────────────────────────────────────
    27: "recoverable",   # metal_can_beverage
    28: "recoverable",   # metal_can_food
    29: "recoverable",   # metal_bottle_cap
    30: "recoverable",   # metal_lid
    31: "recoverable",   # metal_foil
    32: "non_recoverable",  # metal_scrap (sharp/contaminated)
    33: "non_recoverable",  # metal_wire (sharp hazard)
    34: "hazardous",     # aerosol_can (pressurised — hazardous)
    35: "recoverable",   # metal_tin
    # ── PAPER & CARDBOARD ────────────────────────────────────────────────────
    36: "recoverable",   # cardboard_box
    37: "recoverable",   # cardboard_fragment
    38: "recoverable",   # paper_bag
    39: "non_recoverable",  # paper_cup (wax/plastic lined)
    40: "recoverable",   # newspaper
    41: "recoverable",   # magazine
    42: "recoverable",   # paper_packaging
    43: "non_recoverable",  # tissue_paper (contaminated)
    44: "recoverable",   # paper_straw
    45: "recoverable",   # receipt
    46: "recoverable",   # carton_drink (tetra pak)
    # ── ORGANIC ───────────────────────────────────────────────────────────────
    47: "organic",       # food_waste_generic
    48: "organic",       # fruit_peel
    49: "organic",       # vegetable_scraps
    50: "organic",       # eggshell
    51: "organic",       # coffee_grounds
    52: "non_recoverable",  # food_packaging_contaminated
    # ── CIGARETTE / TOBACCO ───────────────────────────────────────────────────
    53: "non_recoverable",  # cigarette_butt (toxic residue)
    54: "non_recoverable",  # cigarette_pack
    55: "non_recoverable",  # lighter (residual fuel — if empty, else hazardous)
    # ── TEXTILE ───────────────────────────────────────────────────────────────
    56: "non_recoverable",  # clothing_item (donate if clean, else general)
    57: "non_recoverable",  # shoe
    58: "non_recoverable",  # textile_scrap
    59: "sanitary",      # mask_disposable
    60: "sanitary",      # glove_disposable
    # ── ELECTRONIC WASTE ──────────────────────────────────────────────────────
    61: "e_waste",       # battery
    62: "e_waste",       # cable_wire
    63: "e_waste",       # circuit_board
    64: "e_waste",       # mobile_phone
    65: "e_waste",       # earphone
    # ── MEDICAL / SANITARY ────────────────────────────────────────────────────
    66: "sanitary",      # syringe
    67: "sanitary",      # medicine_blister
    68: "sanitary",      # bandage
    69: "sanitary",      # sanitary_pad
    70: "sanitary",      # diaper
    # ── BEVERAGE-SPECIFIC ─────────────────────────────────────────────────────
    71: "recoverable",   # coffee_pod (metal/plastic — check local rules)
    72: "organic",       # tea_bag (compostable)
    # ── CONSTRUCTION ──────────────────────────────────────────────────────────
    73: "non_recoverable",  # brick_fragment
    74: "non_recoverable",  # concrete_fragment
    75: "hazardous",     # paint_can
    76: "recoverable",   # wood_scrap (clean untreated)
    # ── AUTOMOTIVE ────────────────────────────────────────────────────────────
    77: "non_recoverable",  # tire (specialist recycling)
    78: "non_recoverable",  # car_part_generic
    # ── BIODEGRADABLE / GARDEN ────────────────────────────────────────────────
    79: "organic",       # leaf_litter
    80: "organic",       # twig_branch
    81: "organic",       # soil_bag
    # ── HAZARDOUS ─────────────────────────────────────────────────────────────
    82: "hazardous",     # chemical_container
    83: "hazardous",     # oil_container
    84: "hazardous",     # propane_tank_small
    # ── LITTER / SMALL ────────────────────────────────────────────────────────
    85: "non_recoverable",  # bottle_label
    86: "non_recoverable",  # rubber_band
    87: "non_recoverable",  # tape
    88: "non_recoverable",  # sticker_label
    89: "non_recoverable",  # zip_tie
    90: "non_recoverable",  # paper_clip
    91: "non_recoverable",  # pen_marker
    92: "non_recoverable",  # broken_glass_mirror
    # ── SPORTS / RECREATION ───────────────────────────────────────────────────
    93: "non_recoverable",  # balloon
    94: "non_recoverable",  # fishing_line
    95: "non_recoverable",  # fishing_net_fragment
    # ── FAST FOOD ─────────────────────────────────────────────────────────────
    96: "non_recoverable",  # fast_food_container (contaminated)
    97: "non_recoverable",  # pizza_box (grease contaminated)
    98: "non_recoverable",  # sauce_packet
    99: "non_recoverable",  # condiment_container
    # ── DURABLE GOODS ─────────────────────────────────────────────────────────
   100: "non_recoverable",  # appliance_small (→ e-waste if has electronics)
   101: "non_recoverable",  # furniture_fragment
    # ── INFRASTRUCTURE ────────────────────────────────────────────────────────
   102: "non_recoverable",  # traffic_cone
   103: "recoverable",      # pallet (wood — recoverable if clean)
    # ── MULTI-MATERIAL ────────────────────────────────────────────────────────
   104: "non_recoverable",  # composite_packaging
   105: "non_recoverable",  # blister_pack
    # ── MISC ──────────────────────────────────────────────────────────────────
   106: "reject",           # unknown_litter
   107: "non_recoverable",  # wet_waste_generic
   108: "non_recoverable",  # dry_waste_generic
}

# Verify no gaps at module load time
_missing = set(range(109)) - set(_CLASS_TO_STREAM.keys())
if _missing:
    raise RuntimeError(f"GuidanceEngine: unmapped class IDs: {_missing}")

# =============================================================================
# Per-class disposal instructions
# =============================================================================

@dataclass
class ClassGuidance:
    class_id: int
    class_name: str
    stream: str
    stream_label: str
    bin_color: str
    summary: str
    preparation: List[str] = field(default_factory=list)
    warnings: List[str]    = field(default_factory=list)
    collection_point: str  = ""
    recyclable: bool       = False
    compostable: bool      = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Per-class preparation steps and warnings
_CLASS_DETAIL: Dict[int, Dict[str, Any]] = {
    # ── PLASTIC ───────────────────────────────────────────────────────────────
    0:  {"prep": ["Empty contents", "Replace cap", "Rinse if dirty", "Crush to save space"],
         "warn": [], "recyclable": True},
    1:  {"prep": ["Attach to bottle before recycling"], "warn": [], "recyclable": True},
    2:  {"prep": ["Shake out debris", "Tie loosely"], "warn": [], "recyclable": True},
    3:  {"prep": ["Bundle together", "Keep dry"], "warn": [], "recyclable": True},
    4:  {"prep": ["Place loose in recycling bin"], "warn": ["Check local rules — straws not accepted everywhere"], "recyclable": True},
    5:  {"prep": ["Remove lid", "Rinse"], "warn": [], "recyclable": True},
    6:  {"prep": ["Rinse if food residue present"], "warn": [], "recyclable": True},
    7:  {"prep": ["Place loose in recycling"], "warn": ["Check local rules"], "recyclable": True},
    8:  {"prep": ["Place loose in recycling"], "warn": ["Check local rules"], "recyclable": True},
    9:  {"prep": ["Place loose in recycling"], "warn": ["Check local rules"], "recyclable": True},
    10: {"prep": ["Rinse thoroughly", "Remove any food residue"], "warn": [], "recyclable": True},
    11: {"prep": ["Rinse", "Remove food residue"], "warn": [], "recyclable": True},
    12: {"prep": ["Flatten to save space"], "warn": [], "recyclable": True},
    13: {"prep": ["Bundle with other plastic film"], "warn": [], "recyclable": True},
    14: {"prep": ["Rinse tube", "Squeeze flat"], "warn": [], "recyclable": True},
    15: {"prep": ["Place in general waste"], "warn": ["Not recyclable — mixed materials"], "recyclable": False},
    16: {"prep": ["Place in general waste"], "warn": ["Mixed materials — not recyclable"], "recyclable": False},
    17: {"prep": ["Place in general waste"], "warn": ["Mixed materials"], "recyclable": False},
    18: {"prep": ["Rinse and recycle"], "warn": [], "recyclable": True},
    19: {"prep": ["Place in general waste"], "warn": ["Polystyrene not accepted in most recycling"], "recyclable": False},
    20: {"prep": ["Place in general waste"], "warn": ["Polystyrene not accepted in most recycling"], "recyclable": False},
    21: {"prep": ["Place in general waste"], "warn": ["Polystyrene not accepted in most recycling"], "recyclable": False},
    # ── GLASS ─────────────────────────────────────────────────────────────────
    22: {"prep": ["Rinse", "Remove cap (recycle separately)"], "warn": [], "recyclable": True},
    23: {"prep": ["Rinse", "Remove lid"], "warn": [], "recyclable": True},
    24: {"prep": ["Wrap in newspaper for safety", "Place in general waste"], "warn": ["⚠️ Handle with care — risk of injury"], "recyclable": False},
    25: {"prep": ["Rinse"], "warn": [], "recyclable": True},
    26: {"prep": ["Wrap in newspaper", "Place in general waste"], "warn": ["⚠️ Handle with care — different glass type, not recyclable with bottles"], "recyclable": False},
    # ── METAL ─────────────────────────────────────────────────────────────────
    27: {"prep": ["Rinse", "Crush if possible"], "warn": [], "recyclable": True},
    28: {"prep": ["Rinse thoroughly", "Remove lid"], "warn": [], "recyclable": True},
    29: {"prep": ["Place loose in recycling"], "warn": [], "recyclable": True},
    30: {"prep": ["Rinse", "Flatten if possible"], "warn": [], "recyclable": True},
    31: {"prep": ["Ball up loosely"], "warn": [], "recyclable": True},
    32: {"prep": ["Take to scrap metal facility"], "warn": ["⚠️ Sharp edges — handle carefully"], "recyclable": False},
    33: {"prep": ["Coil safely", "Take to scrap facility"], "warn": ["⚠️ Sharp ends — handle carefully"], "recyclable": False},
    34: {"prep": ["Do NOT puncture or incinerate", "Take to hazardous waste collection"], "warn": ["⚠️ Pressurised — do not crush or heat"], "recyclable": False},
    35: {"prep": ["Rinse", "Recycle with metal cans"], "warn": [], "recyclable": True},
    # ── PAPER ─────────────────────────────────────────────────────────────────
    36: {"prep": ["Flatten", "Remove tape and staples if possible"], "warn": [], "recyclable": True},
    37: {"prep": ["Flatten"], "warn": [], "recyclable": True},
    38: {"prep": ["Unfold and flatten"], "warn": [], "recyclable": True},
    39: {"prep": ["Place in general waste"], "warn": ["Plastic/wax lining means this cannot be recycled"], "recyclable": False},
    40: {"prep": ["Fold or roll to save space"], "warn": [], "recyclable": True},
    41: {"prep": ["Place in recycling — no need to remove staples"], "warn": [], "recyclable": True},
    42: {"prep": ["Flatten", "Keep dry"], "warn": [], "recyclable": True},
    43: {"prep": ["Place in general waste"], "warn": ["Contaminated paper — not recyclable"], "recyclable": False},
    44: {"prep": ["Place in recycling"], "warn": [], "recyclable": True},
    45: {"prep": ["Recycle with paper"], "warn": ["Some receipts (thermal) are not recyclable — check if shiny"], "recyclable": True},
    46: {"prep": ["Rinse", "Flatten", "Remove plastic straw/cap"], "warn": [], "recyclable": True},
    # ── ORGANIC ───────────────────────────────────────────────────────────────
    47: {"prep": ["Place in compost or green bin", "Drain excess liquid"], "warn": [], "compostable": True},
    48: {"prep": ["Place in compost bin"], "warn": [], "compostable": True},
    49: {"prep": ["Place in compost bin"], "warn": [], "compostable": True},
    50: {"prep": ["Place in compost bin", "Crush to speed composting"], "warn": [], "compostable": True},
    51: {"prep": ["Place in compost bin"], "warn": [], "compostable": True},
    52: {"prep": ["Remove food residue if possible", "Place in general waste"], "warn": ["Too contaminated to recycle"], "recyclable": False},
    # ── CIGARETTE ─────────────────────────────────────────────────────────────
    53: {"prep": ["Ensure fully extinguished", "Place in general waste"], "warn": ["⚠️ Contains toxic chemicals — never compost or recycle"], "recyclable": False},
    54: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
    55: {"prep": ["Ensure empty", "Place in general waste"], "warn": ["If still contains fuel, take to hazardous collection"], "recyclable": False},
    # ── TEXTILE ───────────────────────────────────────────────────────────────
    56: {"prep": ["Clean items → donate or clothing bank", "Worn/stained → general waste or textile recycling"], "warn": [], "recyclable": False},
    57: {"prep": ["Wearable → donate", "Unusable → general waste or shoe recycling point"], "warn": [], "recyclable": False},
    58: {"prep": ["Place in general waste or textile recycling bank"], "warn": [], "recyclable": False},
    59: {"prep": ["Seal in a bag before disposal", "Place in sealed sanitary bin"], "warn": ["⚠️ Biohazard risk — do not recycle"], "recyclable": False},
    60: {"prep": ["Seal in a bag", "Place in sealed sanitary bin"], "warn": ["⚠️ Biohazard risk — do not recycle"], "recyclable": False},
    # ── E-WASTE ───────────────────────────────────────────────────────────────
    61: {"prep": ["Take to battery collection point (supermarkets, electronics stores)"], "warn": ["⚠️ Never place batteries in general or recycling bins — fire risk"], "recyclable": False},
    62: {"prep": ["Coil neatly", "Take to e-waste collection"], "warn": [], "recyclable": False},
    63: {"prep": ["Take to e-waste collection point", "Data wipe if applicable"], "warn": ["⚠️ Contains hazardous materials"], "recyclable": False},
    64: {"prep": ["Factory reset / wipe data", "Take to e-waste or donate if working"], "warn": ["⚠️ Contains battery — never place in general waste"], "recyclable": False},
    65: {"prep": ["Take to e-waste collection"], "warn": [], "recyclable": False},
    # ── SANITARY ──────────────────────────────────────────────────────────────
    66: {"prep": ["Place in sharps container", "Never place loose in any bin"], "warn": ["⚠️ Sharps — serious injury risk", "Take to pharmacy or medical facility sharps disposal"], "recyclable": False},
    67: {"prep": ["Seal in bag", "Place in sanitary bin"], "warn": ["⚠️ May contain drug residues"], "recyclable": False},
    68: {"prep": ["Seal in bag", "Place in sanitary bin"], "warn": ["⚠️ Biohazard — do not recycle"], "recyclable": False},
    69: {"prep": ["Wrap and seal before disposal", "Place in sanitary bin"], "warn": ["⚠️ Hygienic disposal only — do not place in recycling"], "recyclable": False},
    70: {"prep": ["Fold and seal", "Place in sanitary bin or sealed nappy sack"], "warn": ["⚠️ Hygienic disposal only"], "recyclable": False},
    # ── BEVERAGE-SPECIFIC ─────────────────────────────────────────────────────
    71: {"prep": ["Check if aluminium (recyclable) or plastic (check local rules)", "Rinse"], "warn": ["Check local recycling rules for coffee pods"], "recyclable": True},
    72: {"prep": ["Remove from wrapper", "Place in compost"], "warn": ["Paper tea bags are compostable; plastic mesh bags are not"], "compostable": True},
    # ── CONSTRUCTION ──────────────────────────────────────────────────────────
    73: {"prep": ["Take to construction waste facility or skip"], "warn": ["Not accepted in household bins"], "recyclable": False},
    74: {"prep": ["Take to construction waste facility"], "warn": ["Not accepted in household bins"], "recyclable": False},
    75: {"prep": ["Take to hazardous waste facility", "Do NOT pour down drain"], "warn": ["⚠️ Flammable — keep away from heat", "⚠️ Never pour into general waste"], "recyclable": False},
    76: {"prep": ["Clean untreated wood → woodchip/recycling", "Treated wood → general waste"], "warn": ["Treated/painted wood cannot be composted or recycled normally"], "recyclable": True},
    # ── AUTOMOTIVE ────────────────────────────────────────────────────────────
    77: {"prep": ["Take to tyre retailer or recycling centre"], "warn": ["⚠️ Specialist recycling required — not household bin"], "recyclable": False},
    78: {"prep": ["Take to scrap metal dealer or auto recycler"], "warn": ["May contain hazardous fluids"], "recyclable": False},
    # ── BIODEGRADABLE ─────────────────────────────────────────────────────────
    79: {"prep": ["Place in compost or green waste bin"], "warn": [], "compostable": True},
    80: {"prep": ["Break into smaller pieces", "Compost or green waste"], "warn": [], "compostable": True},
    81: {"prep": ["Empty loose soil into compost", "Dispose of bag by material type"], "warn": [], "compostable": True},
    # ── HAZARDOUS ─────────────────────────────────────────────────────────────
    82: {"prep": ["Keep in original container", "Take to hazardous waste collection"], "warn": ["⚠️ Do not mix with other chemicals", "⚠️ Do not pour down drain or into general waste"], "recyclable": False},
    83: {"prep": ["Keep sealed", "Take to hazardous waste / oil recycling point"], "warn": ["⚠️ Oil contaminates groundwater — never pour down drain"], "recyclable": False},
    84: {"prep": ["Ensure valve is closed", "Take to hazardous waste facility"], "warn": ["⚠️ Pressurised and flammable — do not puncture or incinerate"], "recyclable": False},
    # ── SMALL LITTER ──────────────────────────────────────────────────────────
    85: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
    86: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
    87: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
    88: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
    89: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
    90: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
    91: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
    92: {"prep": ["Wrap in newspaper", "Place in general waste"], "warn": ["⚠️ Handle carefully — sharp"], "recyclable": False},
    # ── RECREATION ────────────────────────────────────────────────────────────
    93: {"prep": ["Deflate", "Place in general waste"], "warn": ["Balloons are a wildlife hazard — never release outdoors"], "recyclable": False},
    94: {"prep": ["Coil safely", "Place in general waste"], "warn": ["⚠️ Fishing line is a serious wildlife hazard — never leave in environment"], "recyclable": False},
    95: {"prep": ["Place in general waste"], "warn": ["⚠️ Marine wildlife hazard — dispose of responsibly"], "recyclable": False},
    # ── FAST FOOD ─────────────────────────────────────────────────────────────
    96: {"prep": ["Remove food debris", "Place in general waste"], "warn": ["Food contamination prevents recycling"], "recyclable": False},
    97: {"prep": ["Remove food debris — if clean, recycle cardboard", "If greasy, general waste"], "warn": ["Greasy pizza boxes cannot be recycled"], "recyclable": False},
    98: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
    99: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
    # ── DURABLE ───────────────────────────────────────────────────────────────
   100: {"prep": ["If contains batteries/circuit board → e-waste", "Otherwise → general waste or council bulky item collection"], "warn": [], "recyclable": False},
   101: {"prep": ["Book council bulky item collection or skip hire"], "warn": ["Not accepted in household bins"], "recyclable": False},
    # ── INFRASTRUCTURE ────────────────────────────────────────────────────────
   102: {"prep": ["Return to council / highways authority if on public land", "General waste if private"], "warn": [], "recyclable": False},
   103: {"prep": ["Return to supplier or take to wood recycling"], "warn": [], "recyclable": True},
    # ── MULTI-MATERIAL ────────────────────────────────────────────────────────
   104: {"prep": ["Place in general waste"], "warn": ["Mixed materials cannot be separated for recycling"], "recyclable": False},
   105: {"prep": ["Place in general waste"], "warn": ["Mixed materials — not recyclable"], "recyclable": False},
    # ── MISC ──────────────────────────────────────────────────────────────────
   106: {"prep": ["Set aside for manual inspection"], "warn": ["Cannot be classified — inspect before disposal"], "recyclable": False},
   107: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
   108: {"prep": ["Place in general waste"], "warn": [], "recyclable": False},
}


# =============================================================================
# GuidanceEngine
# =============================================================================

class GuidanceEngine:
    """
    Produces disposal guidance for each detected waste object.

    Usage
    -----
    >>> engine = GuidanceEngine()
    >>> guidance = engine.get_guidance(class_id=0, class_name="plastic_bottle")
    >>> print(guidance.stream)         # "recoverable"
    >>> print(guidance.stream_label)   # "Recoverable Waste"
    >>> print(guidance.preparation)    # ["Empty contents", "Rinse", ...]
    """

    def __init__(self) -> None:
        # Verify at construction time that all 109 IDs are covered
        missing = set(range(109)) - set(_CLASS_TO_STREAM.keys())
        if missing:
            raise RuntimeError(f"GuidanceEngine: unmapped IDs {missing}")
        log.debug("GuidanceEngine initialised — 109 classes, 7 streams")

    def get_guidance(
        self,
        class_id: int,
        class_name: str = "",
    ) -> ClassGuidance:
        """Return disposal guidance for a single detection."""
        stream = _CLASS_TO_STREAM.get(class_id, "reject")
        stream_info = STREAMS.get(stream, STREAMS["reject"])
        detail = _CLASS_DETAIL.get(class_id, {
            "prep": ["Place in appropriate bin"], "warn": [], "recyclable": False
        })

        return ClassGuidance(
            class_id=class_id,
            class_name=class_name or f"class_{class_id}",
            stream=stream,
            stream_label=stream_info["label"],
            bin_color=stream_info["bin_color"],
            collection_point=stream_info["bin"],
            summary=self._make_summary(class_name or f"class_{class_id}", stream_info),
            preparation=detail.get("prep", []),
            warnings=detail.get("warn", []),
            recyclable=detail.get("recyclable", False),
            compostable=detail.get("compostable", False),
        )

    def get_stream(self, class_id: int) -> str:
        """Return just the stream name for a class ID."""
        return _CLASS_TO_STREAM.get(class_id, "reject")

    def get_stream_info(self, stream: str) -> Dict[str, str]:
        """Return stream metadata (label, description, bin, color, icon)."""
        return STREAMS.get(stream, STREAMS["reject"])

    def summarise_detections(
        self, detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate guidance across all detections in one image.
        Returns stream breakdown counts and primary action.
        """
        stream_counts: Dict[str, int] = {}
        for det in detections:
            stream = det.get("waste_stream", self.get_stream(det.get("class_id", 106)))
            stream_counts[stream] = stream_counts.get(stream, 0) + 1

        primary = max(stream_counts, key=stream_counts.get) if stream_counts else "reject"
        streams_present = [
            {
                "stream": s,
                "label": STREAMS.get(s, {}).get("label", s),
                "count": c,
                "color": STREAMS.get(s, {}).get("color", "#9E9E9E"),
                "bin":   STREAMS.get(s, {}).get("bin", ""),
            }
            for s, c in sorted(stream_counts.items(), key=lambda x: -x[1])
        ]
        return {
            "stream_breakdown": streams_present,
            "primary_stream": primary,
            "primary_stream_label": STREAMS.get(primary, {}).get("label", primary),
            "total_items": len(detections),
            "has_hazardous": stream_counts.get("hazardous", 0) > 0,
            "has_sanitary":  stream_counts.get("sanitary", 0) > 0,
            "has_e_waste":   stream_counts.get("e_waste", 0) > 0,
        }

    @staticmethod
    def _make_summary(class_name: str, stream_info: Dict[str, str]) -> str:
        name = class_name.replace("_", " ").capitalize()
        return f"{name} → {stream_info['label']} — {stream_info['bin']}"

    @staticmethod
    def all_streams() -> Dict[str, Dict[str, str]]:
        """Return the complete stream definitions dict."""
        return STREAMS

    @staticmethod
    def stream_for_class(class_id: int) -> str:
        return _CLASS_TO_STREAM.get(class_id, "reject")
