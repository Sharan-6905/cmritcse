"""Disposal knowledge base and approximate environmental-impact factors.

General guidance only: local recycling rules differ by municipality.
CO2 factors are rough, illustrative kg CO2e avoided per kg recycled instead of
landfilled/virgin production; they are not audited figures.
"""

GUIDES = {
    "cardboard": {
        "bin": "Dry recycling (paper/cardboard)",
        "steps": [
            "Remove tape, labels and staples where easy.",
            "Flatten boxes to save space in the bin.",
            "Keep it dry; wet or grease-soaked cardboard (e.g. pizza-box bottoms) cannot be recycled - tear off the clean part, compost or bin the rest.",
        ],
        "decomposition": "Cardboard breaks down in about 2-3 months in compost; in landfill it releases methane, so recycle or compost clean scraps.",
        "avoid": "Do not burn it or send clean cardboard to landfill.",
    },
    "glass": {
        "bin": "Glass recycling (bottle bank or glass bin)",
        "steps": [
            "Empty and rinse the container; no need to scrub.",
            "Remove metal or plastic lids and recycle them separately.",
            "Sort by colour if your local scheme asks for it.",
            "Do not mix with ceramics, drinking glass, mirrors or window glass - they melt at different temperatures.",
        ],
        "decomposition": "Glass does not biodegrade (about 1 million years), but it can be recycled endlessly without losing quality.",
        "avoid": "Never put broken glass loose in a bag; wrap it safely if it must go in general waste.",
    },
    "metal": {
        "bin": "Dry recycling (metals/cans)",
        "steps": [
            "Empty and rinse food residue.",
            "Squash cans lightly if your scheme allows; keep labels on.",
            "Aluminium foil: scrunch clean pieces into a ball bigger than a fist.",
            "Aerosols must be completely empty; never puncture or crush them.",
        ],
        "decomposition": "Steel cans can take 50+ years and aluminium 200+ years to corrode, but both recycle indefinitely and aluminium recycling saves about 95% of the energy of making new.",
        "avoid": "Do not put batteries, paint tins with wet paint or pressurised cans in the normal recycling.",
    },
    "paper": {
        "bin": "Dry recycling (paper)",
        "steps": [
            "Keep it clean and dry.",
            "Staples and paper clips are fine to leave in; remove plastic windows or covers if easy.",
            "Shred only if needed for privacy, and bag the shreds so they do not jam sorting machines.",
            "Used tissues, paper towels, wax or laminated paper cannot be recycled - compost tissues or bin them.",
        ],
        "decomposition": "Paper breaks down in about 2-6 weeks in compost, but recycling saves more energy and fibre than composting.",
        "avoid": "Do not recycle greasy, waxed or food-soiled paper.",
    },
    "plastic": {
        "bin": "Dry recycling (plastics) - check the resin number on the item",
        "steps": [
            "Empty and rinse; replace the cap if your scheme accepts it.",
            "Check the number in the triangle: #1 (PET) and #2 (HDPE) are widely accepted; #3-#7 vary by area.",
            "Plastic bags and films usually go to supermarket drop-off points, not kerbside bins.",
            "Do not bag recyclables inside a plastic bag; keep them loose.",
        ],
        "decomposition": "Plastic does not truly decompose; it fragments into microplastics over 100-500+ years, so recycling or reducing use matters most.",
        "avoid": "Never burn plastic or leave it in nature; it releases toxic fumes and microplastics.",
    },
    "trash": {
        "bin": "General (residual) waste, or a specialist drop-off if hazardous",
        "steps": [
            "Check whether it is actually special waste: batteries, electronics, light bulbs, paint and chemicals need a household-waste centre.",
            "Food waste should go to a compost or food-waste bin where available.",
            "Everything else non-recyclable goes in general waste, bagged to prevent litter.",
        ],
        "decomposition": "Mixed general waste in landfill breaks down slowly and produces methane; reducing and separating waste is the best remedy.",
        "avoid": "Do not put hazardous items (batteries, chemicals) in general waste.",
    },
    "organic": {
        "bin": "Food-waste / compost bin (or garden compost)",
        "steps": [
            "Put fruit and vegetable scraps, peels, coffee grounds and eggshells in the food-waste or compost bin.",
            "Use a compostable liner or no liner; never a regular plastic bag.",
            "Cooked food, meat and dairy go in the council food-waste bin, not a home compost heap.",
        ],
        "decomposition": "Food scraps break down in weeks to a few months when composted; in landfill they rot without oxygen and release methane.",
        "avoid": "Do not put food waste in general waste or recycling - it contaminates clean recyclables.",
    },
    "battery": {
        "bin": "Battery drop-off point (shops, supermarkets, household-waste centre)",
        "steps": [
            "Never put batteries in any household bin: damaged lithium cells can start fires in bins and collection trucks.",
            "Tape over the terminals of lithium and 9V batteries with clear tape.",
            "Take them to a battery collection box or household-waste recycling centre.",
        ],
        "decomposition": "Batteries do not decompose safely; metals and chemicals leak into soil and water over decades.",
        "avoid": "Never crush, burn or puncture a battery, and never bin swollen or damaged ones - ask your waste centre for advice.",
    },
    "e-waste": {
        "bin": "Electronics recycling point (retailer take-back or household-waste centre)",
        "steps": [
            "Back up and wipe any personal data from devices first.",
            "Remove batteries if easy to do so and recycle them separately.",
            "Return it to a retailer take-back scheme or a household-waste recycling centre.",
            "If it still works, donating or repairing it is better than recycling.",
        ],
        "decomposition": "Electronics do not biodegrade; they contain recoverable metals and also hazardous substances, so they need specialist recycling.",
        "avoid": "Do not put electronics in general waste or kerbside recycling.",
    },
    "textile": {
        "bin": "Clothing donation or textile bank",
        "steps": [
            "If wearable and clean, donate or resell it.",
            "Worn-out but clean textiles can go in a textile bank - they are recycled into rags or insulation.",
            "Only heavily soiled or wet textiles belong in general waste.",
        ],
        "decomposition": "Natural fibres such as cotton take months to years to break down; synthetic fabrics shed microplastics and last for centuries.",
        "avoid": "Do not put textiles in paper or plastic recycling.",
    },
    "hazardous": {
        "bin": "Household hazardous waste (HHW) facility or collection event - never a household bin or drain",
        "steps": [
            "Keep it in its original, tightly closed, labelled container. Never mix products, and never pour it down a drain, toilet or onto the ground.",
            "Paint, thinners, solvents, pesticides, bleach and other chemicals: take them to a household hazardous-waste site. Fully dried-out latex paint cans can often go in metal recycling.",
            "Aerosols: only completely empty cans go in metal recycling; part-full ones go to the hazardous-waste site.",
            "Fluorescent tubes and CFL bulbs contain mercury: do not break them; wrap them and take them to a retailer take-back point or the hazardous-waste site. If one breaks, open a window, leave the room for 10 minutes, do not vacuum, and scoop the pieces into a sealed jar.",
            "Medicines: use a pharmacy take-back scheme; do not flush them unless the label says so.",
            "Needles and sharps: put them in a rigid, puncture-proof sharps container and use a community sharps drop-off.",
            "Motor oil and car fluids: carry them sealed to a garage, recycling centre or the hazardous-waste site.",
            "Store upright, away from heat, children and pets until drop-off, and check with your local council for opening times and accepted items.",
        ],
        "decomposition": "Hazardous products do not break down safely: solvents, metals and chemicals leak into soil and groundwater, and mercury, oil and pesticides stay toxic for years.",
        "avoid": "Never put these in general waste, recycling, compost or a drain, never burn them, and never mix chemicals (for example bleach with ammonia or acids).",
    },
}

# kg CO2e avoided per kg recycled (rough, illustrative)
CO2_PER_KG = {"cardboard": 0.9, "glass": 0.3, "metal": 4.0, "paper": 1.0, "plastic": 1.5, "trash": 0.0, "organic": 0.2, "battery": 0.0, "e-waste": 0.0, "textile": 1.0, "hazardous": 0.0}

# short one-liners for the UI cards: (breaks down in, avoid)
SHORT = {
    "cardboard": ("About 2-3 months in compost", "Greasy or wet cardboard in recycling"),
    "glass": ("About 1 million years", "Mixing with ceramics or mirrors"),
    "metal": ("50-200+ years", "Aerosols and batteries in the bin"),
    "paper": ("2-6 weeks in compost", "Greasy, waxed or tissue paper"),
    "plastic": ("100-500+ years", "Bagging recyclables"),
    "trash": ("Slow; makes methane in landfill", "Hazardous items in general waste"),
    "organic": ("Weeks to months composted", "Food in general waste"),
    "battery": ("Leaks for decades", "Any household bin: fire risk"),
    "e-waste": ("Never breaks down safely", "General waste or kerbside bins"),
    "textile": ("Months to centuries", "Paper or plastic recycling"),
    "hazardous": ("Never breaks down safely", "Drains, soil and household bins"),
}
