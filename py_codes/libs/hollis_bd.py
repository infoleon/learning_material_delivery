# -*- coding: utf-8 -*-

"""
Hollis et al. (2012)
Empirically-derived pedotransfer functions for predicting bulk density in European soils
"""


import numpy as np

def bd_volcanic_materials(OC, horizon_midpoint_cm):
    """
    Volcanic materials:
    Db (g/cm³) = 1.5868
               - 0.4682 * exp(0.0578 * OC)
               - 0.07778 * ln(horizon_midpoint_cm)
    """
    return 1.5868 \
           - 0.4682 * np.exp(0.0578 * OC) \
           - 0.07778 * np.log(horizon_midpoint_cm)

def bd_cultivated_topsoils(OC, total_sand, clay):
    """
    Cultivated topsoils:
    Db (g/cm³) = 0.80806
               + 0.823844 * exp(-0.27993 * OC)
               + 0.0014065 * total_sand
               - 0.0010299 * clay
    """
    return 0.80806 + (0.823844 * np.exp(-0.27993 * OC) \
           + 0.0014065 * total_sand - 0.0010299 * clay)

def bd_compact_subsoils(OC, horizon_midpoint_cm, total_sand):
    """
    Compact subsoils:
    Db (g/cm³) = 1.1257
               - 0.1140245 * ln(OC)
               + 0.0555 * ln(horizon_midpoint_cm)
               + 0.002248 * total_sand
    """
    return 1.1257 \
           - 0.1140245 * np.log(OC) \
           + 0.0555 * np.log(horizon_midpoint_cm) \
           + 0.002248 * total_sand

def bd_all_other_mineral_horizons(OC, total_sand, clay):
    """
    All other mineral horizons:
    Db (g/cm³) = 0.69794
               + 0.750636 * exp(-0.230355 * OC)
               + 0.0008687 * total_sand
               - 0.0005164 * clay
    """
    return 0.69794 \
           + 0.750636 * np.exp(-0.230355 * OC) \
           + 0.0008687 * total_sand \
           - 0.0005164 * clay

def bd_all_organic_horizons(OC):
    """
    All organic horizons:
    Db (g/cm³) = 1.4903
               - 0.33293 * ln(OC)
    """
    return 1.4903 - 0.33293 * np.log(OC)


if __name__ == "__main__":
    OC = 0.1       # percent
    sand = 75.0    # percent
    clay = 7.0     # percent
    depth_mid = 15.0  # cm

    #print("Volcanic materials BD:", bd_volcanic_materials(OC, depth_mid))
    print("Cultivated topsoils BD:", bd_cultivated_topsoils(OC, sand, clay))
    print("Compact subsoils BD:", bd_compact_subsoils(OC, depth_mid, sand))
    print("All other mineral horizons BD:", bd_all_other_mineral_horizons(OC, sand, clay))
    #print("All organic horizons BD:", bd_all_organic_horizons(OC))















































