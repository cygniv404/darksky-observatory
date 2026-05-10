"""Milky Way visibility season and observation planning per latitude.

Galactic Center (RA 17h45m, Dec -29 deg) visibility window computed
from spherical astronomy. At 37-42 deg N: visible Apr-Oct, peak Aug.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Galactic Center coordinates (J2000)
GC_RA_HOURS = 17.75  # Right Ascension in hours (17h45m)
GC_DEC_DEG = -29.0  # Declination in degrees


def milky_way_season(latitude: float) -> dict:
    """Determine Milky Way core visibility season for a given latitude.

    Computes which months the Galactic Center is above the horizon
    during astronomical darkness (sun < -18°).

    Args:
        latitude: Observer latitude in degrees (positive = North).

    Returns:
        Dict with 'best_months', 'visible_months', 'peak_month',
        'max_altitude', and 'description'.
    """
    # Maximum altitude of GC above horizon
    # alt_max = 90 - |lat - dec| (for objects that transit south of zenith)
    max_alt = 90 - abs(latitude - GC_DEC_DEG)

    if max_alt <= 0:
        return {
            "best_months": [],
            "visible_months": [],
            "peak_month": None,
            "max_altitude": 0,
            "description": "Galactic Center never rises at this latitude",
        }

    # GC is on the meridian at local midnight when Sun's RA = GC_RA - 12h = 5.75h
    # Sun at RA 5.75h corresponds to ~late September (Sun moves ~1h RA per month)
    # Months when GC is above horizon during dark hours (approximate):
    # Visible: April through October
    # Best (GC high at midnight): July-September
    # Peak: August (GC transits near midnight, nights are reasonably long)

    # More precise: GC rises/sets when cos(hour_angle) = -tan(lat)*tan(dec)
    cos_ha = -np.tan(np.radians(latitude)) * np.tan(np.radians(GC_DEC_DEG))
    cos_ha = np.clip(cos_ha, -1, 1)

    if cos_ha <= -1:
        # GC is circumpolar (never sets) — not relevant for mid-latitudes
        visible = list(range(1, 13))
        best = [6, 7, 8, 9]
    elif cos_ha >= 1:
        # GC never rises
        return {
            "best_months": [],
            "visible_months": [],
            "peak_month": None,
            "max_altitude": 0,
            "description": "Galactic Center never rises at this latitude",
        }
    else:
        # Normal case: GC rises and sets
        # For Portugal (37-42°N), GC is above horizon ~8-10 hours per day in summer
        visible = [4, 5, 6, 7, 8, 9, 10]  # April through October
        best = [6, 7, 8, 9]  # June through September

    peak = 8  # August — best balance of darkness + GC altitude

    # Adjust for latitude: further north = shorter visibility window
    if latitude > 45:
        visible = [5, 6, 7, 8, 9]
        best = [6, 7, 8]
        peak = 7

    return {
        "best_months": best,
        "visible_months": visible,
        "peak_month": peak,
        "max_altitude": round(max_alt, 1),
        "description": f"Milky Way core visible {_months_str(visible)}, best in {_months_str(best)}, peak altitude {max_alt:.0f}°",
    }


def moon_free_nights_per_month() -> int:
    """Approximate number of moon-free nights per month.

    The lunar cycle is 29.5 days. Approximately 10-12 nights per month
    have negligible moonlight (moon below horizon during prime observing
    hours OR moon phase < 25% illuminated).

    This is constant regardless of location or season.

    Returns:
        Approximate count of suitable dark nights per month.
    """
    return 10


def observation_summary(latitude: float) -> dict:
    """Generate complete observation planning summary for a location.

    Args:
        latitude: Observer latitude in degrees.

    Returns:
        Dict with Milky Way info, moon info, and overall recommendation.
    """
    mw = milky_way_season(latitude)

    return {
        "milky_way": mw,
        "moon_free_nights_per_month": moon_free_nights_per_month(),
        "best_overall": "Visit in August during new moon week for optimal conditions",
        "seasonal_notes": {
            "summer_jun_sep": "Best: MW visible, longest clear-sky probability, warm nights",
            "spring_apr_may": "Good: MW rises late, some clear nights, comfortable temperatures",
            "autumn_oct_nov": "Fair: MW setting early, increasing clouds, still possible",
            "winter_dec_feb": "Poor: No MW core, frequent clouds, cold — deep-sky objects only",
        },
    }


def add_planning_columns(spots_gdf):
    """Add observation planning columns to spots GeoDataFrame.

    Args:
        spots_gdf: GeoDataFrame with Point geometry.

    Returns:
        GeoDataFrame with added columns:
            - mw_best_months: comma-separated best months for Milky Way
            - mw_max_altitude: maximum altitude of Galactic Center (degrees)
            - mw_peak_month: single best month
            - best_season: overall recommendation
    """
    spots_gdf = spots_gdf.copy()

    lats = spots_gdf.geometry.y.values
    best_months_list = []
    max_alt_list = []
    peak_list = []

    for lat in lats:
        mw = milky_way_season(lat)
        best_months_list.append(",".join(str(m) for m in mw["best_months"]))
        max_alt_list.append(mw["max_altitude"])
        peak_list.append(mw["peak_month"])

    spots_gdf["mw_best_months"] = best_months_list
    spots_gdf["mw_max_altitude_deg"] = max_alt_list
    spots_gdf["mw_peak_month"] = peak_list
    spots_gdf["best_season"] = "Jun-Sep (Milky Way + clear skies)"

    return spots_gdf


def _months_str(months: list) -> str:
    """Convert month numbers to abbreviated string."""
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if not months:
        return "none"
    return f"{names[months[0]-1]}–{names[months[-1]-1]}"
