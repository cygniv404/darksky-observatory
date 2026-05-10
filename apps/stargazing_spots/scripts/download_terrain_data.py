"""Download all terrain/land cover tiles for Portugal (one-time setup).

Downloads ~3.4 GB of raster data covering all of Portugal:
- Copernicus GLO-30 DEM (7 tiles, ~1.4 GB)
- ESA WorldCover 10m (3 tiles, ~1.2 GB)
- Hansen Global Forest Change (2 tiles, ~800 MB)

After download, the pipeline reads locally — zero network calls during runs.
Run once: `uv run python scripts/download_terrain_data.py`
"""

import os
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Output directories
BASE = Path(__file__).parent.parent
DEM_DIR = BASE / "input" / "portugal" / "dem"
WORLDCOVER_DIR = BASE / "input" / "portugal" / "worldcover"
HANSEN_DIR = BASE / "input" / "portugal" / "hansen"


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file with progress reporting."""
    if output_path.exists():
        print(f"  ✓ Already exists: {output_path.name}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {description or output_path.name}...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=300)
        with open(output_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                f.write(chunk)
        size_mb = output_path.stat().st_size / 1e6
        print(f"  ✓ Done: {output_path.name} ({size_mb:.0f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {output_path.name} — {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_copernicus_dem():
    """Download Copernicus GLO-30 DEM tiles covering Portugal.

    Mainland: lat 36-42, lon -10 to -6 → tiles N36-N42, W006-W010
    Madeira: lat 32-33, lon -17 to -16
    Azores: lat 36-39, lon -32 to -25
    """
    print("\n=== Copernicus GLO-30 DEM (30m) ===")

    tiles = []
    # Mainland
    for lat in range(36, 43):
        for lon in range(6, 11):
            tiles.append((lat, lon))
    # Madeira
    tiles.extend([(32, 17), (32, 18), (33, 17)])
    # Azores (main islands)
    for lat in [36, 37, 38, 39]:
        for lon in [25, 26, 27, 28, 29, 31, 32]:
            tiles.append((lat, lon))

    base = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
    downloads = []

    for lat, lon in set(tiles):
        lat_str = f"N{lat:02d}"
        lon_str = f"W{lon:03d}"
        tile_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
        url = f"{base}/{tile_name}/{tile_name}.tif"
        output = DEM_DIR / f"{tile_name}.tif"
        downloads.append((url, output, f"DEM {lat_str}_{lon_str}"))

    print(f"  Tiles needed: {len(downloads)}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_file, url, path, desc): desc
                   for url, path, desc in downloads}
        for future in as_completed(futures):
            future.result()


def download_worldcover():
    """Download ESA WorldCover 10m tiles covering Portugal."""
    print("\n=== ESA WorldCover 10m (2021) ===")

    base = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

    # Tiles are 3°×3° blocks by SW corner
    tiles = [
        ("N36", "W012"), ("N36", "W009"), ("N36", "W006"),  # Mainland south
        ("N39", "W012"), ("N39", "W009"), ("N39", "W006"),  # Mainland north
        ("N42", "W012"), ("N42", "W009"),                    # Far north
        ("N30", "W018"),                                      # Madeira
        ("N36", "W027"), ("N36", "W030"),                    # Azores
        ("N39", "W030"), ("N39", "W027"),                    # Azores north
    ]

    downloads = []
    for ns, ew in tiles:
        tile_name = f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map"
        url = f"{base}/{tile_name}.tif"
        output = WORLDCOVER_DIR / f"{tile_name}.tif"
        downloads.append((url, output, f"WorldCover {ns}_{ew}"))

    print(f"  Tiles needed: {len(downloads)}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_file, url, path, desc): desc
                   for url, path, desc in downloads}
        for future in as_completed(futures):
            future.result()


def download_hansen():
    """Download Hansen Global Forest Change treecover2000 tiles."""
    print("\n=== Hansen Global Forest Change v1.11 (2023) ===")

    base = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11"

    # Tiles are 10°×10°, named by UPPER-LEFT corner
    tiles = [
        ("40N", "010W"),  # Mainland south (covers 30-40N, -10 to 0)
        ("50N", "010W"),  # Mainland north (covers 40-50N, -10 to 0)
        ("40N", "020W"),  # Madeira (covers 30-40N, -20 to -10)
        ("40N", "030W"),  # Azores east (covers 30-40N, -30 to -20)
        ("40N", "040W"),  # Azores west (covers 30-40N, -40 to -30)
    ]

    downloads = []
    for lat_str, lon_str in tiles:
        filename = f"Hansen_GFC-2023-v1.11_treecover2000_{lat_str}_{lon_str}.tif"
        url = f"{base}/{filename}"
        output = HANSEN_DIR / filename
        downloads.append((url, output, f"Hansen {lat_str}_{lon_str}"))

    print(f"  Tiles needed: {len(downloads)}")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download_file, url, path, desc): desc
                   for url, path, desc in downloads}
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    print("=== Downloading terrain data for Portugal ===")
    print(f"Output: {BASE / 'input' / 'portugal'}")
    print(f"Expected total size: ~3.4 GB")
    print()

    download_copernicus_dem()
    download_worldcover()
    download_hansen()

    # Report totals
    total_size = 0
    for d in [DEM_DIR, WORLDCOVER_DIR, HANSEN_DIR]:
        if d.exists():
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            total_size += size
            print(f"\n  {d.name}: {size/1e9:.1f} GB ({len(list(d.rglob('*.tif')))} tiles)")

    print(f"\n=== TOTAL: {total_size/1e9:.1f} GB ===")
    print("Done! Pipeline will now read locally — no network calls during runs.")
