# darkskysites.com Validation Data

## Current Data

**File:** `portugal_grid_sqm_2012_2025.json`
- Grid spacing: 0.1° (~11km)
- Points: 1,920
- Coverage: Mainland Portugal (36.9-42.2°N, -9.5 to -6.1°E)
- Data: Annual SQM values 2012-2025 per point
- Downloaded: 2026-05-06

## API Details

**Endpoint:** `https://www.darkskysites.com/api/sqm-history?lat={lat}&lng={lon}`

- Returns monthly + annual SQM history from 2012 to present
- No authentication required
- Spatial resolution: changes at ~0.015-0.02° spacing (~1.5-2 km)
- Rate limit: ~1 request/second (server-side, parallelism doesn't help)
- Error rate increases above 20 concurrent connections

## Validation Results (from 30-point comparison)

- MAE: 0.46 mag
- Bias: +0.46 (our model predicts slightly darker)
- Correlation: 0.905
- 83% of points within ±0.5 mag

## For Completeness: High-Resolution Download

The current 0.1° grid captures broad patterns. For pixel-level validation, a 0.02° grid would be optimal:

```python
# Optimal download script (run overnight — ~8 hours at ~1 req/s)
import urllib.request
import json
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_sqm(lat, lon, retries=3):
    """Fetch SQM history with retry logic."""
    url = f'https://www.darkskysites.com/api/sqm-history?lat={lat:.3f}&lng={lon:.3f}'
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read().decode())
            annual = [{'year': r['year'], 'sqm': r['sqm']} 
                      for r in data['data'] if r['type'] == 'annual']
            return {'lat': round(lat, 3), 'lon': round(lon, 3), 'annual': annual}
        except Exception:
            time.sleep(2 ** attempt)
    return None

# Grid parameters
lat_range = np.arange(36.9, 42.2, 0.02)   # 266 steps
lon_range = np.arange(-9.5, -6.1, 0.02)   # 171 steps
# Total: ~45,500 points

output_file = Path('portugal_grid_sqm_0.02deg_2012_2025.json')

# Use 10 workers (more causes errors due to server rate limiting)
# Estimated time: ~8-10 hours
all_data = []
count = 0
total = len(lat_range) * len(lon_range)

# Process in batches of 10 to respect rate limits
batch_size = 10
points = [(lat, lon) for lat in lat_range for lon in lon_range]

for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(fetch_sqm, lat, lon): (lat, lon) 
                   for lat, lon in batch}
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_data.append(result)
    
    count += len(batch)
    if count % 500 == 0:
        print(f'{count}/{total} ({100*count/total:.0f}%)')
        # Save checkpoint
        with open(output_file, 'w') as f:
            json.dump({'metadata': {'points': len(all_data), 'spacing': 0.02}, 
                       'data': all_data}, f)
    
    time.sleep(0.5)  # Brief pause between batches

# Final save
with open(output_file, 'w') as f:
    json.dump({
        'metadata': {
            'source': 'darkskysites.com API',
            'grid_spacing_deg': 0.02,
            'total_points': len(all_data),
            'bbox': {'lat_min': 36.9, 'lat_max': 42.2, 'lon_min': -9.5, 'lon_max': -6.1},
        },
        'data': all_data
    }, f)
```

**Notes:**
- Server rate-limits to ~1 req/s regardless of concurrency
- More than 20 concurrent connections causes 40%+ error rate
- Best approach: 10 workers + 0.5s batch pause = ~1.5 req/s effective
- Save checkpoints every 500 points (resume on failure)
- Run overnight or in background: `nohup python download_grid.py &`
