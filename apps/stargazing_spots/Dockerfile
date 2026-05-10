FROM python:3.12-slim AS base

# System deps for GDAL/rasterio/geopandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency file first (Docker cache layer)
COPY pyproject.toml .

# Install Python dependencies
RUN uv pip install --system --no-cache .

# Copy source
COPY stargazing_spots/ stargazing_spots/

# Create data directories
RUN mkdir -p input/portugal/{cache,boundary,osm} \
             output/portugal/{mainland,madeira,azores,visualizations}

ENTRYPOINT ["python", "-m", "stargazing_spots"]
