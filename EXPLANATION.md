# AI Pond Recommendation - Technical Explanation

## Overall Architecture

The AI Pond Recommendation API is a **deterministic, single-process geospatial analysis pipeline** that converts KML/KMZ contour maps into ranked pond site recommendations. It follows a **phased pipeline architecture** where each phase builds on the previous one:

```
Phase 1: Contour Parsing → Phase 2: DEM + Hydrology → Phase 3: Candidates → Phase 4: Geo Context → Phase 5: Rainfall/Runoff → Phase 6: Final Ranking
```

**Key design principle**: All computations are deterministic (same input → identical output). No ML models, no randomness. External API calls are cached and bounded.

---

## Module-by-Module Responsibilities

| Module | Responsibility | Key Functions |
|--------|---------------|---------------|
| `backend/parser.py` | KML/KMZ → ContourData | `parse_contour_file`, `contour_diagnostics` |
| `backend/terrain.py` | DEM generation + slope | `build_dem`, `calculate_slope`, `analyze_terrain` |
| `backend/hydrology.py` | D8 hydrology + accumulation | `analyze_hydrology`, `calculate_d8_flow_direction`, `calculate_flow_accumulation` |
| `backend/depression.py` | Priority-flood depression filling | `fill_depressions`, `preprocess_dem` |
| `backend/catchment.py` | Catchment delineation | `delineate_catchment`, `catchment_area`, `mask_to_polygon` |
| `backend/candidates.py` | Candidate generation + ranking | `generate_candidates`, `extract_candidate_seeds`, `rank_candidates` |
| `backend/suitability.py` | Component scoring functions | `evaluate_pond_suitability`, component functions |
| `backend/recommendation_engine.py` | Phase-6 unified ranking | `run_recommendation_engine`, `rank_candidates` |
| `backend/geographic_context.py` | OSM Overpass enrichment | `enrich_candidates`, `decide_feasibility` |
| `backend/rainfall_runoff.py` | Open-Meteo + rational method | `compute_rainfall_runoff` |
| `backend/services/land_service.py` | OSM Overpass client | `fetch_land_context` |
| `backend/services/rainfall_service.py` | Open-Meteo client | `fetch_historical_rainfall` |
| `backend/services/opentopography_service.py` | OpenTopography DEM | `get_dem_for_bbox` |
| `backend/services/cache.py` | File-based cache (SHA-256 + TTL) | `FileCache` |
| `backend/instrumentation.py` | Stage timing + tracing | `stage()`, `attach_payload` |
| `backend/main.py` | FastAPI endpoints | All HTTP routes |

---

## Data Flow: End-to-End

### Input: KML/KMZ Contour File
```xml
<Placemark>
  <name>280.0</name>  <!-- elevation in meters -->
  <LineString>
    <coordinates>81.28,21.26 81.29,21.26 ...</coordinates>
  </LineString>
</Placemark>
```

### Stage 1: Contour Parsing (`backend/parser.py`)

**Input**: KML/KMZ bytes  
**Output**: `ContourData` (dataclass)

```
KML bytes
    │
    ├─ _read_kml_file() / _read_kml_bytes() → XML bytes
    │
    ├─ ET.fromstring() → ElementTree
    │
    ├─ Extract placemarks (XPath: .//kml:Placemark)
    │
    ├─ For each placemark:
    │   ├─ _extract_elevation() → elevation_m (float)
    │   │   ├─ ExtendedData/SchemaData/SimpleData
    │   │   ├─ SimpleData[@name="elev"/"height"/...]
    │   │   ├─ <name> tag
    │   │   └─ <description> tag
    │   │
    │   ├─ _extract_coordinates() → [(lon, lat), ...]
    │   │
    │   ├─ pyproj.Transformer(EPSG:4326 → EPSG:32644) → (x, y) in metres
    │   │
    │   └─ Contour(elevation_m, coordinates[], name)
    │
    └─ ContourData(contours[], source_crs, target_crs, bbox, elevation stats)
```

**Key Algorithm**: `_extract_elevation()` tries multiple KML conventions:
1. ExtendedData/SchemaData/SimpleData with elevation-like names
2. SimpleData with elevation-like `@name` attributes
4. `<description>` tag content

---

## DEM Generation (`backend/terrain.py`)

### Mathematical Foundation

**Linear Interpolation** (SciPy `LinearNDInterpolator` = Qhull Delaunay triangulation):

Given contour vertices `(x_i, y_i)` with elevations `z_i`:
1. Build Delaunay triangulation of sample points
2. For each grid cell, find containing simplex
3. Barycentric interpolation within simplex
4. For cells outside convex hull → nearest-neighbor fallback (cKDTree)

```
Grid creation:
    x = [min_x, min_x + res, ..., max_x]
    y = [min_y, min_y + res, ..., max_y]
    Resolution default: 5.0m
```

**Slope Calculation** (finite differences):

```
dz_dx, dz_dy = np.gradient(elevation, resolution_m)
gradient = sqrt(dz_dx² + dz_dy²)  # m/m
slope_degrees = degrees(arctan(gradient))
```

---

## Hydrology Engine (`backend/hydrology.py`)

### D8 Flow Direction Algorithm

**D8 Encoding** (clockwise from North, row increases southward):

```
Index:  0   1   2   3   4   5   6   7
Dir:    N   NE  E   SE  S   SW  W   NW
(dr,dc):(-1,0)(-1,1)(0,1)(1,1)(1,0)(1,-1)(0,-1)(-1,-1)
Distance: 1   √2   1   √2  1   √2   1   √2
```

**Algorithm** (`calculate_d8_flow_direction`):

```
For each valid cell:
    For each of 8 directions:
        neighbour_elevation = _shifted_neighbours(elevation, dr, dc, inf)
        drop = elevation - neighbour_elevation
        slope = drop / (resolution_m × D8_DISTANCES[direction])
        if drop > 0 and slope > best_slope:
            best_direction = direction
            best_slope = slope

flow_direction = best_direction (or -1 if no lower neighbour)
```

**Tie-breaking**: Strict `>` comparison on slope → lower direction index wins (deterministic).

**Sink Mask**: Valid cells with no lower neighbour AND not on boundary ring  
**Edge Outflow**: Valid cells on boundary ring (flow exits grid)

### Flow Accumulation (Topological)

**Kahn's Algorithm** (topological sort on DAG):

```
1. downstream[i] = flat_index of downstream cell (or -1)
2. indegree[j] = count of cells draining to j
3. queue = cells with indegree == 0
4. While queue not empty:
       cell = pop()
       target = downstream[cell]
       if target >= 0:
           accumulation[target] += accumulation[cell]
           indegree[target] -= 1
           if indegree[target] == 0: push(target)
5. Unresolved (indegree > 0) → NaN (cycles)
```

**Key Property**: Each valid cell contributes exactly 1 unit. Accumulation = upstream cell count.

### Depression Filling (Priority-Flood)

`backend/depression.py` implements Wang & Liu (2006) priority-flood with epsilon:

```
1. Push all boundary cells into min-heap (key = elevation)
2. While heap not empty:
       cell = pop_min()
       For each neighbour:
           if unvisited:
               target_elev = max(neighbour_elev, cell_elev + epsilon)
               if neighbour_elev < target_elev:
                   fill_depth = target_elev - neighbour_elev
                   record fill
                   neighbour_elev = target_elev
               push(neighbour, neighbour_elev)
```

**Epsilon**: Default `1e-3` m prevents flat-surface artifacts.  
**Enclosed regions**: Not filled (left as sinks, reported in diagnostics).

---

## Catchment Delineation (`backend/catchment.py`)

### Upstream Tracing (CSR + Stack)

```
1. Build downstream flat index → downstream_flat_index()
2. Build CSR upstream adjacency:
   sources = valid cells with downstream >= 0
   targets = downstream[sources]
   sort by target → counts per target → offsets
2. Stack-based upstream traversal:
   stack = [outlet_flat]
   while stack:
       cell = pop()
       for upstream in sources[offsets[cell]:offsets[cell+1]]:
           if not catchment[upstream]:
               catchment[upstream] = True
               push(upstream)
```

**Area Calculation**: `cell_count × resolution_m²`

### Raster → Polygon (Horizontal Runs)

```
For each row:
    Find runs of True cells → (row, col_start, col_end)
    Convert each run to rectangle:
        x_left = x[col_start] - dx/2
        x_right = x[col_end] + dx/2
        y = y[row] ± dy/2
    Union rectangles → Polygon/MultiPolygon
```

**Complexity**: O(rows × runs) vs O(cells) per-cell union.

---

## Candidate Generation (`backend/candidates.py`)

### Seed Extraction

```
1. finite_accumulation = accumulation where valid else -inf
2. neighbourhood_max = maximum_filter(finite_accumulation, size=3)
3. peak_mask = valid & (finite_accumulation >= neighbourhood_max)
3. Reduce plateaus: connected components on peak_mask → highest accumulation cell
4. Add outlet cells to seeds
```

### Feature Extraction per Candidate

| Feature | Source |
|---------|--------|
| `elevation_m` | DEM elevation at cell |
| `slope_degrees` | DEM slope at cell |
| `flow_accumulation_cells` | `hydrology.flow_accumulation[row, col]` |
| `drainage_area_m2` | `accumulation × cell_area_m2` |
| `outlet_id` | Terminal cell ID from pointer doubling |
| `outlet_kind` | "boundary" / "internal" / "internal" |
| `latitude/longitude` | pyproj transform (DEM CRS → EPSG:4326) |
| `metric_crs` | UTM zone from centroid |

### Hard Constraints (`apply_hard_constraints`)

| Constraint | Check | Rejection Reason |
|------------|-------|------------------|
| NoData cell | `candidate["nodata"]` | `nodata_cell` |
| Invalid elevation | `elevation_m is None` | `invalid_elevation` |
| Invalid slope | `slope_degrees is None` | `invalid_slope` |
| Invalid accumulation | `< 1.0` | `invalid_accumulation` |
| No outlet | `outlet_id is None` | `unresolved_hydrology` |
| Invalid coords | `lat/lon` not finite | `invalid_coordinate` |
| Outside bounds | `x,y` outside DEM extent | `outside_analysis_region` |
| Min drainage area | `< min_area` | `insufficient_drainage_area` |

### Deduplication (Spatial Hash Grid)

```
bin_size = min_separation_m (default 50m)
For each candidate (ordered by strength = -accumulation):
    bin = (floor(x/bin_size), floor(y/bin_size))
    Check 3×3 neighbour bins for conflicts
    If conflict within separation_sq: reject as duplicate
    Else: keep, add to grid
```

### Phase-3 Ranking (`score_and_rank_candidates`)

| Component | Weight | Normalization |
|-----------|--------|---------------|
| `catchment_area` | 0.35 | `catchment_area_suitability(area)` |
| `flow_accumulation` | 0.25 | `accumulation / max_accumulation` |
| `slope` | 0.20 | `slope_suitability(slope)` |
| `hydrological_confidence` | 0.10 | 1.0 if boundary outlet else 0.6 |
| `elevation` | 0.10 | Normalized [min, max] inverted |

**Tie-breakers**: score DESC → flow_acc DESC → row ASC → col ASC

---

## Phase 4: Geographic Context (`backend/geographic_context.py`)

### Provider: OSM Overpass API

```
Query template:
[out:json][timeout:25];
(
  way[water][bbox:S,W,N,E];
  way[waterway][bbox:S,W,N,E];
  way[natural=water][bbox:S,W,N,E];
  way[highway][bbox:S,W,N,E];
  node[building][bbox:S,W,N,E];
  way[building][bbox:S,W,N,E];
);
out center tags geom;
```

### Clustering + Batching

```
1. Cluster candidates by metric proximity (default 1000m radius)
2. For each cluster: build bbox with 1000m buffer
3. Deduplicate identical bboxes
4. Check cache (FileCache, SHA-256 key)
4. Parallel fetch (ThreadPoolExecutor, max_workers=4)
5. Retry 2× with exponential backoff (10ms, 20ms, 50ms cap)
```

### Distance Calculation (Projected CRS)

```
1. Project candidate (x,y) to UTM zone
2. Project OSM geometries to same UTM
3. Build STRtree index per feature type
4. nearest_distance = point.distance(nearest_geometry)
5. count_within = count within 500m radius
```

### Feasibility Decision (`decide_feasibility`)

| Condition | Result | Reason |
|-----------|--------|--------|
| Provider unavailable | `unknown` | `provider_unavailable` |
| Inside water body (`water_distance <= 0`) | `rejected` | `inside_or_on_water_body` |
| Building < 25m | `rejected` | `insufficient_building_clearance` |
| Road < 25m | `rejected` | `insufficient_road_clearance` |
| Otherwise | `feasible` | None |

---

## Phase 5: Rainfall-Runoff (`backend/rainfall_runoff.py`)

### Data Source: Open-Meteo Historical Archive

```
GET https://archive-api.open-meteo.com/v1/archive
Params: latitude, longitude, start_date, end_date, daily=precipitation_sum, timezone=auto
Default: last 5 full calendar years, full year range
```

### Seasonal Aggregation

```
DEFAULT_SEASONAL_MONTHS = [6, 7, 8, 9]  # Monsoon
For each day in response:
    if month in seasonal_months:
        seasonal_total += precipitation
mean_annual_seasonal = seasonal_total / rainfall_years
```

### Rational Method

```
V = P × A × C

P = mean_annual_seasonal_rainfall_mm / 1000  (meters)
A = catchment_area_m2  (m²)
C = runoff_coefficient (default 0.3)

runoff_m3 = P × A × C
storage_m3 = runoff_m3 × 0.5  (if terrain_support)
```

**Water Availability Status**:
| Runoff (m³) | Status |
|-------------|--------|
| < 10 | `very_low` |
| < 1,000 | `low` |
| < 10,000 | `moderate` |
| < 100,000 | `good` |
| ≥ 100,000 | `high` |

---

## Phase 6: Recommendation Engine (`backend/recommendation_engine.py`)

### Pipeline

```
1. Hard Eligibility → Filter
2. Feature Extraction → CandidateFeatures
3. Component Scoring (6 components)
4. Correlation Penalty
5. Weighted Overall Score
6. Confidence (data quality)
7. Explanation Generation
8. Deterministic Ranking
```

### Component Scores

| Component | Input | Normalization |
|-----------|-------|---------------|
| `hydrology` | flow_accumulation_cells or catchment_area | Linear / ideal range |
| `terrain` | slope_degrees | Ideal range [2°, 12°], invert |
| `geography` | water/road/building distances | Distance-based (closer/farther) |
| `rainfall` | mean_annual_seasonal (mm) | Ideal [500, 2000] mm |
| `runoff` | estimated_runoff_m3 | Ideal [5000, 100000] m³ |
| `storage` | estimated_storage_m3 | Ideal [2500, 50000] m³ |

### Correlation Penalty

```
Correlated group: {catchment_area, runoff, storage}
If ≥2 components > 0.7: multiply each by 0.9
```

### Weighted Overall Score

```
weights = {hydrology: 0.25, terrain: 0.20, geography: 0.15,
           rainfall: 0.15, runoff: 0.15, storage: 0.10}
overall = Σ(weight_i × score_i) / Σ(weights)
```

### Confidence (Data Quality)

```
confidence = Σ(conf_weight_i × quality_i) / Σ(conf_weights)

Sources:
  dem_quality (30%): DEM validation score
  hydrology_quality (25%): Hydrology validation score
  geographic_availability (20%): feasible=1.0, unknown=0.5, rejected=0.0
  rainfall_availability (15%): available=1.0, missing=0.3
  runoff_storage_availability (10%): available=1.0, missing=0.3
```

### Explanation Generation

```
strengths = []
concerns = []
missing_evidence = []

if hydrology >= 0.7: strengths += "Strong hydrological connectivity"
elif hydrology >= 0.4: strengths += "Adequate hydrological connectivity"
else: concerns += "Limited upstream drainage contribution"

if slope <= ideal_max: strengths += "Favorable slope"
elif slope <= max: concerns += "Moderate slope"
else: concerns += "Steep slope exceeds limit"

if feasibility == "feasible": strengths += "Geographically feasible"
elif feasibility == "rejected": concerns += reason
else: missing_evidence += "Geographic context unavailable"

if rainfall_mm: evaluate against ideal/min
else: missing_evidence += "Rainfall data unavailable"

... (similar for runoff, storage, DEM quality, hydro quality)
```

### Deterministic Ranking

```
sort_key = (
    -overall_score,
    -confidence,
    -flow_accumulation_cells,
    latitude,      # ASC
    longitude,     # ASC
    candidate_id   # ASC
)
```

---

## External Services

### OpenTopography (`services/opentopography_service.py`)

```
GET https://portal.opentopography.org/API/globaldem
Params: demtype=SRTMGL1, south, west, north, east, outputFormat=GTiff
Requires: OPENTOPOGRAPHY_API_KEY
```

### Open-Meteo (`services/rainfall_service.py`)

```
GET https://archive-api.open-meteo.com/v1/archive
Params: latitude, longitude, start_date, end_date, daily=precipitation_sum
```

### OSM Overpass (`services/land_service.py`)

```
POST https://overpass-api.de/api/interpreter
Body: Overpass QL query (see geographic_context.py)
Timeout: 25s default, 2 retries
```

---

## Frontend (`frontend/`)

### Stack

- Leaflet 1.9.4 (map)
- Vanilla ES6 JS (no build step)
- Served statically or via `python -m http.server`

### Key Components

| Component | Function |
|-----------|----------|
| `initMap()` | Leaflet map with OSM + Esri baselayers |
| `analyzeSelectedFile()` | Upload KML → `/analyzeContour` |
| `renderDashboard()` | Populate summary/catchment/suitability grids |
| `renderMap()` | Plot pond, alternatives, catchment boundary |
| `loadGISLayers()` | Fetch `/gis/land-context` → OSM overlays |
| `exportLocationKml/kmz` | POST `/exportLocationKml` |

### Map Layers

```
Base: OpenStreetMap + Esri Satellite
Overlays:
  - Recommended Pond (blue marker)
  - Alternatives (gray markers)
  - Catchment Boundary (green polygon)
  - Rivers (blue lines)
  - Water Bodies (blue polygons)
  - Roads (brown lines)
  - Buildings (brown polygons)
```

---

## Configuration

All settings via environment variables (see `config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_RESOLUTION_M` | 5.0 | DEM grid resolution (m) |
| `POND_MAX_SLOPE_DEGREES` | 35.0 | Hard slope limit |
| `POND_MIN_CATCHMENT_AREA_M2` | 0 | Min catchment for eligibility |
| `POND_MIN_RAINFALL_MM` | 0 | Min rainfall for eligibility |
| `OVERPASS_URL` | overpass-api.de | Primary Overpass endpoint |
| `OVERPASS_FALLBACK_URL` | overpass.kumi.systems | Fallback Overpass |
| `OVERPASS_TIMEOUT_SECONDS` | 25.0 | Request timeout |
| `OVERPASS_RETRIES` | 2 | Retry count |
| `POND_RUNOFF_COEFFICIENT` | 0.3 | Rational method C |
| `POND_SEASONAL_MONTHS` | 6,7,8,9 | Monsoon months |
| `POND_RAINFALL_YEARS` | 5 | Historical years |
| `CACHE_DIR` | `.cache` | File cache directory |
| `CACHE_TTL_SECONDS` | 3600 | Cache TTL (1hr) |

---

## Sample Input/Output

### Sample KML (`sample_contours.kml`)

```xml
<Folder>
  <Placemark>
    <name>277.0</name>
    <LineString>
      <coordinates>81.286,21.263 81.286,21.263 ...</coordinates>
    </LineString>
  </Placemark>
  <Placemark>
    <name>280.0</name>
    <LineString>...</LineString>
  </Placemark>
  <!-- 12 contours from 274m to 280m -->
</Folder>
```

### Running the Sample

```bash
curl -X POST http://localhost:8000/analyzeContour \
  -F "file=@sample_contours.kml"
```

### Expected Output (truncated)

```json
{
  "status": "success",
  "pond_candidate": {
    "id": "primary",
    "latitude": 21.2008,
    "longitude": 81.3999,
    "elevation_m": 105.0,
    "slope_degrees": 4.2,
    "catchment_area_m2": 150000.0,
    "suitability_score": 0.78,
    "feasibility": "feasible",
    "road_distance_m": 120.0,
    "water_distance_m": 50.0
  },
  "alternative_candidates": [...],
  "catchment": {
    "area_m2": 150000.0,
    "area_hectares": 15.0,
    "boundary": { "type": "Polygon", "coordinates": [...] }
  },
  "rainfall_runoff": {
    "rainfall_mm": 1150.0,
    "mean_annual_seasonal_rainfall_mm": 230.0,
    "estimated_runoff_m3": 10350.0,
    "estimated_storage_m3": 5175.0,
    "water_availability_status": "good"
  },
  "suitability": {
    "component_scores": {
      "hydrology": 0.82,
      "terrain": 0.85,
      "geography": 0.76,
      "rainfall": 0.71,
      "runoff": 0.79,
      "storage": 0.74
    },
    "overall_score": 0.78
  },
  "recommendation": {
    "best_location": {"latitude": 21.2008, "longitude": 81.3999},
    "suitability_score": 0.78,
    "explanation": "overall=0.782; confidence=0.84; hydrology=0.82; ..."
  }
}
```

---

## Testing

```bash
# All tests
pytest

# Unit suites
pytest tests/test_hydrology_engine.py -v
pytest tests/test_recommendation_engine.py -v
pytest tests/test_candidate_generation.py -v
pytest tests/test_geographic_context.py -v
pytest tests/test_rainfall_runoff.py -v

# Baseline regression (schema locking)
pytest tests/test_baseline_regression.py -v
```

---

## Performance & Benchmarks

| Stage | Time (sample) | Memory |
|-------|--------------|--------|
| Parse | ~50ms | - |
| DEM | ~200ms | ~50MB |
| Hydrology | ~500ms | ~100MB |
| Catchment | ~100ms | - |
| Candidates | ~200ms | - |
| Enrichment | 2-5s (network) | - |
| **Total** | **~3-8s** | ~400-600MB |

---

## Limitations

| Limitation | Severity | Mitigation |
|------------|----------|------------|
| No depression filling before hydrology | High | Deferred to Phase 8 |
| Synchronous CPU in async handlers | High | Thread pool / task queue (Phase 8) |
| Pure-Python D8 double loop | High | Numba/Numba-CUDA (Phase 8) |
| `rasterio` not in requirements | Medium | Add to requirements.txt |
| No climate projections | Medium | Phase 8+ |
| No distributed workers | Low | Phase 8+ |

---

## Security

- No authentication (internal tool)
- Input validation on all endpoints
- File size limit: 100 MB
- File type validation: KML/KMZ only
- No secret logging (redacted in logs)
- `.env` gitignored

---

## License

MIT License