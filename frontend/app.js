const API_URL = "http://localhost:8000/analyzeContour";

const el = (id, fallbackId = null) => document.getElementById(id) || (fallbackId ? document.getElementById(fallbackId) : null);

const fileInput = el("contour-file", "contourFile");
const selectedFilename = el("selectedFilename");
const analyzeButton = el("analyze-button", "analyzeButton");
const loadingIndicator = el("loading", "loadingIndicator");
const errorMessage = el("error-message", "errorMessage");
const summaryGrid = el("summaryGrid");
const catchmentGrid = el("catchmentGrid");
const suitabilityGrid = el("suitabilityGrid");
const recommendationText = el("recommendationText");
const warningsList = el("warningsList");
const locationSelectButton = el("location-select-button");
const analyzeLocationButton = el("analyze-location-button");
const locationLatitudeInput = el("selected-latitude");
const locationLongitudeInput = el("selected-longitude");
const locationRadiusInput = el("location-radius");
const locationLoadingIndicator = el("location-loading");
const locationAnalysisResult = el("location-analysis-result");
const exportLocationKmlButton = el("export-location-kml-button");
const exportLocationKmzButton = el("export-location-kmz-button");

let map;
let baseLayers = {};
let overlayLayers = {};
let recommendedPondLayer;
let alternativeCandidatesLayer;
let catchmentLayer;
let riversLayer = null;
let waterBodiesLayer = null;
let roadsLayer = null;
let buildingsLayer = null;
let layerControl;
let locationSelectionMode = false;
let selectedLocation = null;
let locationSelectionMarker = null;
let locationAnalysisCircle = null;
let latestLocationAnalysis = null;

async function exportLocationAnalysis(endpoint, filename) {
  if (!latestLocationAnalysis) return;
  const response = await fetch(`http://localhost:8000/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(latestLocationAnalysis),
  });
  if (!response.ok) throw new Error(`Export failed with HTTP ${response.status}.`);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function getValue(source, path, fallback = null) {
  return path.reduce((value, key) => {
    if (value && Object.prototype.hasOwnProperty.call(value, key)) {
      return value[key];
    }
    return undefined;
  }, source) ?? fallback;
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "Unavailable";
  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function formatText(value) {
  if (value === null || value === undefined || value === "") return "Unavailable";
  return String(value);
}

function clearNode(node) {
  if (!node) return;
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function addMetric(container, label, value) {
  const item = document.createElement("div");
  item.className = "metric";

  const labelNode = document.createElement("span");
  labelNode.textContent = label;

  const valueNode = document.createElement("strong");
  valueNode.textContent = value;

  item.append(labelNode, valueNode);
  container.appendChild(item);
}

function setLoading(isLoading) {
  if (analyzeButton) analyzeButton.disabled = isLoading;
  if (loadingIndicator) loadingIndicator.hidden = !isLoading;
}

function setLocationLoading(isLoading) {
  if (analyzeLocationButton) {
    analyzeLocationButton.disabled = isLoading || !selectedLocation;
  }
  if (locationLoadingIndicator) {
    locationLoadingIndicator.hidden = !isLoading;
  }
}

function showError(message) {
  if (errorMessage) errorMessage.textContent = message;
}

function validateFile(file) {
  if (!file) {
    return "Select a KML or KMZ file first.";
  }
  const name = file.name.toLowerCase();
  if (!name.endsWith(".kml") && !name.endsWith(".kmz")) {
    return "Unsupported file extension. Upload a .kml or .kmz file.";
  }
  return null;
}

function updateSelectedLocationDisplay() {
  if (!locationLatitudeInput || !locationLongitudeInput) return;

  if (!selectedLocation) {
    locationLatitudeInput.value = "Not selected";
    locationLongitudeInput.value = "Not selected";
    if (analyzeLocationButton) analyzeLocationButton.disabled = true;
    return;
  }

  locationLatitudeInput.value = Number(selectedLocation.latitude).toFixed(6);
  locationLongitudeInput.value = Number(selectedLocation.longitude).toFixed(6);
  if (analyzeLocationButton) analyzeLocationButton.disabled = false;
}

function clearLocationAnalysisCircle() {
  if (locationAnalysisCircle && map) {
    map.removeLayer(locationAnalysisCircle);
    locationAnalysisCircle = null;
  }
}

function setLocationSelectionMode(enabled) {
  locationSelectionMode = enabled;
  if (locationSelectButton) {
    locationSelectButton.textContent = enabled ? "Selection Mode: Active" : "Select Location on Map";
    locationSelectButton.classList.toggle("active", enabled);
  }
}

function renderLocationMarker(latlng) {
  if (!map) return;

  if (!locationSelectionMarker) {
    locationSelectionMarker = L.marker(latlng, {
      draggable: false,
      title: "Selected location",
    }).addTo(map);
  } else {
    locationSelectionMarker.setLatLng(latlng);
  }

  locationSelectionMarker.bindPopup(`Selected location<br>Lat: ${Number(latlng.lat).toFixed(6)}<br>Lon: ${Number(latlng.lng).toFixed(6)}`);
}

function updateLocationAnalysisArea(latlng, radiusKm) {
  clearLocationAnalysisCircle();
  if (!map || !Number.isFinite(radiusKm) || radiusKm <= 0) return;

  locationAnalysisCircle = L.circle([latlng.lat, latlng.lng], {
    radius: radiusKm * 1000,
    color: "#1e88e5",
    fillColor: "#42a5f5",
    fillOpacity: 0.18,
    weight: 2,
  }).addTo(map);
}

function initMap() {
  if (map) return;

  map = L.map("map", {
    zoomControl: true,
  }).setView([20.5937, 78.9629], 5);

  const streetMap = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  });

  const satelliteMap = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19,
    attribution: "Tiles &copy; Esri",
  });

  baseLayers = {
    "Street Map": streetMap,
    "Satellite": satelliteMap,
  };

  recommendedPondLayer = L.layerGroup().addTo(map);
  alternativeCandidatesLayer = L.layerGroup().addTo(map);
  catchmentLayer = null;

  overlayLayers = {
    "Recommended Pond": recommendedPondLayer,
    "Alternatives": alternativeCandidatesLayer,
  };

  if (L.control && typeof L.control.layers === "function") {
    layerControl = L.control.layers(baseLayers, overlayLayers).addTo(map);
  }
  streetMap.addTo(map);
}

function ensureGISLayerGroups() {
  if (!map) return;

  if (!riversLayer) {
    riversLayer = L.layerGroup().addTo(map);
  }
  if (!waterBodiesLayer) {
    waterBodiesLayer = L.layerGroup().addTo(map);
  }
  if (!roadsLayer) {
    roadsLayer = L.layerGroup().addTo(map);
  }
  if (!buildingsLayer) {
    buildingsLayer = L.layerGroup().addTo(map);
  }

  overlayLayers = {
    ...overlayLayers,
    "Rivers": riversLayer,
    "Water Bodies": waterBodiesLayer,
    "Roads": roadsLayer,
    "Buildings": buildingsLayer,
  };

  if (layerControl && L.control && typeof L.control.layers === "function") {
    layerControl.remove();
    layerControl = L.control.layers(baseLayers, overlayLayers).addTo(map);
  }
}

function popupContent(title, candidate) {
  const wrapper = document.createElement("div");
  wrapper.className = "popup-lines";

  const heading = document.createElement("strong");
  heading.textContent = title;
  wrapper.appendChild(heading);

  const rows = [
    ["Latitude", formatNumber(candidate.latitude, 6)],
    ["Longitude", formatNumber(candidate.longitude, 6)],
    ["Elevation", `${formatNumber(candidate.elevation_m)} m`],
    ["Slope", `${formatNumber(candidate.slope_degrees)} deg`],
    ["Catchment Area", `${formatNumber(candidate.catchment_area_m2 ?? getValue(candidate, ["catchment_area", "area_m2"]))} m2`],
    ["Suitability Score", formatNumber(candidate.suitability_score ?? getValue(candidate, ["suitability", "overall_score"]), 3)],
  ];

  rows.forEach(([label, value]) => {
    const line = document.createElement("div");
    line.textContent = `${label}: ${value}`;
    wrapper.appendChild(line);
  });

  return wrapper;
}

function markerIcon(className) {
  return L.divIcon({
    className: "",
    html: `<div class="${className}"></div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });
}

function validLatLon(candidate) {
  const lat = Number(candidate?.latitude);
  const lon = Number(candidate?.longitude);
  return Number.isFinite(lat) && Number.isFinite(lon);
}

function safeBoundsFromBoundary(boundary) {
  if (!boundary || !Array.isArray(boundary.coordinates)) return null;

  const flatten = [];
  const walk = (coords) => {
    if (Array.isArray(coords[0]) && typeof coords[0][0] === "number") {
      coords.forEach((ring) => walk(ring));
      return;
    }
    if (Array.isArray(coords) && coords.length >= 2 && typeof coords[0] === "number" && typeof coords[1] === "number") {
      flatten.push([Number(coords[1]), Number(coords[0])]);
    }
  };

  try {
    walk(boundary.coordinates);
  } catch (error) {
    return null;
  }

  if (!flatten.length) return null;

  return {
    south: Math.min(...flatten.map(([lat]) => lat)),
    north: Math.max(...flatten.map(([lat]) => lat)),
    west: Math.min(...flatten.map(([, lon]) => lon)),
    east: Math.max(...flatten.map(([, lon]) => lon)),
  };
}

function computeAnalysisBounds(payload) {
  const contourExtent = getValue(payload, ["contour_diagnostics", "spatial_extent"], {});
  const candidate = payload.pond_candidate || {};

  const values = [
    [getValue(contourExtent, ["min_latitude"]), getValue(contourExtent, ["max_latitude"]), getValue(contourExtent, ["min_longitude"]), getValue(contourExtent, ["max_longitude"])],
    [Number(candidate.latitude), Number(candidate.latitude), Number(candidate.longitude), Number(candidate.longitude)],
  ];

  const valid = values.filter((entry) => entry.every((value) => Number.isFinite(value)));
  if (valid.length > 0) {
    const [minLat, maxLat, minLon, maxLon] = valid[0];
    if (Number.isFinite(minLat) && Number.isFinite(maxLat) && Number.isFinite(minLon) && Number.isFinite(maxLon)) {
      return { south: minLat, north: maxLat, west: minLon, east: maxLon };
    }
  }

  const boundary = getValue(payload, ["catchment", "boundary"]);
  const boundaryBounds = safeBoundsFromBoundary(boundary);
  if (boundaryBounds) return boundaryBounds;

  return null;
}

function convertLandContextToFeatureCollection(landContext) {
  const groups = {
    rivers: [],
    waterBodies: [],
    roads: [],
    buildings: [],
  };

  const pushFeature = (category, element) => {
    const tags = element?.tags || {};
    const geometry = Array.isArray(element?.geometry) ? element.geometry : [];
    let feature = null;

    if (element?.type === "node") {
      const lon = Number(element.lon);
      const lat = Number(element.lat);
      if (Number.isFinite(lon) && Number.isFinite(lat)) {
        feature = {
          type: "Feature",
          geometry: {
            type: "Point",
            coordinates: [lon, lat],
          },
          properties: { ...tags, category },
        };
      }
    } else if (geometry.length > 1) {
      const coords = geometry.map((point) => [Number(point.lon), Number(point.lat)]).filter((coord) => coord.every(Number.isFinite));
      if (coords.length >= 2) {
        const closed = coords[0][0] === coords[coords.length - 1][0] && coords[0][1] === coords[coords.length - 1][1];
        feature = {
          type: "Feature",
          geometry: {
            type: closed ? "Polygon" : "LineString",
            coordinates: closed ? [coords] : coords,
          },
          properties: { ...tags, category },
        };
      }
    }

    if (feature) {
      groups[category].push(feature);
    }
  };

  const entries = Array.isArray(landContext?.water_bodies) ? landContext.water_bodies : [];
  const roads = Array.isArray(landContext?.roads) ? landContext.roads : [];
  const buildings = Array.isArray(landContext?.buildings) ? landContext.buildings : [];

  entries.forEach((element) => {
    const tags = element?.tags || {};
    const category = tags.waterway ? "rivers" : "waterBodies";
    pushFeature(category, element);
  });
  roads.forEach((element) => pushFeature("roads", element));
  buildings.forEach((element) => pushFeature("buildings", element));

  return groups;
}

function renderGISLayers(landContext) {
  if (!map) return [];

  ensureGISLayerGroups();

  const groups = convertLandContextToFeatureCollection(landContext);
  const warnings = [];

  if (riversLayer) riversLayer.clearLayers();
  if (waterBodiesLayer) waterBodiesLayer.clearLayers();
  if (roadsLayer) roadsLayer.clearLayers();
  if (buildingsLayer) buildingsLayer.clearLayers();

  const renderFeatureGroup = (layerGroup, featureList, style, pointStyle) => {
    if (!featureList.length) return;
    const featureCollection = { type: "FeatureCollection", features: featureList };
    const layer = L.geoJSON(featureCollection, {
      style,
      pointToLayer: pointStyle ? (_, latlng) => L.circleMarker(latlng, pointStyle) : undefined,
    });
    layer.addTo(layerGroup);
  };

  renderFeatureGroup(riversLayer, groups.rivers, {
    color: "#1d8dd8",
    weight: 3,
    opacity: 0.9,
  });

  renderFeatureGroup(waterBodiesLayer, groups.waterBodies, {
    color: "#2d7ef7",
    weight: 2,
    fillColor: "#7ec8ff",
    fillOpacity: 0.45,
  });

  renderFeatureGroup(roadsLayer, groups.roads, {
    color: "#d88b2a",
    weight: 3,
    opacity: 0.8,
  });

  renderFeatureGroup(buildingsLayer, groups.buildings, {
    color: "#9e5b42",
    weight: 1,
    fillColor: "#d9a38a",
    fillOpacity: 0.55,
  });

  if (!Object.values(groups).some((list) => list.length)) {
    warnings.push("No GIS context layers were returned for the analyzed area.");
  }

  return warnings;
}

async function loadGISLayers(payload) {
  if (!map) return [];

  const bounds = computeAnalysisBounds(payload);
  if (!bounds) {
    return ["GIS context could not be determined from the analysis response."];
  }

  try {
    const response = await fetch("http://localhost:8000/gis/land-context", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bbox: bounds }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const detail = payload?.detail || { message: "GIS context request failed." };
      const message = typeof detail === "object" ? detail.message : detail;
      throw new Error(message || "GIS context request failed.");
    }

    const landContext = await response.json();
    return renderGISLayers(landContext);
  } catch (error) {
    return [error && typeof error.message === "string" ? error.message : "External GIS context unavailable."];
  }
}

function renderMap(payload) {
  initMap();
  recommendedPondLayer.clearLayers();
  alternativeCandidatesLayer.clearLayers();
  if (catchmentLayer) {
    catchmentLayer.remove();
    catchmentLayer = null;
  }

  const warnings = [];
  const bounds = L.latLngBounds([]);
  const candidate = payload.pond_candidate;

  if (validLatLon(candidate)) {
    const latLng = [Number(candidate.latitude), Number(candidate.longitude)];
    L.marker(latLng, { icon: markerIcon("candidate-dot") })
      .bindPopup(popupContent("Recommended Pond", candidate))
      .addTo(recommendedPondLayer);
    bounds.extend(latLng);
  } else {
    warnings.push("Pond candidate location is missing or invalid.");
  }

  const alternatives = Array.isArray(payload.alternative_candidates)
    ? payload.alternative_candidates
    : getValue(payload, ["recommendation", "alternatives"], []);

  alternatives.forEach((alternative) => {
    if (!validLatLon(alternative)) return;
    const latLng = [Number(alternative.latitude), Number(alternative.longitude)];
    L.marker(latLng, { icon: markerIcon("candidate-dot alt") })
      .bindPopup(popupContent("Alternative Candidate", alternative))
      .addTo(alternativeCandidatesLayer);
    bounds.extend(latLng);
  });

  const boundary = getValue(payload, ["catchment", "boundary"]);
  if (boundary && (boundary.type === "Polygon" || boundary.type === "MultiPolygon")) {
    try {
      catchmentLayer = L.geoJSON(boundary, {
        style: {
          color: "#1f7a5a",
          weight: 2,
          fillColor: "#8bbf9f",
          fillOpacity: 0.28,
        },
      }).addTo(map);

      const catchmentBounds = catchmentLayer.getBounds();
      if (catchmentBounds.isValid()) {
        map.fitBounds(catchmentBounds, { padding: [24, 24] });
      }
    } catch (error) {
      warnings.push("Catchment boundary could not be rendered.");
    }
  } else {
    warnings.push("Catchment boundary is missing or not a Polygon/MultiPolygon.");
  }

  if (!boundary && bounds.isValid()) {
    map.fitBounds(bounds, { padding: [28, 28], maxZoom: 15 });
  }

  return warnings;
}

function renderWarnings(warnings) {
  if (!warningsList) return;
  clearNode(warningsList);
  warnings.filter(Boolean).forEach((warning) => {
    const item = document.createElement("li");
    item.textContent = warning;
    warningsList.appendChild(item);
  });
}

async function renderDashboard(payload) {
  if (summaryGrid) clearNode(summaryGrid);
  if (catchmentGrid) clearNode(catchmentGrid);
  if (suitabilityGrid) clearNode(suitabilityGrid);

  const terrain = payload.terrain || {};
  const demValidation = payload.dem_validation || getValue(payload, ["dem", "quality"], {});
  const candidate = payload.pond_candidate || {};
  const catchment = payload.catchment || payload.catchment_area || {};
  const suitability = payload.suitability || {};

  if (summaryGrid) {
    addMetric(summaryGrid, "DEM Status", formatText(demValidation.status));
    addMetric(summaryGrid, "DEM Score", formatNumber(demValidation.score, 3));
    addMetric(summaryGrid, "Elevation Range", `${formatNumber(terrain.min_elevation_m)} - ${formatNumber(terrain.max_elevation_m)} m`);
    addMetric(summaryGrid, "Mean Slope", `${formatNumber(terrain.mean_slope_degrees)} deg`);
    addMetric(summaryGrid, "Maximum Slope", `${formatNumber(terrain.max_slope_degrees)} deg`);
    addMetric(summaryGrid, "Pond Candidate", validLatLon(candidate) ? `${formatNumber(candidate.latitude, 6)}, ${formatNumber(candidate.longitude, 6)}` : "Unavailable");
    addMetric(summaryGrid, "Pond Elevation", `${formatNumber(candidate.elevation_m)} m`);
    addMetric(summaryGrid, "Pond Slope", `${formatNumber(candidate.slope_degrees)} deg`);
  }

  if (catchmentGrid) {
    addMetric(catchmentGrid, "Area", `${formatNumber(catchment.area_m2)} m2`);
    addMetric(catchmentGrid, "Hectares", formatNumber(catchment.area_hectares, 4));
    addMetric(catchmentGrid, "Cell Count", formatNumber(catchment.cell_count, 0));
  }

  if (suitabilityGrid) {
    addMetric(suitabilityGrid, "Overall Score", formatNumber(suitability.overall_score ?? candidate.suitability_score, 3));
    const components = suitability.component_scores || {};
    addMetric(suitabilityGrid, "Slope Score", formatNumber(components.slope, 3));
    addMetric(suitabilityGrid, "Catchment Score", formatNumber(components.catchment_area, 3));
    addMetric(suitabilityGrid, "Rainfall Score", formatNumber(components.rainfall, 3));
  }

  if (recommendationText) {
    recommendationText.textContent = formatText(
      getValue(payload, ["recommendation", "explanation"], payload.message)
    );
  }

  const mapWarnings = renderMap(payload);
  const gisWarnings = await loadGISLayers(payload);
  renderWarnings([...(Array.isArray(payload.warnings) ? payload.warnings : []), ...mapWarnings, ...gisWarnings]);
}

async function analyzeSelectedFile() {
  const file = fileInput.files?.[0];
  const validationMessage = validateFile(file);
  if (validationMessage) {
    showError(validationMessage);
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  setLoading(true);
  showError("");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    let payload;
    try {
      payload = await response.json();
    } catch (error) {
      throw new Error("Backend returned malformed JSON.");
    }

    if (!payload || typeof payload !== "object") {
      throw new Error("Backend returned a malformed analysis response.");
    }

    if (!response.ok) {
      const detail = payload?.detail;
      const message = typeof detail === "object" ? detail.message : detail;
      throw new Error(message || `Backend request failed with HTTP ${response.status}.`);
    }

    await renderDashboard(payload);
  } catch (error) {
    showError(error && typeof error.message === "string" ? error.message : "Network request failed.");
  } finally {
    setLoading(false);
  }
}

if (fileInput) {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (selectedFilename) {
      selectedFilename.textContent = file ? file.name : "No file selected";
    }
    showError("");
  });
}

if (locationSelectButton) {
  locationSelectButton.addEventListener("click", () => {
    setLocationSelectionMode(!locationSelectionMode);
    if (!locationSelectionMode) {
      showError("");
    }
  });
}

if (analyzeButton) {
  analyzeButton.addEventListener("click", (event) => {
    event.preventDefault();
    analyzeSelectedFile();
  });
}

if (analyzeLocationButton) {
  analyzeLocationButton.addEventListener("click", async () => {
    if (!selectedLocation) {
      showError("Select a location on the map before analyzing.");
      return;
    }

    const radiusValue = Number(locationRadiusInput?.value);
    if (!Number.isFinite(radiusValue) || radiusValue <= 0 || radiusValue > 100) {
      showError("Radius must be a positive value up to 100 km.");
      return;
    }

    const payload = {
      latitude: Number(selectedLocation.latitude),
      longitude: Number(selectedLocation.longitude),
      radius_km: radiusValue,
    };

    setLocationLoading(true);
    showError("");

    try {
      const response = await fetch("http://localhost:8000/analyzeLocation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      let data;
      try {
        data = await response.json();
      } catch (error) {
        throw new Error("Backend returned malformed JSON for the location analysis request.");
      }

      if (!response.ok) {
        const detail = data?.detail;
        const message = typeof detail === "object" ? detail.message : detail;
        throw new Error(message || `Location analysis failed with HTTP ${response.status}.`);
      }

      if (!data || typeof data !== "object") {
        throw new Error("Backend returned a malformed location response.");
      }

      latestLocationAnalysis = data;
      if (exportLocationKmlButton) exportLocationKmlButton.disabled = data.status !== "success";
      if (exportLocationKmzButton) exportLocationKmzButton.disabled = data.status !== "success";

      const center = data.center || {};
      const boundsSummary = `Center: ${Number(center.latitude).toFixed(6)}, ${Number(center.longitude).toFixed(6)} | Bounds: ${Number(data.min_latitude).toFixed(6)} to ${Number(data.max_latitude).toFixed(6)} lat, ${Number(data.min_longitude).toFixed(6)} to ${Number(data.max_longitude).toFixed(6)} lon`;

      if (locationAnalysisResult) {
        locationAnalysisResult.textContent = boundsSummary;
      }

      if (selectedLocation) {
        updateLocationAnalysisArea(
          { lat: Number(selectedLocation.latitude), lng: Number(selectedLocation.longitude) },
          Number(data.radius_km || radiusValue),
        );
      }

      await renderDashboard(data);

      showError("");
    } catch (error) {
      showError(error && typeof error.message === "string" ? error.message : "Location analysis request failed.");
    } finally {
      setLocationLoading(false);
    }
  });
}

if (exportLocationKmlButton) {
  exportLocationKmlButton.addEventListener("click", async () => {
    try {
      await exportLocationAnalysis("exportLocationKml", "pond-analysis.kml");
    } catch (error) {
      showError(error?.message || "KML export failed.");
    }
  });
}

if (exportLocationKmzButton) {
  exportLocationKmzButton.addEventListener("click", async () => {
    try {
      await exportLocationAnalysis("exportLocationKmz", "pond-analysis.kmz");
    } catch (error) {
      showError(error?.message || "KMZ export failed.");
    }
  });
}

if (locationRadiusInput) {
  locationRadiusInput.addEventListener("input", () => {
    if (selectedLocation) {
      const radiusValue = Number(locationRadiusInput.value);
      if (Number.isFinite(radiusValue) && radiusValue > 0) {
        updateLocationAnalysisArea(
          { lat: Number(selectedLocation.latitude), lng: Number(selectedLocation.longitude) },
          radiusValue,
        );
      }
    }
  });
}

initMap();

if (map && typeof map.on === "function") {
  map.on("click", (event) => {
    if (!locationSelectionMode) return;

    selectedLocation = {
      latitude: event.latlng.lat,
      longitude: event.latlng.lng,
    };
    renderLocationMarker(event.latlng);
    updateSelectedLocationDisplay();
    updateLocationAnalysisArea(event.latlng, Number(locationRadiusInput?.value || 10));
    setLocationSelectionMode(false);
    if (locationAnalysisResult) {
      locationAnalysisResult.textContent = `Selected: ${Number(selectedLocation.latitude).toFixed(6)}, ${Number(selectedLocation.longitude).toFixed(6)}.`;
    }
  });
}

window.addEventListener("resize", () => {
  if (map) {
    map.invalidateSize();
  }
});
