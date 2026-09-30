const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function createNode(tagName = "div") {
  return {
    tagName,
    className: "",
    textContent: "",
    hidden: false,
    disabled: false,
    files: [],
    children: [],
    firstChild: null,
    append(...nodes) {
      nodes.forEach((node) => this.appendChild(node));
    },
    appendChild(node) {
      this.children.push(node);
      this.firstChild = this.children[0] || null;
      return node;
    },
    removeChild(node) {
      const index = this.children.indexOf(node);
      if (index >= 0) {
        this.children.splice(index, 1);
      }
      this.firstChild = this.children[0] || null;
      return node;
    },
    addEventListener() {},
  };
}

function createLeafletStub() {
  const state = {
    maps: [],
    groups: [],
    markers: [],
    geojson: [],
    tileUrls: [],
    fitBoundsCalls: 0,
    invalidateSizeCalls: 0,
    removedGeoJson: 0,
  };

  const L = {
    map(id, options) {
      const map = {
        id,
        options,
        setViews: [],
        setView(latLng, zoom) {
          this.setViews.push({ latLng, zoom });
          return this;
        },
        fitBounds(bounds, options) {
          state.fitBoundsCalls += 1;
          this.lastFitBounds = { bounds, options };
          return this;
        },
        invalidateSize() {
          state.invalidateSizeCalls += 1;
          return this;
        },
      };
      state.maps.push(map);
      return map;
    },
    tileLayer(url, options) {
      return {
        url,
        options,
        addTo(map) {
          state.tileUrls.push(url);
          this.map = map;
          return this;
        },
      };
    },
    layerGroup() {
      const group = {
        items: [],
        addTo(map) {
          this.map = map;
          return this;
        },
        clearLayers() {
          this.items = [];
        },
      };
      state.groups.push(group);
      return group;
    },
    marker(latLng, options) {
      const marker = {
        latLng,
        options,
        popup: null,
        bindPopup(content) {
          this.popup = content;
          return this;
        },
        addTo(group) {
          group.items.push(this);
          state.markers.push(this);
          return this;
        },
      };
      return marker;
    },
    divIcon(options) {
      return options;
    },
    geoJSON(data, options) {
      const layer = {
        data,
        options,
        removed: false,
        addTo(map) {
          this.map = map;
          state.geojson.push(this);
          return this;
        },
        getBounds() {
          return {
            sourceType: data.type,
            isValid() {
              return true;
            },
          };
        },
        remove() {
          this.removed = true;
          state.removedGeoJson += 1;
        },
      };
      return layer;
    },
    latLngBounds() {
      return {
        points: [],
        extend(latLng) {
          this.points.push(latLng);
        },
        isValid() {
          return this.points.length > 0;
        },
      };
    },
  };

  return { L, state };
}

function loadApp() {
  const nodes = new Map();
  [
    "contourFile",
    "selectedFilename",
    "analyzeButton",
    "loadingIndicator",
    "errorMessage",
    "summaryGrid",
    "catchmentGrid",
    "suitabilityGrid",
    "recommendationText",
    "warningsList",
    "map",
  ].forEach((id) => nodes.set(id, createNode()));

  const { L, state } = createLeafletStub();
  const context = {
    console,
    FormData: class {
      constructor() {
        this.entries = [];
      }
      append(key, value) {
        this.entries.push([key, value]);
      }
    },
    document: {
      createElement: createNode,
      getElementById(id) {
        return nodes.get(id);
      },
    },
    window: {
      addEventListener() {},
    },
    L,
  };

  vm.createContext(context);
  const appPath = path.resolve(__dirname, "..", "frontend", "app.js");
  vm.runInContext(fs.readFileSync(appPath, "utf8"), context, {
    filename: appPath,
  });

  return { context, nodes, state };
}

function payload(boundaryType = "Polygon") {
  const polygon = {
    type: "Polygon",
    coordinates: [[
      [81.1, 21.1],
      [81.2, 21.1],
      [81.2, 21.2],
      [81.1, 21.2],
      [81.1, 21.1],
    ]],
  };
  const multiPolygon = {
    type: "MultiPolygon",
    coordinates: [[polygon.coordinates]],
  };

  return {
    status: "success",
    pond_candidate: {
      latitude: 21.25,
      longitude: 81.35,
      elevation_m: 267,
      slope_degrees: 2.4,
      catchment_area_m2: 3600,
      suitability_score: 0.73,
    },
    alternative_candidates: [
      {
        latitude: 21.26,
        longitude: 81.36,
        elevation_m: 270,
        slope_degrees: 3.1,
        catchment_area_m2: 1800,
        suitability_score: 0.65,
      },
      {
        latitude: 21.27,
        longitude: 81.37,
        elevation_m: 272,
        slope_degrees: 4.5,
        catchment_area_m2: 1500,
        suitability_score: 0.61,
      },
    ],
    catchment: {
      area_m2: 3600,
      boundary: boundaryType === "MultiPolygon" ? multiPolygon : polygon,
    },
    suitability: {
      overall_score: 0.73,
      component_scores: {},
    },
    recommendation: {
      explanation: "Backend recommendation.",
    },
    warnings: [],
  };
}

async function test(name, fn) {
  try {
    await fn();
    console.log(`ok - ${name}`);
  } catch (error) {
    console.error(`not ok - ${name}`);
    throw error;
  }
}

(async () => {
await test("map initializes on startup with OpenStreetMap", () => {
  const { state } = loadApp();
  assert.equal(state.maps.length, 1);
  assert.equal(state.groups.length, 2);
  assert.match(state.tileUrls[0], /openstreetmap/);
});

await test("valid analysis renders recommended pond from backend coordinates", () => {
  const { context, state } = loadApp();
  context.renderMap(payload());
  assert.equal(state.groups[0].items[0].latLng[0], 21.25);
  assert.equal(state.groups[0].items[0].latLng[1], 81.35);
  assert.equal(state.groups[0].items[0].popup.children[0].textContent, "Recommended Pond");
});

await test("all valid alternative candidates are rendered", () => {
  const { context, state } = loadApp();
  context.renderMap(payload());
  assert.equal(state.groups[1].items.length, 2);
});

await test("polygon catchment renders directly as GeoJSON", () => {
  const { context, state } = loadApp();
  context.renderMap(payload("Polygon"));
  assert.equal(state.geojson.length, 1);
  assert.equal(state.geojson[0].data.type, "Polygon");
  assert.equal(state.fitBoundsCalls, 1);
});

await test("multipolygon catchment renders directly as GeoJSON", () => {
  const { context, state } = loadApp();
  context.renderMap(payload("MultiPolygon"));
  assert.equal(state.geojson.length, 1);
  assert.equal(state.geojson[0].data.type, "MultiPolygon");
  assert.equal(state.fitBoundsCalls, 1);
});

await test("repeated analysis clears previous result layers", () => {
  const { context, state } = loadApp();
  context.renderMap(payload());
  context.renderMap(payload());
  assert.equal(state.groups[0].items.length, 1);
  assert.equal(state.groups[1].items.length, 2);
  assert.equal(state.removedGeoJson, 1);
});

await test("missing catchment does not crash and returns warning", () => {
  const { context } = loadApp();
  const nextPayload = payload();
  delete nextPayload.catchment;
  const warnings = context.renderMap(nextPayload);
  assert.ok(warnings.some((warning) => warning.includes("Catchment boundary")));
});

await test("empty alternatives do not crash", () => {
  const { context, state } = loadApp();
  const nextPayload = payload();
  nextPayload.alternative_candidates = [];
  context.renderMap(nextPayload);
  assert.equal(state.groups[1].items.length, 0);
});

await test("backend/API failure shows an error", async () => {
  const { context, nodes } = loadApp();
  nodes.get("contourFile").files = [{ name: "sample_contours.kml" }];
  context.fetch = async () => {
    throw new Error("Backend unavailable");
  };
  await context.analyzeSelectedFile();
  assert.equal(nodes.get("errorMessage").textContent, "Backend unavailable");
});

await test("malformed API response shows an error", async () => {
  const { context, nodes } = loadApp();
  nodes.get("contourFile").files = [{ name: "sample_contours.kml" }];
  context.fetch = async () => ({
    ok: true,
    json: async () => null,
  });
  await context.analyzeSelectedFile();
  assert.equal(
    nodes.get("errorMessage").textContent,
    "Backend returned a malformed analysis response.",
  );
});
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
