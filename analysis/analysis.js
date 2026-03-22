/**
 * Analysis JS - Single-run trace with cross section view
 */

(function() {

  const DEFAULTS = { pointSize: 1 };

  function sliderToSize(v) {
    return Math.exp(0.5 * v);
  }

  // ── State ──

  const state = {
    metadata: null,
    bounds: null,
    overlays: [],
    slicePlane: null,
    slicePoints: null,
    originalColors: null,
  };

  // ── DOM refs ──

  const statusEl = document.getElementById('status');
  const container3d = document.getElementById('viewport-3d');
  const container2d = document.getElementById('cross-section-view');
  const csInfo = document.getElementById('cross-section-info');
  const traceStatus = document.getElementById('trace-status');
  const btnTrace = document.getElementById('btn-trace');
  const btnTraceStations = document.getElementById('btn-trace-stations');
  const btnReset = document.getElementById('btn-reset');
  const sliceZEl = document.getElementById('slice-z');
  const sliceThickEl = document.getElementById('slice-thick');
  const traceZStepEl = document.getElementById('trace-zstep');
  const traceClaimEl = document.getElementById('trace-claim');
  const traceBranchEl = document.getElementById('trace-branch');
  const traceBudgetEl = document.getElementById('trace-budget');
  const toggleSpokesEl = document.getElementById('toggle-spokes');

  // ── Viewer ──

  const viewer = new PCOViewer(container3d, {
    pointSize: sliderToSize(DEFAULTS.pointSize),
  });
  viewer.initSliceView(container2d);

  // ── Slice plane ──

  function createSlicePlane() {
    const group = new THREE.Group();
    const geo = new THREE.BoxGeometry(1, 1, 1);
    group.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: 0xe8a44a, transparent: true, opacity: 0.4,
      side: THREE.DoubleSide, depthWrite: false,
    })));
    group.add(new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x000000 })
    ));
    state.slicePlane = group;
    group.visible = false;  // starts hidden, toggled by checkbox
    viewer.addOverlay(state.slicePlane);
  }

  function updateSlicePlane(z, thickness) {
    if (!state.slicePlane || !state.bounds) return;
    const b = state.bounds;
    state.slicePlane.scale.set(
      (b.max[0] - b.min[0]) * 1.2,
      (b.max[1] - b.min[1]) * 1.2,
      thickness
    );
    state.slicePlane.position.set(
      (b.max[0] + b.min[0]) / 2,
      (b.max[1] + b.min[1]) / 2,
      z
    );
  }

  // ── Cross section (client-side) ──

  function updateCrossSection(z, thickness) {
    const positions = viewer.positions;
    if (!positions) return;

    const pointCloud = viewer.scene.children.find(c => c.isPoints);
    const colorAttr = pointCloud && pointCloud.geometry.attributes.color;
    const colors = colorAttr ? colorAttr.array : null;

    const halfT = thickness / 2;
    const n = positions.length / 3;

    let count = 0;
    for (let i = 0; i < n; i++) {
      if (Math.abs(positions[i * 3 + 2] - z) <= halfT) count++;
    }

    const slicePos = new Float32Array(count * 3);
    const sliceCol = colors ? new Float32Array(count * 3) : null;

    let idx = 0;
    for (let i = 0; i < n; i++) {
      if (Math.abs(positions[i * 3 + 2] - z) <= halfT) {
        slicePos[idx * 3]     = positions[i * 3];
        slicePos[idx * 3 + 1] = positions[i * 3 + 1];
        slicePos[idx * 3 + 2] = z;
        if (sliceCol) {
          sliceCol[idx * 3]     = colors[i * 3];
          sliceCol[idx * 3 + 1] = colors[i * 3 + 1];
          sliceCol[idx * 3 + 2] = colors[i * 3 + 2];
        }
        idx++;
      }
    }

    if (state.slicePoints) {
      state.slicePoints.geometry.dispose();
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(slicePos, 3));
      if (sliceCol) geo.setAttribute('color', new THREE.Float32BufferAttribute(sliceCol, 3));
      state.slicePoints.geometry = geo;
    } else {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(slicePos, 3));
      if (sliceCol) geo.setAttribute('color', new THREE.Float32BufferAttribute(sliceCol, 3));
      state.slicePoints = new THREE.Points(geo, new THREE.PointsMaterial({
        size: sliderToSize(parseInt(document.getElementById('point-size').value)),
        vertexColors: !!sliceCol, sizeAttenuation: false, depthTest: false,
      }));
      viewer.addToSlice(state.slicePoints);
    }

    viewer.updateSliceView(state.bounds, z, thickness);
    updateCsAxes();
    csInfo.textContent = `z=${z.toFixed(1)}  thick=${thickness.toFixed(2)}  pts=${count.toLocaleString()}`;
  }

  // ── 2D axis overlay ──

  function updateCsAxes() {
    const svg = document.getElementById('cs-axes');
    if (!svg || !state.bounds) return;

    const rect = svg.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w === 0 || h === 0) return;

    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

    const b = state.bounds;
    const xMin = b.min[0], xMax = b.max[0];
    const yMin = b.min[1], yMax = b.max[1];
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    // Match the ortho camera's fitting logic
    const aspect = w / h;
    const pad = 1.15;
    let halfW, halfH;
    if (xRange / yRange > aspect) {
      halfW = xRange * pad / 2;
      halfH = halfW / aspect;
    } else {
      halfH = yRange * pad / 2;
      halfW = halfH * aspect;
    }

    const cx = (xMin + xMax) / 2;
    const cy = (yMin + yMax) / 2;
    const viewXMin = cx - halfW;
    const viewXMax = cx + halfW;
    const viewYMin = cy - halfH;
    const viewYMax = cy + halfH;

    // World to pixel
    function wx(x) { return (x - viewXMin) / (viewXMax - viewXMin) * w; }
    function wy(y) { return (1 - (y - viewYMin) / (viewYMax - viewYMin)) * h; }  // flip Y

    // Compute tick spacing
    const viewRange = Math.max(viewXMax - viewXMin, viewYMax - viewYMin);
    const rawStep = viewRange / 8;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const candidates = [1, 2, 5, 10];
    let tickStep = mag;
    for (const c of candidates) {
      if (c * mag >= rawStep) { tickStep = c * mag; break; }
    }

    const margin = 30;  // pixels from edge for axis line
    let html = '';

    // Left axis (Y)
    const axX = margin;
    html += `<line x1="${axX}" y1="0" x2="${axX}" y2="${h}" stroke-opacity="0.4"/>`;

    for (let v = Math.ceil(viewYMin / tickStep) * tickStep; v <= viewYMax; v += tickStep) {
      const py = wy(v);
      if (py < 5 || py > h - 5) continue;
      html += `<line x1="${axX - 3}" y1="${py}" x2="${axX}" y2="${py}" stroke-opacity="0.4"/>`;
      html += `<text class="tick-label" x="${axX - 5}" y="${py + 3}" text-anchor="end">${v.toFixed(1)}</text>`;
    }
    html += `<text class="axis-label" x="12" y="${h/2}" transform="rotate(-90, 12, ${h/2})">Y</text>`;

    // Bottom axis (X)
    const axY = h - margin;
    html += `<line x1="0" y1="${axY}" x2="${w}" y2="${axY}" stroke-opacity="0.4"/>`;

    for (let v = Math.ceil(viewXMin / tickStep) * tickStep; v <= viewXMax; v += tickStep) {
      const px = wx(v);
      if (px < margin + 5 || px > w - 5) continue;
      html += `<line x1="${px}" y1="${axY}" x2="${px}" y2="${axY + 3}" stroke-opacity="0.4"/>`;
      html += `<text class="tick-label" x="${px}" y="${axY + 13}" text-anchor="middle">${v.toFixed(1)}</text>`;
    }
    html += `<text class="axis-label" x="${w/2}" y="${h - 5}" text-anchor="middle">X</text>`;

    svg.innerHTML = html;
  }

  function refreshSlice() {
    updateCrossSection(parseFloat(sliceZEl.value), parseFloat(sliceThickEl.value));
  }

  // ── Sliders ──

  document.getElementById('point-size').value = DEFAULTS.pointSize;
  document.getElementById('val-psize').textContent = DEFAULTS.pointSize;
  document.getElementById('point-size').addEventListener('input', function() {
    const v = parseInt(this.value);
    document.getElementById('val-psize').textContent = v;
    viewer.setPointSize(sliderToSize(v));
    if (state.slicePoints) state.slicePoints.material.size = sliderToSize(v);
  });

  function onSliceChange() {
    const z = parseFloat(sliceZEl.value);
    const thick = parseFloat(sliceThickEl.value);
    document.getElementById('val-slice-z').textContent = z.toFixed(1);
    document.getElementById('val-slice-thick').textContent = thick.toFixed(2);
    updateSlicePlane(z, thick);
    updateCrossSection(z, thick);
  }

  sliceZEl.addEventListener('input', onSliceChange);
  sliceThickEl.addEventListener('input', onSliceChange);

  traceZStepEl.addEventListener('input', function() {
    document.getElementById('val-zstep').textContent = parseFloat(this.value).toFixed(2);
  });
  traceClaimEl.addEventListener('input', function() {
    document.getElementById('val-claim').textContent = parseFloat(this.value).toFixed(2);
  });
  traceBranchEl.addEventListener('input', function() {
    document.getElementById('val-branch').textContent = parseInt(this.value);
  });
  traceBudgetEl.addEventListener('input', function() {
    document.getElementById('val-budget').textContent = parseInt(this.value);
  });

  // ── Color helpers ──

  function recolorClaimed(claimedIndices, color) {
    const pointCloud = viewer.scene.children.find(c => c.isPoints);
    if (!pointCloud) return;
    const colorAttr = pointCloud.geometry.attributes.color;
    if (!colorAttr) return;
    const arr = colorAttr.array;
    const r = ((color >> 16) & 0xff) / 255;
    const g = ((color >> 8) & 0xff) / 255;
    const b = (color & 0xff) / 255;
    const offset = viewer.lodOffset || 0;
    for (const idx of claimedIndices) {
      const vi = idx + offset;
      arr[vi * 3] = r; arr[vi * 3 + 1] = g; arr[vi * 3 + 2] = b;
    }
    colorAttr.needsUpdate = true;
  }

  function restoreColors() {
    if (!state.originalColors) return;
    const pointCloud = viewer.scene.children.find(c => c.isPoints);
    if (!pointCloud) return;
    const colorAttr = pointCloud.geometry.attributes.color;
    if (!colorAttr) return;
    colorAttr.array.set(state.originalColors);
    colorAttr.needsUpdate = true;
  }

  // ── Overlay helpers ──

  function clearOverlays() {
    for (const obj of state.overlays) viewer.removeOverlay(obj);
    state.overlays = [];
  }

  function addPolyline(positions, color = 0xe8a44a) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
    viewer.addToBoth(line);
    state.overlays.push(line);
    return line;
  }

  function addCentroids(positions, color = 0xff4444, size = 6) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({
      color, size, sizeAttenuation: false, depthTest: false
    }));
    viewer.addToBoth(pts);
    state.overlays.push(pts);
    return pts;
  }

  // ── Trace ──

  btnTrace.addEventListener('click', async function() {
    const b = state.bounds;
    const seed = [
      (b.min[0] + b.max[0]) / 2 - 2.25,
      (b.min[1] + b.max[1]) / 2,
      b.min[2] + 0.1,
    ];

    const params = {
      seed: seed,
      z_step: parseFloat(traceZStepEl.value),
      claim_radius: parseFloat(traceClaimEl.value),
      min_branch_points: parseInt(traceBranchEl.value),
    };

    traceStatus.textContent = 'tracing...';
    btnTrace.disabled = true;

    try {
      const resp = await fetch('/api/trace', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const result = await resp.json();

      const nFlags = result.branch_flags ? result.branch_flags.length : 0;
      traceStatus.textContent = `${result.steps} steps, ${result.all_claimed.length} pts, ${nFlags} branch flags`;

      // Draw centerline
      const cl = result.centerline;
      const clFlat = new Float32Array(cl.length * 3);
      for (let i = 0; i < cl.length; i++) {
        clFlat[i * 3] = cl[i][0]; clFlat[i * 3 + 1] = cl[i][1]; clFlat[i * 3 + 2] = cl[i][2];
      }
      addPolyline(clFlat, 0x00ff88);
      addCentroids(clFlat, 0xffffff, 3);

      // Recolor claimed green
      recolorClaimed(result.all_claimed, 0x00ff88);

      // Draw branch flag markers (magenta)
      if (result.branch_flags && result.branch_flags.length > 0) {
        const bfFlat = new Float32Array(result.branch_flags.length * 3);
        for (let i = 0; i < result.branch_flags.length; i++) {
          const bf = result.branch_flags[i];
          bfFlat[i * 3] = bf.xy[0];
          bfFlat[i * 3 + 1] = bf.xy[1];
          bfFlat[i * 3 + 2] = bf.z;
        }
        addCentroids(bfFlat, 0xff00ff, 8);
      }

      refreshSlice();

    } catch (e) {
      traceStatus.textContent = 'error: ' + e.message;
    } finally {
      btnTrace.disabled = false;
    }
  });

  // ── Wheel & spoke drawing ──

  function drawStationSpokes(station, color = 0xe8a44a) {
    const z = station.z;
    const cx = station.center[0], cy = station.center[1];
    const boundary = station.boundary; // Nx2 array
    const n = boundary.length;

    // Rim: closed polyline around boundary at this Z
    const rimPositions = new Float32Array((n + 1) * 3);
    for (let i = 0; i <= n; i++) {
      const bi = i % n;
      rimPositions[i * 3]     = boundary[bi][0];
      rimPositions[i * 3 + 1] = boundary[bi][1];
      rimPositions[i * 3 + 2] = z;
    }
    const rimGeo = new THREE.BufferGeometry();
    rimGeo.setAttribute('position', new THREE.Float32BufferAttribute(rimPositions, 3));
    const rim = new THREE.Line(rimGeo, new THREE.LineBasicMaterial({
      color, transparent: true, opacity: 0.7, depthTest: false,
    }));
    viewer.addToBoth(rim);
    state.overlays.push(rim);

    // Spokes: lines from center to each boundary vertex
    const spokePositions = new Float32Array(n * 6);
    for (let i = 0; i < n; i++) {
      spokePositions[i * 6]     = cx;
      spokePositions[i * 6 + 1] = cy;
      spokePositions[i * 6 + 2] = z;
      spokePositions[i * 6 + 3] = boundary[i][0];
      spokePositions[i * 6 + 4] = boundary[i][1];
      spokePositions[i * 6 + 5] = z;
    }
    const spokeGeo = new THREE.BufferGeometry();
    spokeGeo.setAttribute('position', new THREE.Float32BufferAttribute(spokePositions, 3));
    const spokes = new THREE.LineSegments(spokeGeo, new THREE.LineBasicMaterial({
      color, transparent: true, opacity: 0.25, depthTest: false,
    }));
    viewer.addToBoth(spokes);
    state.overlays.push(spokes);
  }

  // ── Station trace (progressive polling) ──

  let pollTimer = null;

  function applyProgressUpdate(snap) {
    // Lightweight: just update claimed colors + centerline
    restoreColors();
    clearOverlays();

    if (snap.all_claimed && snap.all_claimed.length > 0) {
      recolorClaimed(snap.all_claimed, 0x00ff88);
    }

    if (snap.centerline && snap.centerline.length > 1) {
      const cl = snap.centerline;
      const clFlat = new Float32Array(cl.length * 3);
      for (let i = 0; i < cl.length; i++) {
        clFlat[i * 3] = cl[i][0]; clFlat[i * 3 + 1] = cl[i][1]; clFlat[i * 3 + 2] = cl[i][2];
      }
      addPolyline(clFlat, 0x00ff88);
      addCentroids(clFlat, 0xffffff, 3);
    }
  }

  function applyFinalResult(result) {
    // Full render with stations + branch flags
    restoreColors();
    clearOverlays();

    if (result.all_claimed && result.all_claimed.length > 0) {
      recolorClaimed(result.all_claimed, 0x00ff88);
    }

    if (result.centerline && result.centerline.length > 1) {
      const cl = result.centerline;
      const clFlat = new Float32Array(cl.length * 3);
      for (let i = 0; i < cl.length; i++) {
        clFlat[i * 3] = cl[i][0]; clFlat[i * 3 + 1] = cl[i][1]; clFlat[i * 3 + 2] = cl[i][2];
      }
      addPolyline(clFlat, 0x00ff88);
      addCentroids(clFlat, 0xffffff, 3);
    }

    if (toggleSpokesEl.checked && result.stations) {
      const maxSteps = result.steps || result.n_stations || 1;
      for (const station of result.stations) {
        if (station.boundary && station.boundary.length > 0) {
          const t = Math.min(1, station.age / Math.max(maxSteps, 1));
          const r = Math.round(0xe8 * (1 - t * 0.4));
          const g = Math.round(0xa4 * (1 - t * 0.3));
          const b = Math.round(0x4a + t * 0x40);
          const color = (r << 16) | (g << 8) | b;
          drawStationSpokes(station, color);
        }
      }
    }

    if (result.branch_flags && result.branch_flags.length > 0) {
      const bfFlat = new Float32Array(result.branch_flags.length * 3);
      for (let i = 0; i < result.branch_flags.length; i++) {
        const bf = result.branch_flags[i];
        bfFlat[i * 3] = bf.xy[0];
        bfFlat[i * 3 + 1] = bf.xy[1];
        bfFlat[i * 3 + 2] = bf.z;
      }
      addCentroids(bfFlat, 0xff00ff, 8);
    }
  }

  async function pollProgress() {
    try {
      const resp = await fetch('/api/trace_stations_progress');
      const snap = await resp.json();

      const phase = snap.phase || '?';
      const step = snap.step || 0;
      const nStations = snap.n_stations || (snap.stations ? snap.stations.length : 0);
      const nClaimed = snap.all_claimed ? snap.all_claimed.length : 0;

      traceStatus.textContent = `${phase} step=${step} stations=${nStations} claimed=${nClaimed}`;

      if (snap.done) {
        clearInterval(pollTimer);
        pollTimer = null;
        btnTraceStations.disabled = false;
        applyFinalResult(snap);
        traceStatus.textContent = `done — ${snap.steps || step} steps, ${nStations} stations, ${nClaimed} pts`;
        refreshSlice();
      } else if (snap.all_claimed) {
        applyProgressUpdate(snap);
      }
    } catch (e) {
      traceStatus.textContent = 'poll error: ' + e.message;
    }
  }

  btnTraceStations.addEventListener('click', async function() {
    const b = state.bounds;
    const seed = [
      (b.min[0] + b.max[0]) / 2 - 2.25,
      (b.min[1] + b.max[1]) / 2,
      b.min[2] + 0.1,
    ];

    const params = {
      seed: seed,
      z_step: parseFloat(traceZStepEl.value),
      claim_radius: parseFloat(traceClaimEl.value),
      min_branch_points: parseInt(traceBranchEl.value),
      nudge_budget: parseInt(traceBudgetEl.value),
      use_tip_center: true,
      progress_interval: 1,
    };

    // Clear previous
    clearOverlays();
    restoreColors();

    traceStatus.textContent = 'starting trace...';
    btnTraceStations.disabled = true;

    try {
      await fetch('/api/trace_stations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });

      // Start polling
      pollTimer = setInterval(pollProgress, 500);

    } catch (e) {
      traceStatus.textContent = 'error: ' + e.message;
      btnTraceStations.disabled = false;
    }
  });

  // ── Reset ──

  btnReset.addEventListener('click', async function() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    traceStatus.textContent = 'resetting...';
    await fetch('/api/reset', { method: 'POST' });
    clearOverlays();
    restoreColors();
    refreshSlice();
    btnTraceStations.disabled = false;
    traceStatus.textContent = 'ready';
  });

  // ── Init ──

  (async () => {
    statusEl.textContent = 'loading...';

    // Start render loop immediately so progressive chunks are visible
    viewer.start();

    const progressBar = document.getElementById('load-progress');
    const progressFill = document.getElementById('load-progress-bar');
    const statPts = document.getElementById('stat-points');
    const statNodes = document.getElementById('stat-nodes');
    progressBar.style.display = 'block';

    const meta = await viewer.loadFromServer('/metadata', '/level', 10000, (info) => {
      statPts.textContent = info.loadedPoints.toLocaleString();
      statNodes.textContent = info.loadedNodes.toLocaleString();
      if (info.phase === 'lod') {
        statusEl.textContent = `loading LOD... ${info.loadedPoints.toLocaleString()} pts`;
        progressFill.style.width = '2%';
      } else if (info.totalNodes > 0) {
        const pct = Math.min(100, (info.loadedNodes / info.totalNodes) * 100);
        progressFill.style.width = pct + '%';
        statusEl.textContent = `loading... ${info.loadedPoints.toLocaleString()} pts`;
      }
    });
    state.metadata = meta;

    progressBar.style.display = 'none';
    statPts.textContent = meta.total_points.toLocaleString();
    statNodes.textContent = meta.num_nodes.toLocaleString();

    const boundsResp = await fetch('/api/bounds');
    state.bounds = await boundsResp.json();

    // Place axes at bottom-left of point cloud
    viewer.placeAxes(state.bounds);

    // Load node bounds for grid overlay
    const nbResp = await fetch('/api/all_node_bounds');
    const allNodeBounds = await nbResp.json();
    viewer.setNodeBounds(allNodeBounds);

    // Wire up node bounds toggle
    const nbToggle = document.getElementById('toggle-nodes');
    if (nbToggle) {
      nbToggle.addEventListener('change', function() {
        viewer.toggleNodeBounds(this.checked);
      });
    }

    // Wire up z-slice toggle
    const sliceToggle = document.getElementById('toggle-slice');
    if (sliceToggle) {
      sliceToggle.addEventListener('change', function() {
        if (state.slicePlane) state.slicePlane.visible = this.checked;
      });
    }

    // Store original colors
    const pointCloud = viewer.scene.children.find(c => c.isPoints);
    if (pointCloud && pointCloud.geometry.attributes.color) {
      state.originalColors = new Float32Array(pointCloud.geometry.attributes.color.array);
    }

    const zMin = state.bounds.min[2];
    const zMax = state.bounds.max[2];
    sliceZEl.min = zMin.toFixed(1);
    sliceZEl.max = zMax.toFixed(1);
    sliceZEl.value = ((zMin + zMax) / 2).toFixed(1);
    document.getElementById('val-slice-z').textContent = sliceZEl.value;

    createSlicePlane();
    onSliceChange();

    btnTrace.disabled = false;
    btnTraceStations.disabled = false;
    btnReset.disabled = false;

    statusEl.textContent = '';
  })().catch(err => { statusEl.textContent = 'error: ' + err.message; });

})();