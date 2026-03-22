/**
 * PCOViewer - Reusable point cloud viewer with Eye Dome Lighting
 * 
 * Usage:
 *   const viewer = new PCOViewer(containerElement, { edlStrength: 1.0 });
 *   await viewer.loadFromServer('/metadata', '/level');
 *   viewer.start();
 * 
 * The container element should be a positioned div (relative/absolute).
 * The viewer will fill it completely.
 * 
 * Requires Three.js r128 and OrbitControls loaded globally.
 */

class PCOViewer {

  // Layer assignments
  static LAYER_3D = 0;     // Main 3D view (point cloud, EDL)
  static LAYER_SLICE = 1;  // 2D slice view (cross section points, overlays)
  static LAYER_BOTH = 2;   // Visible in both views

  constructor(container, options = {}) {
    this.container = container;
    this.options = Object.assign({
      pointSize: 4,
      edlStrength: 1.0,
      edlRadius: 1.4,
      bgColor: 0x0c0e12,
      showAxes: true,
      axisLength: 12,
    }, options);

    this.scene = new THREE.Scene();         // EDL-processed (point clouds)
    this.overlayScene = new THREE.Scene();  // Rendered after EDL (UI, overlays)
    this.camera = null;
    this.orthoCamera = null;               // 2D slice view camera
    this.controls = null;
    this.renderer = null;
    this.pointsMat = null;
    this.metadata = null;
    this.running = false;

    // 2D slice view
    this._sliceRenderer = null;
    this._sliceContainer = null;
    this._sliceObjects = [];               // objects on LAYER_SLICE

    // Node bounds overlay
    this._nodeBoundsGroup = null;
    this._nodeBoundsVisible = false;

    // EDL internals
    this._edlMaterial = null;
    this._rtEDL = null;
    this._quadScene = null;
    this._quadCamera = null;
    this._neighbourCount = 8;
    this._depthOnlyMat = null;

    this._initRenderer();
    this._initCamera();
    this._initEDL();
    this._initDepthOnly();
    // Axes deferred until bounds are known — call placeAxes(bounds) after loading
  }

  // ── Initialization ──

  _initRenderer() {
    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(this.options.bgColor);
    this.renderer.autoClear = false;
    this.container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
  }

  _initCamera() {
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
    this.camera.position.set(12, 8, 14);
    this.camera.layers.enable(PCOViewer.LAYER_3D);
    this.camera.layers.enable(PCOViewer.LAYER_BOTH);

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
  }

  _initAxes(origin) {
    const len = this.options.axisLength;
    const ox = origin ? origin.x : 0;
    const oy = origin ? origin.y : 0;
    const oz = origin ? origin.z : 0;

    const addAxis = (dx, dy, dz, color) => {
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(ox, oy, oz),
        new THREE.Vector3(ox + dx * len, oy + dy * len, oz + dz * len)
      ]);
      this.overlayScene.add(new THREE.Line(g, new THREE.LineBasicMaterial({
        color, opacity: 0.7, transparent: true
      })));
    };
    addAxis(1, 0, 0, 0xe8a44a);  // X = orange
    addAxis(0, 1, 0, 0x4a9de8);  // Y = blue
    addAxis(0, 0, 1, 0x50c878);  // Z = green

    // Axis label sprites
    const makeLabel = (text, color, x, y, z) => {
      const canvas = document.createElement('canvas');
      canvas.width = 64;
      canvas.height = 64;
      const ctx = canvas.getContext('2d');
      ctx.font = 'bold 48px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = color;
      ctx.fillText(text, 32, 32);

      const tex = new THREE.CanvasTexture(canvas);
      tex.minFilter = THREE.LinearFilter;
      const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.6, depthTest: false });
      const sprite = new THREE.Sprite(mat);
      sprite.position.set(x, y, z);
      sprite.scale.set(1.5, 1.5, 1);
      this.overlayScene.add(sprite);
    };

    const lp = len + 1;
    makeLabel('X', '#e8a44a', ox + lp, oy, oz);
    makeLabel('Y', '#4a9de8', ox, oy + lp, oz);
    makeLabel('Z', '#50c878', ox, oy, oz + lp);
  }

  _initEDL() {
    const nc = this._neighbourCount;

    // Neighbour sample offsets (circle pattern)
    const neighbours = new Float32Array(nc * 2);
    for (let i = 0; i < nc; i++) {
      neighbours[i * 2]     = Math.cos(2 * i * Math.PI / nc);
      neighbours[i * 2 + 1] = Math.sin(2 * i * Math.PI / nc);
    }

    // Render target (float for log-depth in alpha)
    this._rtEDL = new THREE.WebGLRenderTarget(1, 1, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.FloatType,
    });

    // EDL fullscreen quad material
    this._edlMaterial = new THREE.ShaderMaterial({
      uniforms: {
        screenWidth:  { value: 1 },
        screenHeight: { value: 1 },
        edlStrength:  { value: this.options.edlStrength },
        radius:       { value: this.options.edlRadius },
        neighbours:   { value: neighbours },
        uEDLColor:    { value: this._rtEDL.texture },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        #define NEIGHBOUR_COUNT ${nc}
        uniform float screenWidth;
        uniform float screenHeight;
        uniform vec2 neighbours[NEIGHBOUR_COUNT];
        uniform float edlStrength;
        uniform float radius;
        uniform sampler2D uEDLColor;
        varying vec2 vUv;

        float response(float depth) {
          vec2 uvRadius = radius / vec2(screenWidth, screenHeight);
          float sum = 0.0;
          for (int i = 0; i < NEIGHBOUR_COUNT; i++) {
            vec2 uvNeighbor = vUv + uvRadius * neighbours[i];
            float neighbourDepth = texture2D(uEDLColor, uvNeighbor).a;
            neighbourDepth = (neighbourDepth == 1.0) ? 0.0 : neighbourDepth;
            if (neighbourDepth != 0.0) {
              if (depth == 0.0) {
                sum += 100.0;
              } else {
                sum += max(0.0, depth - neighbourDepth);
              }
            }
          }
          return sum / float(NEIGHBOUR_COUNT);
        }

        void main() {
          vec4 cEDL = texture2D(uEDLColor, vUv);
          float depth = cEDL.a;
          depth = (depth == 1.0) ? 0.0 : depth;
          if (depth == 0.0) discard;
          float res = response(depth);
          float shade = exp(-res * 300.0 * edlStrength);
          gl_FragColor = vec4(cEDL.rgb * shade, 1.0);
        }
      `,
      depthTest: false,
      depthWrite: false,
    });

    // Fullscreen quad
    this._quadScene = new THREE.Scene();
    this._quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this._quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this._edlMaterial));
  }

  _initDepthOnly() {
    // Depth-only shader: writes gl_PointSize and depth but no color
    this._depthOnlyMat = new THREE.ShaderMaterial({
      uniforms: {
        pointSize: { value: this.options.pointSize },
      },
      vertexShader: `
        uniform float pointSize;
        void main() {
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mvPos;
          gl_PointSize = pointSize;
        }
      `,
      fragmentShader: `
        void main() {
          gl_FragColor = vec4(0.0);
        }
      `,
      colorWrite: false,
      depthTest: true,
      depthWrite: true,
    });
  }

  // ── Point material (custom shader: writes log2 depth to alpha) ──

  _createPointMaterial(hasColors) {
    this.pointsMat = new THREE.ShaderMaterial({
      uniforms: {
        pointSize: { value: this.options.pointSize },
      },
      vertexShader: `
        uniform float pointSize;
        varying vec3 vColor;
        varying float vLogDepth;
        void main() {
          vColor = color;
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          vLogDepth = log2(-mvPos.z);
          gl_Position = projectionMatrix * mvPos;
          gl_PointSize = pointSize;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        varying float vLogDepth;
        void main() {
          gl_FragColor = vec4(vColor, vLogDepth);
        }
      `,
      vertexColors: hasColors,
      depthTest: true,
      depthWrite: true,
    });
    return this.pointsMat;
  }

  // ── Public API ──

  /** Place axes at the bottom-left corner of the given bounds
   * @param {Object} bounds - {min: [x,y,z], max: [x,y,z]}
   */
  placeAxes(bounds) {
    if (!this.options.showAxes) return;
    const origin = new THREE.Vector3(bounds.min[0], bounds.min[1], bounds.min[2]);
    this._initAxes(origin);
  }

  /** Set point size in pixels */
  setPointSize(size) {
    this.options.pointSize = size;
    if (this.pointsMat) this.pointsMat.uniforms.pointSize.value = size;
    if (this._depthOnlyMat) this._depthOnlyMat.uniforms.pointSize.value = size;
  }

  /** Set EDL strength (0 = off) */
  setEdlStrength(v) {
    this.options.edlStrength = v;
    this._edlMaterial.uniforms.edlStrength.value = v;
  }

  /** Set EDL sample radius in pixels */
  setEdlRadius(v) {
    this.options.edlRadius = v;
    this._edlMaterial.uniforms.radius.value = v;
  }

  /** Focus camera on a center point with a given distance */
  lookAt(center, distance) {
    this.controls.target.copy(center);
    this.camera.position.set(center.x + distance * 0.6, center.y + distance * 0.4, center.z + distance * 0.6);
  }

  /** Add arbitrary Three.js object to the overlay scene (bypasses EDL) */
  addOverlay(object3d) {
    this.overlayScene.add(object3d);
  }

  /** Remove an overlay object */
  removeOverlay(object3d) {
    this.overlayScene.remove(object3d);
  }

  // ── Node Bounds ──

  /**
   * Store node bounds data for lazy building.
   * Geometry is only created when first toggled on.
   * @param {Array} nodes - [{min: [x,y,z], max: [x,y,z], point_count}]
   */
  setNodeBounds(nodes) {
    this._nodeBoundsData = nodes;
    // Don't build geometry yet — wait for toggle
    if (this._nodeBoundsGroup) {
      this.overlayScene.remove(this._nodeBoundsGroup);
      this._nodeBoundsGroup = null;
    }
  }

  /** Build batched wireframe geometry from stored data */
  _buildNodeBounds() {
    const nodes = this._nodeBoundsData;
    if (!nodes || nodes.length === 0) return;

    // Each box = 12 edges × 2 vertices = 24 vertices × 3 floats
    const verts = new Float32Array(nodes.length * 24 * 3);
    let vi = 0;

    for (const node of nodes) {
      const x0 = node.min[0], y0 = node.min[1], z0 = node.min[2];
      const x1 = node.max[0], y1 = node.max[1], z1 = node.max[2];

      const edges = [
        x0,y0,z0, x1,y0,z0,  x1,y0,z0, x1,y1,z0,  x1,y1,z0, x0,y1,z0,  x0,y1,z0, x0,y0,z0,
        x0,y0,z1, x1,y0,z1,  x1,y0,z1, x1,y1,z1,  x1,y1,z1, x0,y1,z1,  x0,y1,z1, x0,y0,z1,
        x0,y0,z0, x0,y0,z1,  x1,y0,z0, x1,y0,z1,  x1,y1,z0, x1,y1,z1,  x0,y1,z0, x0,y1,z1,
      ];
      for (let k = 0; k < edges.length; k++) {
        verts[vi++] = edges[k];
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    const mat = new THREE.LineBasicMaterial({
      color: 0x2a3040, transparent: true, opacity: 0.5,
    });
    this._nodeBoundsGroup = new THREE.LineSegments(geo, mat);
    this._nodeBoundsGroup.visible = true;
    this.overlayScene.add(this._nodeBoundsGroup);
  }

  /**
   * Toggle node bounds visibility.
   * Builds geometry lazily on first enable.
   * @param {boolean} [visible] - if omitted, toggles current state
   * @returns {boolean} new visibility state
   */
  toggleNodeBounds(visible) {
    if (visible === undefined) {
      this._nodeBoundsVisible = !this._nodeBoundsVisible;
    } else {
      this._nodeBoundsVisible = visible;
    }
    if (this._nodeBoundsVisible && !this._nodeBoundsGroup) {
      this._buildNodeBounds();
    }
    if (this._nodeBoundsGroup) {
      this._nodeBoundsGroup.visible = this._nodeBoundsVisible;
    }
    return this._nodeBoundsVisible;
  }

  // ── 2D Slice View ──

  /**
   * Attach a second renderer for the 2D orthographic slice view.
   * @param {HTMLElement} container - div to hold the 2D renderer
   */
  initSliceView(container) {
    this._sliceContainer = container;
    this._sliceRenderer = new THREE.WebGLRenderer({ antialias: true });
    this._sliceRenderer.setPixelRatio(window.devicePixelRatio);
    this._sliceRenderer.setClearColor(this.options.bgColor);
    this._sliceRenderer.domElement.style.display = 'block';
    this._sliceRenderer.domElement.style.width = '100%';
    this._sliceRenderer.domElement.style.height = '100%';
    container.appendChild(this._sliceRenderer.domElement);

    // Ortho camera looking down Z axis
    this.orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1000, 1000);
    this.orthoCamera.position.set(0, 0, 100);
    this.orthoCamera.lookAt(0, 0, 0);
    this.orthoCamera.layers.set(PCOViewer.LAYER_SLICE);
    this.orthoCamera.layers.enable(PCOViewer.LAYER_BOTH);
  }

  /**
   * Update the ortho camera bounds and clipping to match a cross section slab.
   * @param {object} bounds - { min: [x,y,z], max: [x,y,z] }
   * @param {number} z - slice center Z
   * @param {number} thickness - slab thickness
   */
  updateSliceView(bounds, z, thickness) {
    if (!this.orthoCamera) return;

    const xMin = bounds.min[0], xMax = bounds.max[0];
    const yMin = bounds.min[1], yMax = bounds.max[1];
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    // Fit with padding, keeping aspect ratio
    const rect = this._sliceContainer.getBoundingClientRect();
    const aspect = rect.width / rect.height;
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

    this.orthoCamera.left = cx - halfW;
    this.orthoCamera.right = cx + halfW;
    this.orthoCamera.top = cy + halfH;
    this.orthoCamera.bottom = cy - halfH;
    this.orthoCamera.position.set(cx, cy, z + 500);
    this.orthoCamera.lookAt(cx, cy, z);
    this.orthoCamera.near = 500 - thickness / 2;
    this.orthoCamera.far = 500 + thickness / 2;
    this.orthoCamera.updateProjectionMatrix();
  }

  /**
   * Add an object visible only in the slice view.
   * @param {THREE.Object3D} object3d
   */
  addToSlice(object3d) {
    object3d.layers.set(PCOViewer.LAYER_SLICE);
    this.overlayScene.add(object3d);
    this._sliceObjects.push(object3d);
  }

  /**
   * Add an object visible in both 3D and slice views.
   * @param {THREE.Object3D} object3d
   */
  addToBoth(object3d) {
    object3d.layers.set(PCOViewer.LAYER_BOTH);
    this.overlayScene.add(object3d);
  }

  /**
   * Remove all slice-only objects.
   */
  clearSlice() {
    for (const obj of this._sliceObjects) {
      this.overlayScene.remove(obj);
    }
    this._sliceObjects = [];
  }

  // ── Data loading ──

  /** Parse the binary level response format from the PCO server */
  parseLevelResponse(buffer, bytesPerPoint, schema) {
    const view = new DataView(buffer);
    let offset = 0;

    const numNodes = view.getUint32(offset, true); offset += 4;
    const nodes = [];
    for (let i = 0; i < numNodes; i++) {
      const idBytes = new Uint8Array(buffer, offset, 24);
      let nodeId = '';
      for (let j = 0; j < 24 && idBytes[j] !== 0; j++) nodeId += String.fromCharCode(idBytes[j]);
      offset += 24;
      const count = view.getUint32(offset, true); offset += 4;
      nodes.push({ nodeId, count });
    }

    const totalPoints = nodes.reduce((s, n) => s + n.count, 0);
    const hasRGB = (schema & 0b0010) !== 0;

    const positions = new Float32Array(totalPoints * 3);
    const colors = hasRGB ? new Float32Array(totalPoints * 3) : null;

    let ptIdx = 0;
    for (const node of nodes) {
      for (let i = 0; i < node.count; i++) {
        positions[ptIdx * 3]     = view.getFloat32(offset, true); offset += 4;
        positions[ptIdx * 3 + 1] = view.getFloat32(offset, true); offset += 4;
        positions[ptIdx * 3 + 2] = view.getFloat32(offset, true); offset += 4;
        if (hasRGB) {
          colors[ptIdx * 3]     = view.getUint8(offset) / 255; offset += 1;
          colors[ptIdx * 3 + 1] = view.getUint8(offset) / 255; offset += 1;
          colors[ptIdx * 3 + 2] = view.getUint8(offset) / 255; offset += 1;
        }
        ptIdx++;
      }
    }

    return { positions, colors, totalPoints, numNodes: nodes.length };
  }

  /** Load point cloud progressively: levels 0 to max_depth-1 first, then max_depth in chunks
   *  @param {Object} options
   *  @param {string} options.metadataUrl
   *  @param {string} options.levelUrlBase
   *  @param {number} options.chunkSize
   *  @param {function} options.onProgress - called with {loadedPoints, loadedNodes, totalNodes, phase}
   */
  async loadFromServer(metadataUrl = '/metadata', levelUrlBase = '/level', chunkSize = 10000, onProgress = null) {
    const meta = await (await fetch(metadataUrl)).json();
    this.metadata = meta;

    // Auto-frame camera
    const rm = meta.root_min;
    const rs = meta.root_size;
    const center = new THREE.Vector3(rm[0] + rs / 2, rm[1] + rs / 2, rm[2] + rs / 2);
    this.lookAt(center, rs);

    // Track all chunks for final merge
    let allPositions = [];
    let allColors = [];
    let totalPoints = 0;
    let loadedNodes = 0;
    let lodPoints = 0;
    let tempMeshes = [];

    const addChunk = (positions, colors, nNodes) => {
      allPositions.push(positions);
      if (colors) allColors.push(colors);
      totalPoints += positions.length / 3;
      loadedNodes += nNodes || 0;

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      if (colors) geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      const mat = this._createPointMaterial(!!colors);
      const mesh = new THREE.Points(geo, mat);
      this.scene.add(mesh);
      tempMeshes.push(mesh);
    };

    const finalize = () => {
      for (const m of tempMeshes) {
        m.geometry.dispose();
        this.scene.remove(m);
      }
      tempMeshes = [];

      const mergedPos = new Float32Array(totalPoints * 3);
      const mergedCol = allColors.length > 0 ? new Float32Array(totalPoints * 3) : null;
      let off = 0;
      for (let i = 0; i < allPositions.length; i++) {
        mergedPos.set(allPositions[i], off);
        if (mergedCol) mergedCol.set(allColors[i], off);
        off += allPositions[i].length;
      }
      allPositions = null;
      allColors = null;

      this._replacePoints(mergedPos, mergedCol);
    };

    const progress = (phase, maxDepthTotalNodes) => {
      if (onProgress) {
        onProgress({
          loadedPoints: totalPoints,
          loadedNodes: loadedNodes,
          totalNodes: maxDepthTotalNodes,
          phase: phase,
        });
      }
    };

    // Load levels 0 through max_depth-1
    for (let lvl = 0; lvl < meta.max_depth; lvl++) {
      try {
        const resp = await fetch(`${levelUrlBase}/${lvl}`);
        const buffer = await resp.arrayBuffer();
        if (buffer.byteLength > 4) {
          const { positions, colors, numNodes } = this.parseLevelResponse(buffer, meta.bytes_per_point, meta.schema);
          if (positions.length > 0) {
            addChunk(positions, colors, numNodes);
            progress('lod', 0);
          }
        }
      } catch (e) { /* level might be empty */ }
    }

    lodPoints = totalPoints;

    // Load max_depth in chunks
    let offset = 0;
    let maxDepthTotalNodes = 0;
    while (true) {
      const resp = await fetch(`${levelUrlBase}/${meta.max_depth}?offset=${offset}&limit=${chunkSize}`);
      maxDepthTotalNodes = parseInt(resp.headers.get('X-Total-Nodes') || '0');
      const buffer = await resp.arrayBuffer();

      if (buffer.byteLength <= 4) break;

      const { positions, colors, numNodes } = this.parseLevelResponse(buffer, meta.bytes_per_point, meta.schema);

      if (positions.length > 0) {
        addChunk(positions, colors, numNodes);
        progress('detail', maxDepthTotalNodes);
      }

      offset += chunkSize;
      if (offset >= maxDepthTotalNodes) break;
    }

    // Merge all chunks into single geometry
    finalize();

    // Store the offset: indices from the pipeline refer to max_depth points,
    // which start after lodPoints in the viewer's merged array
    this.lodOffset = lodPoints;

    return meta;
  }

  /** Replace the point cloud mesh with new data (used during progressive loading) */
  _replacePoints(positions, colors) {
    // Store raw arrays for client-side queries
    this.positions = positions;
    this.colors = colors;

    // Remove old point cloud
    const old = this.scene.children.find(c => c.isPoints);
    if (old) {
      old.geometry.dispose();
      this.scene.remove(old);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    if (colors) geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const mat = this._createPointMaterial(!!colors);
    this.scene.add(new THREE.Points(geo, mat));
  }

  /** Add a point cloud from raw arrays (standalone, for external use) */
  addPoints(positions, colors) {
    this.positions = positions;
    this.colors = colors;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    if (colors) geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const mat = this._createPointMaterial(!!colors);
    this.scene.add(new THREE.Points(geo, mat));
  }

  // ── Render loop ──

  start() {
    if (this.running) return;
    this.running = true;
    this._animate();
  }

  stop() {
    this.running = false;
  }

  _animate() {
    if (!this.running) return;
    requestAnimationFrame(() => this._animate());
    this.controls.update();

    const rect = this.container.getBoundingClientRect();
    const w = Math.floor(rect.width * devicePixelRatio);
    const h = Math.floor(rect.height * devicePixelRatio);

    if (this.renderer.domElement.width !== w || this.renderer.domElement.height !== h) {
      this.renderer.setSize(rect.width, rect.height, false);
      this.camera.aspect = rect.width / rect.height;
      this.camera.updateProjectionMatrix();
      this._rtEDL.setSize(w, h);
    }

    // Pass 1: scene to framebuffer
    this.renderer.setRenderTarget(this._rtEDL);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear(true, true, true);
    this.renderer.render(this.scene, this.camera);

    // Pass 2: EDL quad to screen
    this._edlMaterial.uniforms.screenWidth.value = w;
    this._edlMaterial.uniforms.screenHeight.value = h;
    this._edlMaterial.uniforms.uEDLColor.value = this._rtEDL.texture;

    this.renderer.setRenderTarget(null);
    this.renderer.setClearColor(this.options.bgColor, 1);
    this.renderer.clear(true, true, true);
    this.renderer.render(this._quadScene, this._quadCamera);

    // Pass 3: restore point cloud depth buffer (depth-only, no color)
    this._depthOnlyMat.uniforms.pointSize.value = this.options.pointSize;
    this.scene.overrideMaterial = this._depthOnlyMat;
    this.renderer.render(this.scene, this.camera);
    this.scene.overrideMaterial = null;

    // Pass 4: overlays with proper depth testing against point cloud
    this.renderer.render(this.overlayScene, this.camera);

    // 2D slice view (separate renderer, ortho camera, layer 1 + 2 only)
    if (this._sliceRenderer && this.orthoCamera) {
      const sr = this._sliceRenderer;
      const sc = this._sliceContainer;
      const srect = sc.getBoundingClientRect();
      const sw = Math.floor(srect.width * devicePixelRatio);
      const sh = Math.floor(srect.height * devicePixelRatio);

      if (sr.domElement.width !== sw || sr.domElement.height !== sh) {
        sr.setSize(srect.width, srect.height, false);
      }

      sr.setClearColor(this.options.bgColor, 1);
      sr.clear(true, true, true);
      // Render both the main scene (slice points) and overlay scene (analysis overlays)
      sr.render(this.scene, this.orthoCamera);
      sr.render(this.overlayScene, this.orthoCamera);
    }
  }
}