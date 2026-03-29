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

    this.scene = new THREE.Scene();
    this.camera = null;
    this.controls = null;
    this.renderer = null;
    this.pointsMat = null;
    this.metadata = null;
    this.running = false;

    // EDL internals
    this._edlMaterial = null;
    this._rtEDL = null;
    this._quadScene = null;
    this._quadCamera = null;
    this._neighbourCount = 8;

    this._initRenderer();
    this._initCamera();
    this._initEDL();
    if (this.options.showAxes) this._initAxes();
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

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
  }

  _initAxes() {
    const len = this.options.axisLength;
    const addAxis = (dx, dy, dz, color) => {
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(dx * len, dy * len, dz * len)
      ]);
      this.scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({
        color, opacity: 0.35, transparent: true
      })));
    };
    addAxis(1, 0, 0, 0xe8a44a);
    addAxis(0, 1, 0, 0x4a9de8);
    addAxis(0, 0, 1, 0x50c878);
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

  /** Set point size in pixels */
  setPointSize(size) {
    this.options.pointSize = size;
    if (this.pointsMat) this.pointsMat.uniforms.pointSize.value = size;
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

  /** Add arbitrary Three.js object to the scene (polylines, meshes, etc.) */
  addOverlay(object3d) {
    this.scene.add(object3d);
  }

  /** Remove an overlay object */
  removeOverlay(object3d) {
    this.scene.remove(object3d);
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

  /** Load point cloud from PCO server endpoints */
  async loadFromServer(metadataUrl = '/metadata', levelUrlBase = '/level') {
    const meta = await (await fetch(metadataUrl)).json();
    this.metadata = meta;

    // Auto-frame camera
    const rm = meta.root_min;
    const rs = meta.root_size;
    const center = new THREE.Vector3(rm[0] + rs / 2, rm[1] + rs / 2, rm[2] + rs / 2);
    this.lookAt(center, rs);

    // Fetch all points at max depth
    const resp = await fetch(`${levelUrlBase}/${meta.max_depth}`);
    const buffer = await resp.arrayBuffer();
    const { positions, colors, totalPoints } = this.parseLevelResponse(buffer, meta.bytes_per_point, meta.schema);

    this.addPoints(positions, colors);

    return meta;
  }

  /** Add a point cloud from raw arrays (can be called multiple times) */
  addPoints(positions, colors) {
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
  }
}
