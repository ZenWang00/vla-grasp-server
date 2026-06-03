"""Inline HTML for the grasp server web UI (no build step required)."""

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Grasp Pose Server</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; padding: 24px; }
  h1 { font-size: 1.4rem; margin-bottom: 20px; color: #fff; letter-spacing: 0.5px; }
  .layout { display: flex; gap: 24px; flex-wrap: wrap; }
  .panel { background: #1e1e1e; border-radius: 10px; padding: 20px; }
  .image-panel { flex: 1 1 480px; }
  .form-panel { flex: 0 0 320px; display: flex; flex-direction: column; gap: 12px; }

  .image-wrapper { position: relative; }
  #capture-img {
    width: 100%; border-radius: 6px; display: block;
    background: #2a2a2a; min-height: 200px; object-fit: contain;
    transition: outline 0.1s;
  }
  #capture-img.flash {
    outline: 3px solid #4caf50;
  }
  #capture-badge {
    display: none; position: absolute; top: 10px; left: 10px;
    background: #4caf50; color: #fff; font-size: 0.75rem; font-weight: bold;
    padding: 3px 8px; border-radius: 4px; letter-spacing: 0.5px;
  }
  #capture-status { margin-top: 8px; font-size: 0.8rem; color: #888; }

  label { font-size: 0.85rem; color: #aaa; display: block; margin-bottom: 4px; }
  input[type=text], input[type=number] {
    width: 100%; padding: 8px 10px; background: #2a2a2a; border: 1px solid #444;
    border-radius: 6px; color: #eee; font-size: 0.9rem;
  }
  input:focus { outline: none; border-color: #5a9fd4; }
  .field { display: flex; flex-direction: column; }
  .row { display: flex; gap: 10px; }
  .row .field { flex: 1; }
  .section-label {
    font-size: 0.72rem; color: #666; text-transform: uppercase;
    letter-spacing: 0.6px; margin-top: 14px; margin-bottom: 4px;
  }

  button#run-btn {
    padding: 10px; background: #2a7ae2; border: none; border-radius: 6px;
    color: #fff; font-size: 0.95rem; cursor: pointer; margin-top: 4px;
  }
  button#run-btn:hover { background: #1f66c7; }
  button#run-btn:disabled { background: #444; cursor: not-allowed; }

  button#align-btn {
    padding: 10px; background: #b5803f; border: none; border-radius: 6px;
    color: #fff; font-size: 0.95rem; cursor: pointer; margin-top: 8px;
  }
  button#align-btn:hover { background: #9a6a30; }
  button#align-btn:disabled { background: #444; cursor: not-allowed; }

  button#geo-btn {
    padding: 10px; background: #2d7a5a; border: none; border-radius: 6px;
    color: #fff; font-size: 0.95rem; cursor: pointer; margin-top: 8px;
  }
  button#geo-btn:hover { background: #236147; }
  button#geo-btn:disabled { background: #444; cursor: not-allowed; }

  button#send-btn {
    padding: 10px; background: #2e7d32; border: none; border-radius: 6px;
    color: #fff; font-size: 0.95rem; cursor: pointer; margin-top: 8px; display: none;
  }
  button#send-btn:hover { background: #1b5e20; }
  button#send-btn:disabled { background: #444; cursor: not-allowed; }
  #send-status { font-size: 0.82rem; margin-top: 4px; }

  button#ik-btn {
    padding: 10px; background: #b08020; border: none; border-radius: 6px;
    color: #fff; font-size: 0.95rem; cursor: pointer; margin-top: 4px; display: none;
  }
  button#ik-btn:hover { background: #8a6318; }
  button#ik-btn:disabled { background: #444; cursor: not-allowed; }
  #ik-status { font-size: 0.82rem; margin-top: 4px; }

  button#view3d-btn {
    padding: 10px; background: #6a3fb5; border: none; border-radius: 6px;
    color: #fff; font-size: 0.95rem; cursor: pointer; margin-top: 4px; display: none;
  }
  button#view3d-btn:hover { background: #5430a0; }

  #result-panel { margin-top: 20px; display: none; }
  #result-panel h2 { font-size: 1rem; margin-bottom: 12px; color: #ccc; }
  .grasp-card {
    background: #2a2a2a; border-radius: 8px; padding: 12px; margin-bottom: 10px;
    font-size: 0.82rem; line-height: 1.7;
  }
  .grasp-card .rank { font-weight: bold; color: #5a9fd4; margin-bottom: 4px; }
  .badge-ok { color: #4caf50; font-weight: bold; }
  .badge-err { color: #f44336; font-weight: bold; }
  #status-msg { font-size: 0.85rem; margin-top: 6px; }
  .spinner {
    display: inline-block; width: 14px; height: 14px; border: 2px solid #555;
    border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite;
    margin-right: 6px; vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  details { margin-top: 8px; }
  summary { cursor: pointer; color: #888; font-size: 0.8rem; }
  pre { background: #111; padding: 8px; border-radius: 4px; overflow-x: auto; font-size: 0.75rem; color: #aaa; }
</style>
</head>
<body>
<h1>Grasp Pose Server</h1>
<div class="layout">

  <!-- Left: live camera stream -->
  <div class="panel image-panel">
    <div class="image-wrapper">
      <img id="capture-img" alt="No capture yet — click Run Grasp to capture">
      <div id="capture-badge">CAPTURED</div>
    </div>
    <div id="capture-status">Click Run Grasp to capture a frame</div>
  </div>

  <!-- Right: form -->
  <div class="panel form-panel">
    <div class="field">
      <label for="task_spec">Task spec <span style="color:#f44">*</span></label>
      <input type="text" id="task_spec" placeholder='e.g. "grasp the red cup"' required>
    </div>
    <details>
      <summary>Advanced (VLM overrides)</summary>
      <div style="margin-top:10px; display:flex; flex-direction:column; gap:10px;">
        <div class="field">
          <label for="provider">Provider</label>
          <input type="text" id="provider" placeholder="e.g. gemini">
        </div>
        <div class="field">
          <label for="model">Model</label>
          <input type="text" id="model" placeholder="e.g. gemini-robotics-er-1.6-preview">
        </div>
      </div>
    </details>

    <div class="section-label">Run Grasp (VLM + SAM2 + CGN)</div>
    <div class="row">
      <div class="field">
        <label for="grasp_num_candidates">VLM regions</label>
        <input type="number" id="grasp_num_candidates" value="1" min="1" max="10">
      </div>
      <div class="field">
        <label for="top_k">Top K grasps</label>
        <input type="number" id="top_k" value="1" min="1" max="10">
      </div>
    </div>
    <button id="run-btn" onclick="runGrasp()">Run Grasp</button>

    <div class="section-label">Run Align (VLM 2D point)</div>
    <div class="row">
      <div class="field">
        <label for="align_num_candidates">Align candidates</label>
        <input type="number" id="align_num_candidates" value="1" min="1" max="10">
      </div>
    </div>
    <button id="align-btn" onclick="runAlign()">Run Align (2D point)</button>

    <div class="section-label">Run Geometry (PCA, no AI)</div>
    <div class="row">
      <div class="field">
        <label for="geo_top_k">Candidates</label>
        <input type="number" id="geo_top_k" value="3" min="1" max="20">
      </div>
    </div>
    <button id="geo-btn" onclick="runGeometry()">Run Geometry (PCA)</button>
    <div id="status-msg"></div>
    <button id="send-btn" onclick="sendToRobot()">&#9654; Send to Robot</button>
    <div id="send-status"></div>
    <button id="ik-btn" onclick="triggerIkCheck()">&#10003; IK Check + Execute</button>
    <div id="ik-status"></div>
    <button id="view3d-btn" onclick="window.open('/grasp_viz_3d','_blank')">&#9706; View 3D Point Cloud</button>

    <div id="result-panel">
      <h2>Results</h2>
      <div id="grasp-cards"></div>
    </div>
  </div>

</div>

<script>
// ── Capture image display ─────────────────────────────────────────────────
function refreshCaptureImage() {
  const img = document.getElementById('capture-img');
  img.src = '/latest_image?t=' + Date.now();
}

// ── Capture flash (2 Hz) ──────────────────────────────────────────────────
let lastCaptureTs = null;
function pollCaptureStatus() {
  fetch('/capture_status')
    .then(r => r.json())
    .then(data => {
      const ts = data.uploaded_at;
      if (ts && ts !== lastCaptureTs) {
        lastCaptureTs = ts;
        refreshCaptureImage();
        const img = document.getElementById('capture-img');
        const badge = document.getElementById('capture-badge');
        img.classList.add('flash');
        badge.style.display = 'block';
        const d = new Date(ts * 1000);
        document.getElementById('capture-status').textContent =
          'Captured at ' + d.toLocaleTimeString() + ' — ready for Run Grasp';
        setTimeout(() => {
          img.classList.remove('flash');
          badge.style.display = 'none';
        }, 2000);
      }
    })
    .catch(() => {});
}
setInterval(pollCaptureStatus, 500);

// ── Capture step (shared by Run Grasp and Run Align) ──────────────────────
async function captureFrame(statusEl) {
  statusEl.innerHTML = '<span class="spinner"></span>Capturing...';
  const capResp = await fetch('/request_capture', { method: 'POST' });
  if (!capResp.ok) throw new Error('request_capture failed');

  // Poll /capture_status until uploaded_at changes (max 5 s).
  const prevTs = lastCaptureTs;
  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 200));
    const s = await fetch('/capture_status').then(r => r.json()).catch(() => ({}));
    if (s.uploaded_at && s.uploaded_at !== prevTs) return true;
  }
  return false;
}

// ── Run grasp ─────────────────────────────────────────────────────────────
async function runGrasp() {
  const taskSpec = document.getElementById('task_spec').value.trim();
  if (!taskSpec) {
    document.getElementById('status-msg').innerHTML =
      '<span class="badge-err">task_spec is required.</span>';
    return;
  }

  const btn = document.getElementById('run-btn');
  const statusEl = document.getElementById('status-msg');
  btn.disabled = true;

  // Step 1: request capture and wait for ROS2 node to upload it.
  try {
    if (!await captureFrame(statusEl)) {
      statusEl.innerHTML = '<span class="badge-err">Capture timed out — is the ROS2 node running?</span>';
      btn.disabled = false;
      return;
    }
  } catch (e) {
    statusEl.innerHTML = '<span class="badge-err">Capture error:</span> ' + escapeHtml(String(e));
    btn.disabled = false;
    return;
  }

  // Step 2: run the grasp pipeline.
  statusEl.innerHTML = '<span class="spinner"></span>Running pipeline...';
  const fd = new FormData();
  fd.append('task_spec', taskSpec);
  fd.append('num_candidates', document.getElementById('grasp_num_candidates').value);
  fd.append('top_k', document.getElementById('top_k').value);
  const provider = document.getElementById('provider').value.trim();
  const model = document.getElementById('model').value.trim();
  if (provider) fd.append('provider', provider);
  if (model) fd.append('model', model);

  try {
    const resp = await fetch('/run_grasp', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) {
      const msg = data.detail || JSON.stringify(data);
      statusEl.innerHTML = '<span class="badge-err">Error ' + resp.status + ':</span> ' +
        escapeHtml(typeof msg === 'string' ? msg : JSON.stringify(msg));
    } else {
      renderResult(data);
      statusEl.innerHTML = '<span class="badge-ok">Done</span> — ' +
        data.elapsed_ms + ' ms &nbsp;|&nbsp; run_id: ' + escapeHtml(data.run_id);
    }
  } catch (e) {
    statusEl.innerHTML = '<span class="badge-err">Network error:</span> ' + escapeHtml(String(e));
  } finally {
    btn.disabled = false;
  }
}

// ── Run align (2D point + depth -> 6-DoF) ─────────────────────────────────
async function runAlign() {
  const taskSpec = document.getElementById('task_spec').value.trim();
  if (!taskSpec) {
    document.getElementById('status-msg').innerHTML =
      '<span class="badge-err">task_spec is required.</span>';
    return;
  }

  const btn = document.getElementById('align-btn');
  const statusEl = document.getElementById('status-msg');
  btn.disabled = true;

  // Step 1: request capture (same as Run Grasp).
  try {
    if (!await captureFrame(statusEl)) {
      statusEl.innerHTML = '<span class="badge-err">Capture timed out — is the ROS2 node running?</span>';
      btn.disabled = false;
      return;
    }
  } catch (e) {
    statusEl.innerHTML = '<span class="badge-err">Capture error:</span> ' + escapeHtml(String(e));
    btn.disabled = false;
    return;
  }

  // Step 2: run the lightweight align flow.
  statusEl.innerHTML = '<span class="spinner"></span>Running align...';
  const fd = new FormData();
  fd.append('task_spec', taskSpec);
  fd.append('num_candidates', document.getElementById('align_num_candidates').value);
  const provider = document.getElementById('provider').value.trim();
  const model = document.getElementById('model').value.trim();
  if (provider) fd.append('provider', provider);
  if (model) fd.append('model', model);

  try {
    const resp = await fetch('/run_align', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) {
      const msg = data.detail || JSON.stringify(data);
      statusEl.innerHTML = '<span class="badge-err">Error ' + resp.status + ':</span> ' +
        escapeHtml(typeof msg === 'string' ? msg : JSON.stringify(msg));
    } else {
      renderResult(data);
      statusEl.innerHTML = '<span class="badge-ok">Done (align)</span> — ' +
        data.elapsed_ms + ' ms &nbsp;|&nbsp; run_id: ' + escapeHtml(data.run_id);
    }
  } catch (e) {
    statusEl.innerHTML = '<span class="badge-err">Network error:</span> ' + escapeHtml(String(e));
  } finally {
    btn.disabled = false;
  }
}

// ── Run geometry (PCA-based, no AI) ──────────────────────────────────────
async function runGeometry() {
  const btn = document.getElementById('geo-btn');
  const statusEl = document.getElementById('status-msg');
  btn.disabled = true;

  // Step 1: request capture (same as other run modes).
  try {
    if (!await captureFrame(statusEl)) {
      statusEl.innerHTML = '<span class="badge-err">Capture timed out — is the ROS2 node running?</span>';
      btn.disabled = false;
      return;
    }
  } catch (e) {
    statusEl.innerHTML = '<span class="badge-err">Capture error:</span> ' + escapeHtml(String(e));
    btn.disabled = false;
    return;
  }

  // Step 2: run the geometry pipeline.
  statusEl.innerHTML = '<span class="spinner"></span>Running geometry PCA...';
  const fd = new FormData();
  fd.append('top_k', document.getElementById('geo_top_k').value);

  try {
    const resp = await fetch('/run_geometry', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) {
      const msg = data.detail || JSON.stringify(data);
      statusEl.innerHTML = '<span class="badge-err">Error ' + resp.status + ':</span> ' +
        escapeHtml(typeof msg === 'string' ? msg : JSON.stringify(msg));
    } else {
      renderResult(data);
      statusEl.innerHTML = '<span class="badge-ok">Done (geometry)</span> — ' +
        data.elapsed_ms + ' ms &nbsp;|&nbsp; run_id: ' + escapeHtml(data.run_id);
    }
  } catch (e) {
    statusEl.innerHTML = '<span class="badge-err">Network error:</span> ' + escapeHtml(String(e));
  } finally {
    btn.disabled = false;
  }
}

// ── Send to Robot ─────────────────────────────────────────────────────────
async function sendToRobot() {
  const btn = document.getElementById('send-btn');
  const statusEl = document.getElementById('send-status');
  btn.disabled = true;
  statusEl.innerHTML = '<span class="spinner"></span>Sending...';
  try {
    const resp = await fetch('/trigger_publish', { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) {
      const msg = data.detail || JSON.stringify(data);
      statusEl.innerHTML = '<span class="badge-err">Error:</span> ' + escapeHtml(String(msg));
    } else {
      statusEl.innerHTML = '<span class="badge-ok">Sent</span> — run_id: ' + escapeHtml(data.run_id);
    }
  } catch (e) {
    statusEl.innerHTML = '<span class="badge-err">Network error:</span> ' + escapeHtml(String(e));
  } finally {
    btn.disabled = false;
  }
}

function renderResult(data) {
  const panel = document.getElementById('result-panel');
  const cards = document.getElementById('grasp-cards');
  panel.style.display = 'block';
  document.getElementById('send-btn').style.display = 'block';
  document.getElementById('send-status').innerHTML = '';
  document.getElementById('ik-btn').style.display = 'block';
  document.getElementById('ik-status').innerHTML = '';
  document.getElementById('view3d-btn').style.display = 'block';
  cards.innerHTML = '';

  // Update the left panel image to show the grasp visualization.
  if (data.grasp_viz) {
    const img = document.getElementById('capture-img');
    img.src = '/grasp_viz_image?t=' + Date.now();
    document.getElementById('capture-status').textContent =
      'Grasp visualization — run_id: ' + (data.run_id || '');
  }

  const grasps = data.grasps || [];
  if (!grasps.length) {
    cards.innerHTML = '<p style="color:#888">No grasps returned.</p>';
    return;
  }
  grasps.forEach((g, i) => {
    const pos = (g.position_xyz || []).map(v => v.toFixed(4)).join(', ');
    const q = (g.quaternion_xyzw || []).map(v => v.toFixed(4)).join(', ');
    const score = g.score != null ? g.score.toFixed(4) : 'n/a';
    const width = g.width_m != null ? (g.width_m * 1000).toFixed(1) + ' mm' : 'n/a';
    const div = document.createElement('div');
    div.className = 'grasp-card';
    div.innerHTML =
      '<div class="rank">#' + (i + 1) + '</div>' +
      '<b>Score:</b> ' + score + '<br>' +
      '<b>Width:</b> ' + width + '<br>' +
      '<b>Position (xyz):</b> [' + pos + ']<br>' +
      '<b>Quaternion (xyzw):</b> [' + q + ']<br>' +
      '<details><summary>Raw JSON</summary><pre>' +
        escapeHtml(JSON.stringify(g, null, 2)) + '</pre></details>';
    cards.appendChild(div);
  });
}

// ── IK Check + Execute ────────────────────────────────────────────────────
async function triggerIkCheck() {
  const btn = document.getElementById('ik-btn');
  const statusEl = document.getElementById('ik-status');
  btn.disabled = true;

  // Step 1: trigger IK check on the server
  statusEl.innerHTML = '<span class="spinner"></span>Triggering IK check...';
  let traceId;
  try {
    const r = await fetch('/trigger_ik_check', { method: 'POST' });
    const d = await r.json();
    if (!r.ok) {
      statusEl.innerHTML = '<span class="badge-err">Error:</span> ' + escapeHtml(String(d.detail));
      btn.disabled = false; return;
    }
    traceId = d.trace_id;
  } catch (e) {
    statusEl.innerHTML = '<span class="badge-err">Network error</span>';
    btn.disabled = false; return;
  }

  // Step 2: poll until ROS2 client submits IK results (max 10 s)
  statusEl.innerHTML = '<span class="spinner"></span>Waiting for IK result (trace: ' + escapeHtml(traceId) + ')...';
  const deadline = Date.now() + 10000;
  let ready = false;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 500));
    try {
      const s = await fetch('/ik_result_status?trace_id=' + encodeURIComponent(traceId)).then(r => r.json());
      if (s.ready) { ready = true; break; }
    } catch (_) {}
  }
  if (!ready) {
    statusEl.innerHTML = '<span class="badge-err">Timeout waiting for IK result</span>';
    btn.disabled = false; return;
  }

  // Step 3: run server-side selection and queue for execution
  try {
    const r = await fetch('/select_and_execute', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({trace_id: traceId})
    });
    const d = await r.json();
    if (!r.ok) {
      statusEl.innerHTML = '<span class="badge-err">Select error:</span> ' + escapeHtml(String(d.detail));
    } else {
      statusEl.innerHTML = '<span class="badge-ok">IK done, executing</span> — trace_id: ' + escapeHtml(traceId);
    }
  } catch (e) {
    statusEl.innerHTML = '<span class="badge-err">Network error on select</span>';
  } finally {
    btn.disabled = false;
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
</script>
</body>
</html>
"""
