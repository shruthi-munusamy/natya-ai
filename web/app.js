// Natya — real-time mudra recognition in the browser.
// Pipeline: MediaPipe Hands (JS, on-device) -> normalize landmarks ->
// ONNX classifier (asamyukta if one hand, samyukta if two) -> render.

import { HandLandmarker, PoseLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm";
import * as ort from
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/+esm";

// Point onnxruntime-web at its WASM binaries on the CDN. The ESM build
// doesn't auto-locate them, so we set this explicitly. numThreads=1 avoids
// requiring COOP/COEP headers, which a static file server doesn't send.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
ort.env.wasm.numThreads = 1;

// Single place to flip between local dev and the deployed backend — see
// DEPLOY.md. Swap this one line after deploying the FastAPI server.
const BACKEND_URL = "http://localhost:8001";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const predEl = document.getElementById("prediction");
const confEl = document.getElementById("confidence");
const statusEl = document.getElementById("status");
const interpretEl = document.getElementById("interpretation");

const HAND_LANDMARKS = 21;
const COORDS = 3;
const MIDDLE_MCP = 9;

let handLandmarker;
let poseLandmarker;
let asamyuktaSession;
let samyuktaSession;
let classes;

// --- Landmark normalization (mirrors src/data/preprocess.py) ---

function normalizeSingleHand(landmarks) {
  // landmarks: 21 objects with x, y, z. Returns Float32Array(63).
  const out = new Float32Array(HAND_LANDMARKS * COORDS);
  const wx = landmarks[0].x, wy = landmarks[0].y, wz = landmarks[0].z;
  const mx = landmarks[MIDDLE_MCP].x - wx;
  const my = landmarks[MIDDLE_MCP].y - wy;
  const mz = landmarks[MIDDLE_MCP].z - wz;
  let scale = Math.hypot(mx, my, mz);
  if (scale < 1e-6) scale = 1;
  for (let i = 0; i < HAND_LANDMARKS; i++) {
    out[i * 3 + 0] = (landmarks[i].x - wx) / scale;
    out[i * 3 + 1] = (landmarks[i].y - wy) / scale;
    out[i * 3 + 2] = (landmarks[i].z - wz) / scale;
  }
  return out;
}

function normalizeTwoHands(left, right) {
  // shared coordinate space: centroid of wrists, average wrist-to-MCP scale.
  const out = new Float32Array(2 * HAND_LANDMARKS * COORDS);
  const cx = (left[0].x + right[0].x) / 2;
  const cy = (left[0].y + right[0].y) / 2;
  const cz = (left[0].z + right[0].z) / 2;
  const sL = Math.hypot(
    left[MIDDLE_MCP].x - left[0].x,
    left[MIDDLE_MCP].y - left[0].y,
    left[MIDDLE_MCP].z - left[0].z
  );
  const sR = Math.hypot(
    right[MIDDLE_MCP].x - right[0].x,
    right[MIDDLE_MCP].y - right[0].y,
    right[MIDDLE_MCP].z - right[0].z
  );
  let scale = (sL + sR) / 2;
  if (scale < 1e-6) scale = 1;
  for (let i = 0; i < HAND_LANDMARKS; i++) {
    out[i * 3 + 0] = (left[i].x - cx) / scale;
    out[i * 3 + 1] = (left[i].y - cy) / scale;
    out[i * 3 + 2] = (left[i].z - cz) / scale;
  }
  const off = HAND_LANDMARKS * COORDS;
  for (let i = 0; i < HAND_LANDMARKS; i++) {
    out[off + i * 3 + 0] = (right[i].x - cx) / scale;
    out[off + i * 3 + 1] = (right[i].y - cy) / scale;
    out[off + i * 3 + 2] = (right[i].z - cz) / scale;
  }
  return out;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

// --- Temporal smoothing ---
// Average softmax distributions across a sliding window of recent frames.
// At ~30fps, a 10-frame window is roughly 333ms of context. The display still
// updates every frame; only the *decision* about which mudra to show is
// averaged. Reset the buffer when hand mode changes (asamyukta and samyukta
// have different label spaces) or when no hand is detected (avoid carrying
// stale state across pauses).
const SMOOTHING_WINDOW = 10;
let probBuffer = [];
let bufferMode = null;

function smoothProbs(probs, mode) {
  if (mode !== bufferMode) {
    probBuffer = [];
    bufferMode = mode;
  }
  probBuffer.push(probs);
  if (probBuffer.length > SMOOTHING_WINDOW) probBuffer.shift();
  const avg = new Array(probs.length).fill(0);
  for (const p of probBuffer) {
    for (let i = 0; i < p.length; i++) avg[i] += p[i];
  }
  for (let i = 0; i < avg.length; i++) avg[i] /= probBuffer.length;
  return avg;
}

function resetBuffer() {
  probBuffer = [];
  bufferMode = null;
}

// --- Skeleton overlay ---

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

function drawHand(landmarks) {
  const w = canvas.width, h = canvas.height;
  ctx.strokeStyle = "rgba(212, 165, 116, 0.85)";
  ctx.lineWidth = 2;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(245, 245, 245, 0.95)";
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 3, 0, 2 * Math.PI);
    ctx.fill();
  }
}

// --- Pose context extraction ---
// Cheap, threshold-based rules over MediaPipe Pose landmarks (normalized
// image coordinates, y=0 at top). These give the LLM body-position context
// beyond the raw mudra name — "Pathaka near the chest" and "Pathaka raised
// above the head" retrieve different viniyogas. Thresholds are hand-tuned
// against a webcam framing where the torso fills most of the vertical
// frame; they're heuristics, not biomechanical constants.
const POSE = {
  NOSE: 0, L_SHOULDER: 11, R_SHOULDER: 12,
  L_WRIST: 15, R_WRIST: 16, L_HIP: 23, R_HIP: 24,
};
const RAISED_MARGIN = 0.05;
const FACE_BAND = 0.08;
const CHEST_TOP_MARGIN = 0.03;
const FORWARD_Z_MARGIN = 0.15;

function wristContext(poseLm, wristIdx, shoulderIdx, hipIdx) {
  const wrist = poseLm[wristIdx];
  const shoulder = poseLm[shoulderIdx];
  const hip = poseLm[hipIdx];
  const nose = poseLm[POSE.NOSE];
  const feats = [];

  if (wrist.y < nose.y - RAISED_MARGIN) {
    feats.push("arm raised above head");
  } else if (Math.abs(wrist.y - nose.y) < FACE_BAND) {
    feats.push("hand near face");
  } else if (wrist.y > shoulder.y - CHEST_TOP_MARGIN && wrist.y < hip.y) {
    feats.push("hand near chest");
  }
  if (shoulder.z - wrist.z > FORWARD_Z_MARGIN) {
    feats.push("arm extended forward");
  }
  return feats;
}

// Associates each detected hand with the nearer pose wrist (left/right) by
// on-screen distance, then derives context features for it. If pose isn't
// detected (person out of frame, poor lighting), returns null and the
// interpretation request simply omits context rather than failing.
function computePoseContext(poseLandmarksList, hands) {
  if (!poseLandmarksList || poseLandmarksList.length === 0 || hands.length === 0) {
    return null;
  }
  const poseLm = poseLandmarksList[0];
  const feats = new Set();
  for (const hand of hands) {
    const hx = hand[0].x, hy = hand[0].y;
    const dL = Math.hypot(poseLm[POSE.L_WRIST].x - hx, poseLm[POSE.L_WRIST].y - hy);
    const dR = Math.hypot(poseLm[POSE.R_WRIST].x - hx, poseLm[POSE.R_WRIST].y - hy);
    const useLeft = dL <= dR;
    for (const f of wristContext(
      poseLm,
      useLeft ? POSE.L_WRIST : POSE.R_WRIST,
      useLeft ? POSE.L_SHOULDER : POSE.R_SHOULDER,
      useLeft ? POSE.L_HIP : POSE.R_HIP,
    )) feats.add(f);
  }
  return feats.size ? Array.from(feats).join(", ") : null;
}

// --- LLM interpretation ---
// Only fires once a prediction has been stable for STABLE_MS at reasonable
// confidence — mudra transitions are noisy frame-to-frame, and we don't want
// to spam the LLM (or the UI) on every intermediate hand shape.
const STABLE_MS = 1000;
const CONFIDENCE_THRESHOLD = 0.6;

let stableLabel = null;
let stableSince = 0;
let lastInterpretedKey = null;
let currentAbortController = null;
let activeMudra = null; // mudra the panel currently reflects (loading or streamed text)

function setInterpretationPlaceholder() {
  interpretEl.innerHTML = '<span class="placeholder">Hold a mudra steady to see its interpretation</span>';
}

function showInterpretationLoading() {
  interpretEl.innerHTML =
    '<span class="loading-dots"><span></span><span></span><span></span></span>';
}

function resetInterpretation() {
  if (currentAbortController) currentAbortController.abort();
  stableLabel = null;
  lastInterpretedKey = null;
  activeMudra = null;
  setInterpretationPlaceholder();
}

// Clears/aborts whatever's currently shown without touching stability
// tracking — used when the top-1 mudra changes but hasn't been stable long
// enough yet to start a new interpretation. Keeps stale text from a
// previous mudra on screen while a new one is still being recognized.
function clearActiveInterpretation() {
  if (currentAbortController) currentAbortController.abort();
  activeMudra = null;
  lastInterpretedKey = null;
  setInterpretationPlaceholder();
}

// Parses the SSE stream from /interpret/stream. sse-starlette writes
// "event: <type>\r\ndata: <payload>\r\n\r\n" blocks; per the SSE spec, only
// a single leading space after "data:" is a delimiter, so we strip exactly
// one and preserve the rest (tokens frequently start with a meaningful
// leading space, e.g. " the").
async function maybeInterpret(mudras, context) {
  const key = `${mudras.join(",")}|${context || ""}`;
  if (key === lastInterpretedKey) return;
  lastInterpretedKey = key;
  activeMudra = mudras[0];

  if (currentAbortController) currentAbortController.abort();
  const controller = new AbortController();
  currentAbortController = controller;

  showInterpretationLoading();
  let first = true;
  try {
    const res = await fetch(`${BACKEND_URL}/interpret/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mudras, context: context || null }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      const blocks = buf.split("\n\n");
      buf = blocks.pop();
      for (const block of blocks) {
        let eventType = "message";
        let data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) eventType = line.slice(6).trim();
          else if (line.startsWith("data:")) {
            let val = line.slice(5);
            if (val.startsWith(" ")) val = val.slice(1);
            data += val;
          }
        }
        if (eventType === "token" && data) {
          if (first) {
            interpretEl.textContent = "";
            first = false;
          }
          interpretEl.textContent += data;
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      console.error("Interpretation request failed:", err);
      interpretEl.textContent = "Interpretation unavailable — is the server running?";
    }
  }
}

function updateInterpretationState(pred, context) {
  if (!pred) {
    if (stableLabel !== null || activeMudra !== null) resetInterpretation();
    return;
  }
  if (pred.name !== stableLabel) {
    stableLabel = pred.name;
    stableSince = performance.now();
  }
  // The displayed/streaming interpretation belongs to a mudra we're no
  // longer predicting — clear it right away instead of leaving stale text
  // up until the new mudra earns its own interpretation a second later.
  if (activeMudra !== null && activeMudra !== pred.name) {
    clearActiveInterpretation();
  }
  const stableFor = performance.now() - stableSince;
  if (stableFor >= STABLE_MS && pred.conf > CONFIDENCE_THRESHOLD) {
    maybeInterpret([pred.name], context);
  }
}

// --- Inference loop ---

async function predict(handsList) {
  if (handsList.length === 0) {
    resetBuffer();
    return null;
  }
  if (handsList.length === 1) {
    const x = normalizeSingleHand(handsList[0]);
    const t = new ort.Tensor("float32", x, [1, HAND_LANDMARKS * COORDS]);
    const r = await asamyuktaSession.run({ landmarks: t });
    const logits = Array.from(r.logits.data);
    const probs = smoothProbs(softmax(logits), "asamyukta");
    const idx = probs.indexOf(Math.max(...probs));
    return { name: classes.asamyukta[idx], conf: probs[idx], type: "asamyukta" };
  }
  // two hands
  const x = normalizeTwoHands(handsList[0], handsList[1]);
  const t = new ort.Tensor("float32", x, [1, 2 * HAND_LANDMARKS * COORDS]);
  const r = await samyuktaSession.run({ landmarks: t });
  const logits = Array.from(r.logits.data);
  const probs = smoothProbs(softmax(logits), "samyukta");
  const idx = probs.indexOf(Math.max(...probs));
  return { name: classes.samyukta[idx], conf: probs[idx], type: "samyukta" };
}

let lastVideoTime = -1;

async function loop() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const results = handLandmarker.detectForVideo(video, performance.now());

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const hands = results.landmarks || [];
    for (const h of hands) drawHand(h);

    const pred = await predict(hands);
    if (pred) {
      predEl.textContent = pred.name;
      confEl.textContent = `${pred.type} • ${Math.round(pred.conf * 100)}% confidence`;
    } else {
      predEl.textContent = "—";
      confEl.textContent = "Show one or two hands";
    }

    // Pose detection is best-effort context, not core to the demo — if it's
    // ever unavailable or unreliable, computePoseContext returns null and
    // interpretation still runs, just without body-position context.
    let context = null;
    if (poseLandmarker) {
      try {
        const poseResults = poseLandmarker.detectForVideo(video, performance.now());
        context = computePoseContext(poseResults.landmarks, hands);
      } catch (err) {
        console.warn("Pose detection failed, continuing without context:", err);
      }
    }
    updateInterpretationState(pred, context);
  }
  requestAnimationFrame(loop);
}

async function main() {
  statusEl.textContent = "Loading MediaPipe…";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  // Pose is used only for coarse body-position context passed to the LLM
  // (arm raised, hand near chest, etc.) — the "lite" model is enough for
  // that and keeps load time down. If it fails to load, we degrade
  // gracefully to context-free interpretation rather than blocking the demo.
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  } catch (err) {
    console.warn("PoseLandmarker failed to load, continuing without pose context:", err);
    poseLandmarker = null;
  }

  statusEl.textContent = "Loading classifiers…";
  asamyuktaSession = await ort.InferenceSession.create("./models/asamyukta_int8.onnx");
  samyuktaSession = await ort.InferenceSession.create("./models/samyukta_int8.onnx");
  classes = await fetch("./models/classes.json").then(r => r.json());

  statusEl.textContent = "Requesting camera…";
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
  });
  video.srcObject = stream;
  await new Promise(r => (video.onloadedmetadata = r));

  statusEl.textContent = "Live — show a mudra";
  loop();
}

main().catch(err => {
  console.error(err);
  statusEl.textContent = `Error: ${err.message}`;
});
