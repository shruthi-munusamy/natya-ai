# Natya

Real-time Bharatanatyam mudra recognition, grounded in traditional viniyoga
(meaning) interpretation via retrieval-augmented generation over a
domain-expert-authored knowledge base.

<!-- TODO: record a short demo GIF (mudra -> live classification -> streamed
interpretation) and drop it here. -->

**Live demo:** _not yet deployed — see [DEPLOY.md](./DEPLOY.md)._

## Problem

A mudra (hand gesture) in Bharatanatyam is not just a shape — it's a symbol
with a *viniyoga*, a traditional meaning documented in texts like the
Abhinaya Darpana, and the same mudra can carry different meanings depending
on where it's held and how it moves. Pataka above the head is a cloud;
Pataka at the chest can be something else entirely. Recognizing the mudra
is a (comparatively) tractable classification problem. Recognizing *what it
means in context*, faithfully and without inventing meanings that aren't
canonical, is the harder and more interesting problem — and it's where a
generic hand-gesture classifier stops being useful to an actual dancer or
student.

This project does both: a real-time, in-browser mudra classifier, and an
LLM layer that interprets detected mudras grounded strictly in a
domain-expert-authored knowledge base, so it explains meaning instead of
just naming a class label.

## System architecture

```
Webcam
  │
  ▼
MediaPipe Hands + Pose (JS, on-device, WASM)
  │  21 keypoints/hand, 33 pose landmarks
  ▼
Landmark normalization (wrist-centered, scale-normalized)
  │
  ▼
ONNX classifier (in-browser, onnxruntime-web)
  │  1 hand  -> asamyukta model (28 classes, attention)
  │  2 hands -> samyukta model  (16 classes, attention)
  ▼
Temporal smoothing (10-frame softmax average)
  │
  ▼
Stability gate (same top-1 label, >1s, >60% confidence)
  │
  ▼                                    Pose-derived context
  │                                    (arm raised / near chest /
  │                                    near face / extended forward)
  ▼                                              │
POST /interpret/stream  ◄───────────────────────┘
  │  (FastAPI + SSE)
  ▼
ChromaDB retrieval (top-6 viniyogas by embedding similarity,
  filtered to detected mudra)
  │
  ▼
Llama 3.3-class LLM (Groq), constrained to retrieved viniyogas only
  │
  ▼
Streamed natural-language interpretation → browser
```

Everything left of the `POST` runs entirely client-side — no server
round-trip for classification, which is what makes it feel real-time.

## Results

### A train/serve skew bug, found and fixed

Live testing on a real webcam kept producing misclassifications on mudras
that are geometrically easy to tell apart (e.g. Tripataka vs. Mayura: the
thumb sits outside the fist for one, connects to the ring finger for the
other), while held-out validation accuracy stayed above 97%. That gap
between "validation looks great" and "obviously-distinct shapes fail live"
was the signal that something more specific than generic distribution shift
was going on.

Root cause: the dataset's landmarks were originally extracted with
`mp.solutions.hands` (MediaPipe's legacy Python API), but the browser's live
inference uses `@mediapipe/tasks-vision` HandLandmarker (the newer Tasks
API) — same 21-point format, but a different underlying landmark-detection
model with measurably different coordinate precision. Verified directly: ran
the then-trained classifier on 222 dataset images, once using landmarks from
each API for the *same* images. Legacy-API landmarks (what training used):
**100%** accuracy. Tasks-API landmarks (what the browser actually sends):
**88.3%** — and every single error went that direction; Tasks-API landmarks
never fixed a case Legacy-API got wrong. That's a train/serve skew, not
dataset-vs-webcam distribution shift: the classifier had never actually seen
the kind of landmarks it gets at inference time.

Fix: rewrote `src/features/extractor.py` to use the Tasks-vision API (the
exact same model files `web/app.js` loads) instead of the legacy API,
re-extracted landmarks for the full 28,431-image dataset, and retrained both
classifiers on the corrected cache.

First re-check of the retrained model used the same 222 images as before and
showed 100% accuracy, 0 errors — which turned out to be a flawed test: 81%
of those images were in the *training* set, not held out, so that number
was mostly memorization, not generalization. Caught this, redid it properly
using only genuinely held-out validation images for the same problem
classes (909 images, confirmed excluded from training): **96.4%** accuracy,
32 errors — consistent with the model's overall held-out accuracy below,
not a suspicious perfect score. Katakamukha_1 and Tripataka still show real
residual confusion in this proper test (e.g. Katakamukha_1 confused with
Bramaram, Mrigasirsha, Ardhachandran, Chaturam, Simhamukham, Kangulam across
different validation images). So: the specific train/serve mismatch this
section describes is real and the fix is real — that conclusion doesn't
depend on the flawed re-check, since it came from a separate, correctly-
held-out before/after comparison on the *original* model. What's corrected
here is the claim that the fix eliminated error entirely; it didn't.
Separately, genuine distribution shift between the dataset's studio-style
photos and a particular live webcam (different camera, lighting,
background, hand presentation) is still a separate, real effect this fix
doesn't touch — the
numbers below are the corrected, now train/serve-consistent baseline, not a
guarantee that every live-webcam error reported earlier is gone. That needs
a live retest to confirm.

One side effect worth noting plainly: validation accuracy is *lower* than
it was before this fix (asamyukta 96.20% vs. the previous 97.70%). That's
not a regression — the old number was measured on the same Legacy-API
landmarks the model trained on, which is exactly the comparison that hid
this bug. The new number is lower because Tasks-API landmarks are
genuinely slightly noisier, but it's the number that actually reflects what
ships to the browser.

### Classifier accuracy

Two separate classifiers, split by hand count, because asamyukta
(single-hand) and samyukta (double-hand) mudras are geometrically distinct
problems — see [Design decisions](#design-decisions).

**Asamyukta (28 classes, attention model)** — top-1 **96.20%**, top-3
**98.64%**, macro F1 **0.962**, on a held-out validation split of 2,208
samples. (See the train/serve skew note above for why this is lower than an
earlier 97.70% — this number is the one that's actually consistent with
production.)

**Samyukta (16 classes, attention model)** — top-1 **99.69%**, top-3
**100.00%**, macro F1 **0.997**, on a held-out validation split of 651
samples. Attention now edges out the MLP baseline (98.77%) by about 1pp —
previously MLP had tied or slightly beaten attention here; see
[Design decisions](#design-decisions) for why that flipped.

The tables below use the classifier's raw output labels, which follow the
Samarth0710 dataset's spellings (e.g. "Pathaka", "Tripathaka"). The app maps
these to standard Sanskrit transliteration ("Pataka", "Tripataka") for
display and for KB/LLM lookups — see
[`web/models/mudra_display_names.json`](web/models/mudra_display_names.json).
The dataset spellings are kept here since that's what the model actually
outputs and what the confusion matrices are indexed by.

<details>
<summary>Per-class F1 (asamyukta)</summary>

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Aralam | 0.950 | 0.941 | 0.945 | 101 |
| Mayura | 0.990 | 0.990 | 0.990 | 97 |
| Sikharam | 0.979 | 1.000 | 0.989 | 92 |
| Ardhachandran | 0.968 | 1.000 | 0.984 | 91 |
| Kapith | 0.924 | 0.944 | 0.934 | 90 |
| Trishulam | 0.977 | 0.944 | 0.960 | 90 |
| Chandrakala | 1.000 | 0.988 | 0.994 | 84 |
| Tripathaka | 0.962 | 0.893 | 0.926 | 84 |
| Hamsasyam | 0.976 | 0.964 | 0.970 | 83 |
| Katrimukha | 0.929 | 0.940 | 0.934 | 83 |
| Ardhapathaka | 0.929 | 0.975 | 0.952 | 81 |
| Mushti | 0.988 | 0.975 | 0.981 | 81 |
| Pathaka | 0.951 | 0.963 | 0.957 | 81 |
| Shukatundam | 0.963 | 0.951 | 0.957 | 81 |
| Suchi | 0.988 | 0.988 | 0.988 | 80 |
| Chaturam | 0.963 | 1.000 | 0.981 | 79 |
| Katakamukha_3 | 0.987 | 0.974 | 0.980 | 77 |
| Padmakosha | 0.987 | 0.974 | 0.980 | 77 |
| Tamarachudam | 0.971 | 0.895 | 0.932 | 76 |
| Bramaram | 0.912 | 0.973 | 0.942 | 75 |
| Katakamukha_1 | 0.917 | 0.880 | 0.898 | 75 |
| Sarpasirsha | 1.000 | 0.973 | 0.986 | 73 |
| Simhamukham | 0.932 | 0.944 | 0.938 | 72 |
| Katakamukha_2 | 0.957 | 0.985 | 0.971 | 68 |
| Kangulam | 0.938 | 0.968 | 0.953 | 63 |
| Mukulam | 0.984 | 0.984 | 0.984 | 61 |
| Mrigasirsha | 0.966 | 0.982 | 0.974 | 57 |
| Alapadmam | 0.946 | 0.946 | 0.946 | 56 |

</details>

<details>
<summary>Per-class F1 (samyukta)</summary>

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Kapotham | 1.000 | 0.990 | 0.995 | 97 |
| Shanka | 1.000 | 1.000 | 1.000 | 74 |
| Anjali | 0.984 | 1.000 | 0.992 | 62 |
| Kartariswastika | 1.000 | 1.000 | 1.000 | 59 |
| Garuda | 1.000 | 1.000 | 1.000 | 44 |
| Pasha | 1.000 | 1.000 | 1.000 | 44 |
| Karkatta | 1.000 | 1.000 | 1.000 | 38 |
| Katakavardhana | 1.000 | 1.000 | 1.000 | 37 |
| Swastikam | 0.973 | 1.000 | 0.986 | 36 |
| Kilaka | 1.000 | 1.000 | 1.000 | 35 |
| Sakata | 1.000 | 1.000 | 1.000 | 33 |
| Khatva | 1.000 | 1.000 | 1.000 | 30 |
| Nagabandha | 1.000 | 0.955 | 0.977 | 22 |
| Berunda | 1.000 | 1.000 | 1.000 | 21 |
| Pushpaputa | 1.000 | 1.000 | 1.000 | 13 |
| Samputa | 1.000 | 1.000 | 1.000 | 6 |

Per-class support shifted somewhat from the previous table (e.g. Sakata
96→33) — the Tasks-vision API has different per-class occlusion sensitivity
than the legacy API it replaced, so "both hands detected" rates aren't
identical class-by-class even though the aggregate rate barely moved
(~3,284 vs ~3,292 samples total). Checked directly, not just asserted.

</details>

| Asamyukta confusion matrix | Samyukta confusion matrix |
|---|---|
| ![asamyukta confusion matrix](eval/confusion_matrix_asamyukta.png) | ![samyukta confusion matrix](eval/confusion_matrix_samyukta.png) |

Full numbers: [`eval/classifier_metrics.json`](eval/classifier_metrics.json)
(regenerate with `python scripts/eval_all_classifiers.py`).

### LLM faithfulness: RAG vs. no-RAG

20 (mudra, context) test cases, run through the same model with and without
retrieval, scored against the canonical KB for grounding, hallucination, and
context-sensitivity. Methodology (lexical-overlap heuristic against the
model's named claims, not semantic entailment) and full transcripts are in
[`eval/llm_faithfulness.json`](eval/llm_faithfulness.json) — worth spot-
checking by hand, not just trusting the aggregate.

| Metric | With RAG | Without RAG |
|---|---|---|
| Grounded in a canonical viniyoga | **95%** | 45% |
| Hallucinated (claimed a non-canonical viniyoga) | **0%** | 10% |
| Context-sensitive (top-mentioned meaning matches the context given) | **100%** | 20% |

(One RAG case returned an empty response — a transient Groq API hiccup during
that run, not a grounding failure — which is why grounded isn't 100%; see
the transcript for that case in the JSON.)

Regenerate with `python scripts/eval_llm.py` (requires `GROQ_API_KEY`).

### Latency

| Stage | P50 | P95 | Mean |
|---|---|---|---|
| Asamyukta classifier inference¹ | 0.17 ms | 0.56 ms | 0.23 ms |
| Samyukta classifier inference¹ | 0.01 ms | 0.01 ms | 0.01 ms |
| LLM time-to-first-token (HTTP, Groq) | 679 ms | 1567 ms | 787 ms |
| End-to-end² (classify → first token) | ~679 ms | ~1568 ms | ~787 ms |

¹ Measured via onnxruntime's native CPU execution provider as a proxy for
onnxruntime-web's WASM backend — WASM in-browser will be slower than this,
but classifier inference is not the bottleneck either way; the LLM call
dominates by 3+ orders of magnitude.
² Excludes the UI's deliberate 1000ms stability debounce (`STABLE_MS` in
`web/app.js`) — that's a UX choice to avoid firing on transient hand shapes,
not a pipeline cost. Regenerate with `python scripts/eval_latency.py`
(needs the server running locally).

## Design decisions

**Attention wins on both asamyukta and samyukta — a conclusion that changed
once a bug got fixed.** Asamyukta classification benefits from attention
over the 21 keypoints because distinguishing shapes like Katakamukha's
sub-variants depends on relative keypoint geometry the attention layer can
learn explicitly. On samyukta, the original finding (on the pre-fix,
legacy-API landmarks) was that attention and the MLP baseline tied — the
hypothesis at the time was that samyukta's smaller per-class sample count
(MediaPipe only detects both hands in ~24% of double-hand samples — see
[Limitations](#real-world-limitations)) left attention nothing extra to
learn over the 126-dim concatenated MLP features. After retraining on the
corrected Tasks-API landmarks (see the train/serve skew section above),
attention edges out the MLP by about 1pp (99.69% vs. 98.77%) on the same
data split. That doesn't fully invalidate the original reasoning — low
sample count is still real — but it shows the *landmark noise
characteristics*, not sample count alone, were doing more of the work in
that original "tie" than the first write-up gave them credit for. Worth
stating plainly rather than quietly editing the old claim away.

**RAG over a fine-tuned or trained viniyoga model.** Viniyoga interpretation
isn't a classification problem with a fixed label space — it's contextual
selection over an authored knowledge base that can grow, and it needs to be
auditable (which canonical entry did the model pick, and why) rather than
baked into opaque weights. Retrieval also means updating the KB (adding a
mudra, correcting a translation) doesn't require retraining anything.

**ONNX + INT8 + in-browser inference.** No inference server for
classification means no per-request cost, no latency floor from a network
round trip, and no privacy concern about streaming someone's webcam feed to
a server. INT8 dynamic quantization cuts both models significantly
(asamyukta: 456KB → 193KB; samyukta: 1404KB → 473KB — larger than before
since samyukta now uses the attention architecture, not the smaller MLP)
with negligible accuracy loss, since landmark coordinates are a
well-behaved input distribution for weight-only quantization. Both are
still well within browser-load budget.

**Llama (70B-class) via Groq.** Free-tier, fast (Groq's LPU inference is
materially faster time-to-first-token than most hosted alternatives at this
model size), and OpenAI-compatible, so swapping providers later is a
one-line change to `LLM_MODEL` in `src/llm/interpreter.py`. (Originally
built against `llama-3.3-70b-versatile`; Groq retired that model from the
free tier mid-project, so the shipped version uses `openai/gpt-oss-120b` —
a real example of a dependency on a third-party model catalog changing
under a live project, handled by isolating the model name to one constant.)

**Temporal smoothing, at three separate layers.** Raw per-frame predictions
are noisy — a hand mid-transition between two mudras produces flickering
top-1 labels frame to frame. Averaging softmax distributions over a
10-frame window (~333ms at 30fps) smooths the underlying *decision*, but
live testing showed that wasn't sufficient on its own: the *displayed*
label was still raw argmax of that average recomputed every frame, which
could itself flip near a decision boundary (observed: one hand angle
flashed through ~10 different labels in under a second), and pose-derived
context (used for LLM grounding) was recomputed from unsmoothed landmarks
every frame, so threshold-adjacent values like "near chest" vs. "near face"
flickered on a hand that never moved — which fed a changing key into the
interpretation dedup and caused the app to abort and restart the LLM call
before any text could render, reading as "it never finishes thinking."
Fixed with two more layers: an EMA filter on raw landmark positions
(reduces jitter in the overlay and in what both the classifier and pose
context see), and hysteresis on the *displayed* label and the *context
string actually sent to the LLM* (require N consecutive matching frames /
majority-vote over a short window before accepting a change), independent
of the softmax-level smoothing. All three reset on hand loss or count
change so none of it carries stale state across a different gesture space.

**The knowledge base is authored, not scraped.** Viniyoga interpretation is
specific enough, and wrong often enough in generic web sources, that an LLM
answering from parametric knowledge alone hallucinates or picks
plausible-sounding but non-canonical meanings (see the no-RAG hallucination
numbers above). The KB entries were AI-drafted from the Abhinaya Darpana and
then verified against 14+ years of Bharatanatyam training — the same
standard I'd want an authoritative source to meet regardless of whether an
LLM helped write it.

## Real-world limitations

The validation metrics above are near-perfect because the validation split
comes from the *same dataset distribution* as training (same collection
conditions, similar subjects). They do not reflect what happens under
distribution shift on a novel webcam, and that gap is worth stating plainly
rather than hiding behind a 97-99% headline number:

- **This is static-pose recognition, not motion or sequence understanding —
  a real ceiling on what "viniyoga interpretation" means here, not a bug.**
  The classifier looks at one hand shape per frame and nothing else. It has
  no notion of two *different* mudras held simultaneously on two hands (the
  samyukta model only knows 16 specific *combined* two-hand shapes like
  Anjali or Garuda — it cannot represent, say, Katakamukha on one hand and
  Shikara on the other), no notion of a gesture traced through space (a
  bow-drawing motion across the chest), and no notion of a mudra defined by
  continuous motion between two hand shapes (Sandamsha — also not in the
  dataset's class list at all, so it's doubly out of scope). Static-image
  classification was the right scope for this iteration — it's what the
  Samarth0710 dataset supports — but extending to real sequence/motion
  understanding would mean a different model architecture (temporal/
  sequence modeling) and new training data that doesn't exist yet, not a
  parameter to tune in the current one.
- **Partial-bend transitions aren't recognized until fully formed.** The
  dataset only contains static, fully-formed poses — no transition frames —
  so mudras that differ by degree of finger bend (Pataka → Tripataka as the
  ring finger bends, Pataka ↔ Ardhapataka, Arala → Shukatunda) don't switch
  labels until the gesture matches the dataset's convention for the target
  class, not at the biomechanical halfway point a dancer would expect.
  Tripataka specifically: the dataset's samples use a full ring-finger bend
  rather than the canonical half-bend, so the model learned the dataset's
  convention, not the canonical form.
- **A specific root cause for much of the live-testing confusion below was
  found and fixed — see [the train/serve skew section](#a-trainserve-skew-bug-found-and-fixed)
  above — but it hasn't been live-retested yet.** The following confusion
  pairs (Chandrakala effectively always read as Suchi; Bhramara →
  Katakamukha and, separately, Bhramara → Hamsasya; Katakamukha_1 →
  Hamsasya; Shikara ↔ Kapita; Mrigasirsha → Katakamukha_2; Chatura →
  Shukatunda; Tripataka → Mayura; and Mushti → Katakamukha_2 — Mushti is
  otherwise the easiest shape in the set, a closed fist, which made this one
  the most surprising) were all observed *before* the landmark-extraction
  fix, using the classifier that had been trained on a different API's
  landmarks than the browser ever sent it. A controlled before/after
  diagnostic confirmed that specific mismatch alone accounts for a real,
  measured accuracy drop (100% → 88.3% on identical images). It's plausible
  — but not yet confirmed — that fixing it resolves some or most of these
  specific pairs. What the fix does *not* address is separate, genuine
  distribution shift between the dataset's studio-style collection
  conditions and any particular live webcam (lighting, camera, background,
  hand presentation), which is a real, different effect. In every observed
  case the hand-landmark overlay rendered correctly on screen, which was
  the basis for concluding (at the time) that these were classifier
  decision-boundary failures rather than upstream MediaPipe detection
  failures — that reasoning was correct as far as it went, but incomplete,
  since it didn't yet account for which *version* of MediaPipe landmarks
  the classifier had actually learned from. Checked directly against the
  held-out validation confusion matrix at the time: all of these pairs
  showed **zero** validation-set misclassifications except Katakamukha_1 →
  Hamsasya (1 out of 77). (Caveat: the original list came from qualitative
  testing sessions by one person on one camera setup, not a systematic
  re-evaluation — treat it as a set of observed failure examples from
  before the fix, not a current, exhaustive, or statistically characterized
  error rate. This whole bullet needs an update once retested live.)
- **The model only knows the mudras it was trained on.** Showing it a real
  Bharatanatyam mudra outside the Samarth0710 dataset's 28+16 classes (e.g.
  Hamsapaksha) isn't a "misclassification" in any meaningful sense — a
  closed-set classifier has no "none of the above" option and will always
  report its nearest match among the classes it knows, regardless of how
  little the input actually resembles that class.
- **MediaPipe's two-hand detection rate**: only ~24% of the dataset's
  double-hand samples have both hands detected by MediaPipe at all (24.2%
  under the Tasks-vision API, essentially unchanged from the legacy API) —
  hand occlusion when hands are close together is precisely when samyuktas
  happen. The samyukta model is trained and evaluated on the ~3,254 usable
  samples that survive that filter; the other 76% of double-hand images
  were simply undetectable at the landmark-extraction stage, upstream of
  anything the classifier could fix.

## Setup and reproduction

```bash
# 1. Environment (Python 3.11 required — mediapipe has no 3.13 wheel)
conda create -n natya python=3.11 -y
conda activate natya
pip install -r requirements.txt

# 2. Secrets
cp .env.example .env
# edit .env and set GROQ_API_KEY (https://console.groq.com/keys)

# 3. Dataset + landmark extraction (downloads Samarth0710/bharatanatyam-mudra-dataset,
#    ~28k images; caches extracted landmarks to data/landmarks.npz)
python scripts/extract_landmarks.py

# 4. Train both classifiers (checkpoints saved to checkpoints/)
python scripts/train_asamyukta.py
python scripts/train_samyukta.py

# 5. Export to ONNX + quantize (writes web/models/*.onnx, classes.json)
python -m src.deploy.export_onnx
python -m src.deploy.quantize

# 6. Build the viniyoga vector index (writes data/kb/chroma/)
python -m src.kb.retrieval

# 7. Run the eval suite
python scripts/eval_all_classifiers.py
python scripts/eval_llm.py
python scripts/eval_latency.py   # needs the server running (step 9) in another terminal

# 8. Serve the frontend
cd web && python -m http.server 8000

# 9. Run the interpretation backend (separate terminal)
uvicorn src.llm.server:app --port 8001
```

Then open `http://localhost:8000`, grant camera access, and show a mudra.

Deploying to Render + Vercel: see [DEPLOY.md](./DEPLOY.md).

**Troubleshooting a `protobuf`/`mediapipe` import error:** if `import mediapipe`
raises `AttributeError: 'MessageFactory' object has no attribute
'GetPrototype'`, something in the environment has pulled in `protobuf>=5`
(mediapipe needs `<5,>=4.25.3`; chromadb's `opentelemetry-proto` dependency
wants `>=5` — a genuine conflict between this project's training and
serving halves in one shared environment, hit firsthand after installing a
few extra packages into an already-set-up env over time, not from a clean
`pip install -r requirements.txt`). Fix: `pip install "protobuf==4.25.3"`.
That'll print a pip conflict warning about `opentelemetry-proto` — harmless
for the training-side scripts, since they never import chromadb.

## Tech stack

- **Vision**: MediaPipe Hands + Pose (Python for dataset landmark
  extraction, JS/WASM for live browser inference)
- **Classifiers**: PyTorch (MLP + Transformer-encoder attention), exported
  to ONNX (opset 17), INT8 dynamic-quantized, served via onnxruntime-web
- **Retrieval**: ChromaDB (persistent, cosine similarity) +
  sentence-transformers `all-MiniLM-L6-v2`
- **LLM**: Groq-hosted Llama-class model, OpenAI-compatible streaming API
- **Backend**: FastAPI + sse-starlette (SSE streaming), uvicorn
- **Frontend**: vanilla JS (ES modules), no framework
- **Deploy**: Render (API), Vercel (static frontend)

## Attribution

- Dataset: [Samarth0710/bharatanatyam-mudra-dataset](https://huggingface.co/datasets/Samarth0710/bharatanatyam-mudra-dataset)
  on Hugging Face.
- Viniyoga knowledge base: drafted with AI assistance from the *Abhinaya
  Darpana* of Nandikeshvara, verified by the author against 14+ years of
  Bharatanatyam training.
