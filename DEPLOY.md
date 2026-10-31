# Deploying Natya

Everything except account connection is already set up. This should take
about 10 minutes.

## 1. Backend — Render

The backend is `src/llm/server.py` (FastAPI + SSE), configured by `render.yaml`
and `requirements-server.txt` (skips mediapipe/opencv/datasets — the training
pipeline's deps, not needed to serve the API). No torch either: the first
deploy used sentence-transformers, which pulled in torch and ran out of
memory on the free tier's 512MB. Embeddings now come from Chroma's ONNX
build of the same model (`src/kb/retrieval.py`, identical vectors): a
~500MB install, and ~380MB peak for the running server in a clean-venv
test, index rebuild included.

1. Go to [render.com](https://render.com) and sign in / sign up (GitHub OAuth is easiest).
2. **New > Blueprint**, connect the `shruthi-munusamy/natya-ai` repo. Render
   will detect `render.yaml` and propose a service named `natya-api`.
3. When prompted for the `GROQ_API_KEY` env var (marked `sync: false` in
   `render.yaml` so it isn't committed), paste your key from
   [console.groq.com/keys](https://console.groq.com/keys).
   - **Rotate this key first** if it's the same one that was ever in a
     committed file — see the note at the bottom of this doc.
4. Deploy. First build takes a few minutes (installs chromadb, onnxruntime,
   etc., then downloads the ~80MB embedding model). The vector index rebuilds automatically on server
   startup from `data/kb/viniyoga.json` (see `build_index_if_empty` in
   `server.py`) — no manual step needed.
5. Once live, note the URL, something like `https://natya-api.onrender.com`.
   Confirm it's healthy: `curl https://natya-api.onrender.com/health` should
   return `{"status":"ok"}`.

**Free tier note:** Render's free web services spin down after 15 minutes of
inactivity and take ~30-50s to cold-start on the next request. The first
interpretation after idle time will be slow — this is a free-tier
characteristic, not a bug. Worth mentioning if you demo this live.

## 2. Frontend — Vercel

The frontend is the static `web/` directory (no build step).

1. Go to [vercel.com](https://vercel.com), sign in with GitHub.
2. **Add New > Project**, import `shruthi-munusamy/natya-ai`.
3. `vercel.json` at the repo root sets `outputDirectory: web`, so the
   default "Other" framework preset with no build command should just work.
   If Vercel's UI asks for a Root Directory instead, set it to `web` and you
   can skip `vercel.json` behavior entirely — either approach works.
4. Deploy. You'll get a URL like `https://natya-ai.vercel.app`.

## 3. Wire the two together

1. `web/app.js` picks the backend by hostname: `localhost` uses
   `http://localhost:8001`, anything else uses `https://natya-api.onrender.com`.
   If your Render URL differs, change that one string.
2. On Render, set the `ALLOWED_ORIGINS` env var (already templated in
   `render.yaml` to `https://natya-ai.vercel.app`) to your actual Vercel
   URL if it differs, then redeploy. `src/llm/server.py` reads this to add
   to its CORS allowlist alongside localhost.
3. Vercel redeploys on every push to `main` once the repo is connected.
4. Open the Vercel URL, grant camera access, show a mudra, confirm the
   interpretation panel streams text from the deployed backend (check the
   Network tab for a request to your Render URL, not localhost).

## Rotate the Groq key

`.env.example` briefly contained a live Groq API key (same one as `.env`)
before being cleaned up to a placeholder. It was never pushed to GitHub
(confirmed via `git log -p -- .env.example`), but it did sit in a
plaintext, non-gitignored file on disk. As good hygiene, rotate it at
[console.groq.com/keys](https://console.groq.com/keys) — revoke the old key,
generate a new one, and update it in both your local `.env` and the Render
dashboard's `GROQ_API_KEY` env var.
