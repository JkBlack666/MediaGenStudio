# MediaGenStudio

A local web app + GUI for generating images and videos on your own machine using
two real open-weight models:

- **Image generation:** [`black-forest-labs/FLUX.1-Krea-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)
  ("Krea") — a 12B-parameter open-weight text-to-image model.
- **Video generation:** [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
  ("H3") — a 33B-parameter open-weight text/image-to-video+audio model.

> These are the actual models that best match "Krea" / "MiniMax H3" — there is no
> "Kolors 2" or separate "Krea 2" model publicly released, so this app targets the
> real, current open-weight releases from those two teams.

## ⚠️ Hardware reality check

Both models are large:

- Krea (FLUX) needs ~12B params in memory. A single consumer GPU with 16-24GB VRAM
  works with `enable_model_cpu_offload()` (slower). 8GB+ can work with sequential
  offload, much slower.
- MiniMax-H3 is 33B params and its own model card recommends **4 GPUs** for
  local inference via SGLang/vLLM. Running it on a single consumer GPU is not
  realistic today. Local diffusers support for H3 is also brand new/bleeding-edge
  and may change or fail depending on your installed `diffusers` version.

Because of this, the app supports two paths for video:

1. **Local diffusers** (`backend: "diffusers"`) — attempts real local inference.
   Experimental; needs the exact hardware above and a very recent `diffusers`.
2. **MiniMax hosted API** (`backend: "api"`) — calls MiniMax's official cloud API
   as a practical fallback so the app is actually usable without a multi-GPU rig.
   Requires a `MINIMAX_API_KEY`. Endpoint paths are taken from MiniMax's public
   docs (`platform.minimax.io`); verify request/response fields against your
   account's API docs before relying on it, since MiniMax may adjust the schema.

For a more mature local UI for these models, the ComfyUI workflow templates
linked from the model cards ([Krea ComfyUI](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev),
[H3 ComfyUI](https://huggingface.co/Comfy-Org/MiniMax-H3)) are the officially
recommended path — this app is a lightweight custom alternative with its own GUI.

## ✅ What's verified working

On a CPU-only test machine (no GPU, no Hugging Face account) the full app was
verified end-to-end: server boot, GUI, job queue/progress polling, and a real
generation run through `FluxPipeline` (using a tiny public FLUX-compatible test
model as a stand-in, since the real Krea/H3 weights are gated - see below).
Device/dtype/CPU-offload settings auto-detect from your actual hardware
(`"auto"` in config), so the app no longer assumes a CUDA GPU is present.

What's blocked on your action, not a bug:

- **Krea/H3 are gated Hugging Face models.** The app cannot create a Hugging
  Face account or click "agree" to a license on your behalf - only you can do
  that. Once you do (steps below) and paste a token into Settings, the exact
  same code path that was verified above will download and run the real models.
- **H3's local `diffusers` support is bleeding-edge.** As of testing,
  `diffusers` 0.39.0 from PyPI does not yet have the `MiniMaxH3ModularPipeline`
  class the model card's snippet references - you'll see a clear error
  pointing you to `pip install -U diffusers` (possibly the git version) or to
  switch to the `api` backend / ComfyUI. This may resolve itself as `diffusers`
  releases catch up.
- **MiniMax hosted API backend needs your own API key** from
  [platform.minimax.io](https://platform.minimax.io) (`MINIMAX_API_KEY` env var).

## Setup

```powershell
cd MediaGenStudio
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Both models are gated/licensed on Hugging Face - this is the one part only you
can do (it requires your own identity/account, an agent can't do it for you):

1. Sign in (or sign up, it's free) at [huggingface.co](https://huggingface.co).
2. Click "Agree and access repository" on the
   [Krea model page](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)
   and the [H3 model page](https://huggingface.co/MiniMaxAI/MiniMax-H3).
3. Create an access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Paste it into the app's Settings tab (or set an `HF_TOKEN` env var / run
   `huggingface-cli login`).

For the video hosted-API fallback, sign up at
[platform.minimax.io](https://platform.minimax.io) and set the resulting key as
`MINIMAX_API_KEY` in your environment.

Copy `config.example.json` to `config.json` and adjust model paths/device as
needed (or just let the app create it on first run with defaults - `"auto"`
values resolve against your real hardware).

## Run

```powershell
python server.py
```

Open http://127.0.0.1:7860 in your browser.

## Notes

- Generated images/videos are saved under `outputs/` and served at `/outputs/...`.
- Job history is persisted to `data/jobs.json` so it survives restarts.
- The app loads each model lazily on first request and keeps it resident in
  memory afterward (avoids reloading 12B/33B weights per request).
- Only one generation job runs at a time (single background worker) to avoid
  GPU contention.
