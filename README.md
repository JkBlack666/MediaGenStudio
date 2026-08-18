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
  offload, much slower. This runs through the standard `diffusers` `FluxPipeline` -
  genuinely local, no cloud call involved.
- MiniMax-H3 is 33B params. Its model card recommends **4 GPUs** for full-precision
  SGLang/vLLM serving, but ComfyUI ships official **quantized** checkpoints
  (int8/fp8) plus a turbo LoRA specifically so it can run on a single well-specced
  consumer GPU (see [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)).
  You still need a real GPU with substantial VRAM - this is not something that runs
  usefully on CPU-only hardware.

### How video generation actually runs locally

The public `diffusers` package has **no MiniMax-H3 pipeline support in any release
yet** (confirmed against 0.39.0, the latest on PyPI as of this writing) - despite
the model card's diffusers code snippet. The `MiniMaxH3ImageToVideo` node the
model card references only exists inside **ComfyUI itself** (merged in
[ComfyUI#15224](https://github.com/Comfy-Org/ComfyUI/pull/15224)).

So the app's video backends are:

1. **`comfyui` (default)** - genuinely local. The app drives a **ComfyUI server
   running on your own machine/GPU** over HTTP (`/prompt`, `/history`, `/view`),
   using the same node graph as ComfyUI's official H3 workflow templates. No
   cloud API call is made by this backend at all. Requires you to install
   ComfyUI and the H3 model files yourself (see Setup below) and have it running
   before you click Generate.
2. **`api`** - MiniMax's hosted cloud API. Explicitly **not local** - a practical
   fallback if you don't have a GPU. Requires `MINIMAX_API_KEY`.
3. **`diffusers`** - kept for when/if `diffusers` adds real H3 support upstream;
   currently fails with a clear message since that support doesn't exist yet.

## ✅ What's verified working

- **Image (Krea/FLUX):** verified end-to-end on a CPU-only test machine (no GPU,
  no Hugging Face account) - server boot, GUI, job queue/progress polling, and a
  real generation run through `FluxPipeline` (using a tiny public FLUX-compatible
  test model as a stand-in, since the real Krea weights are gated - see below).
  Device/dtype/CPU-offload settings auto-detect from your actual hardware.
- **Video (H3) `comfyui` backend:** the node graph this app builds
  (`MiniMaxH3ImageToVideo` with `clip`/`vae`/`prompt`/`width`/`height`/`length`/
  `first_frame`/`last_frame` inputs, wired through `BasicGuider` →
  `SamplerCustomAdvanced` → `VAEDecode`/`VAEDecodeAudio` → `CreateVideo` →
  `SaveVideo`) was checked line-by-line against ComfyUI's actual node source
  (`comfy_extras/nodes_minimax_h3.py`) and matches exactly. A live end-to-end run
  wasn't done on this machine since it has no GPU at all (a Hyper-V/AVD virtual
  display only) - installing and running ComfyUI itself needs a real GPU to be
  useful. The "ComfyUI not reachable" and job-error paths were verified for real.

What's blocked on your action, not a bug:

- **Krea/H3 are gated Hugging Face models.** The app cannot create a Hugging
  Face account or click "agree" to a license on your behalf - only you can do
  that. Once you do (steps below) and paste a token into Settings, the exact
  same code path that was verified above will download and run the real models.
- **The `comfyui` backend needs ComfyUI installed and running with the H3 model
  files** in place (see Setup) - the app is a client that drives it, not a
  replacement for it.
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

For local video generation (the default `comfyui` backend), install
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) separately, download the
model files listed on [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
into ComfyUI's `models/` folders, and start it (`python main.py`) before
generating video - the app's Settings tab lets you point at a non-default
ComfyUI URL/port and change which model filenames it requests.

For the video hosted-API fallback instead, sign up at
[platform.minimax.io](https://platform.minimax.io) and set the resulting key as
`MINIMAX_API_KEY` in your environment, then switch the backend to `api`.

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
