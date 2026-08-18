# MediaGenStudio

A local web app + GUI for generating images and videos on your own machine using
two real open-weight models:

- **Image generation:** [`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo)
  ("Krea 2" / "K2") — Krea.ai's 13B-parameter open-weight text-to-image model,
  with native `Krea2Pipeline` support in `diffusers` (>=0.39). The higher-quality,
  slower base checkpoint [`krea/Krea-2-Raw`](https://huggingface.co/krea/Krea-2-Raw)
  and the earlier FLUX-based collaboration
  [`black-forest-labs/FLUX.1-Krea-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)
  are both supported too - just change the repo id in Settings.
- **Video generation:** [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
  ("H3") — a 33B-parameter open-weight text/image-to-video+audio model.

## ⚠️ Hardware reality check

Both models are large:

- Krea 2 is 13B params (Turbo/Raw checkpoints). A single consumer GPU with
  16-24GB VRAM works with `enable_model_cpu_offload()` (slower). 8GB+ can work
  with sequential offload, much slower. Runs through the native `diffusers`
  `Krea2Pipeline` (>=0.39) - genuinely local, no cloud call involved. The Turbo
  checkpoint (8 steps, no CFG) is much faster than Raw (52 steps, CFG 3.5).
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

- **Image (Krea 2):** the engine's pipeline-selection logic and per-checkpoint
  defaults (Turbo: 8 steps/no CFG, Raw: 52 steps/CFG 3.5) were implemented against
  the official model cards; `Krea2Pipeline` was confirmed importable from the
  installed `diffusers` (no git install needed - it landed in the 0.39.0 stable
  release). A full end-to-end generation run was verified on a CPU-only test
  machine (no GPU, no Hugging Face account) using the older `FluxPipeline` path
  with a tiny public FLUX-compatible test model as a stand-in for the real
  (gated) weights - server boot, GUI, job queue/progress polling, and image
  saving all confirmed working through that shared code path. Device/dtype/
  CPU-offload settings auto-detect from your actual hardware.
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
   [Krea 2 Turbo model page](https://huggingface.co/krea/Krea-2-Turbo) (and/or
   [Krea 2 Raw](https://huggingface.co/krea/Krea-2-Raw) /
   [FLUX.1-Krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)
   if you want those instead) and the [H3 model page](https://huggingface.co/MiniMaxAI/MiniMax-H3).
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

## Deploying to Azure (cheapest tier)

`azure/deploy.ps1` deploys just the GUI/API to an Azure App Service **Free F1**
plan (Linux, Python 3.11) - the cheapest option, $0/month. This is a genuine
tradeoff: App Service has **no GPU**, so local Krea 2 / ComfyUI-driven H3
generation cannot run there. What you get is the web app itself, reachable
from anywhere, with the MiniMax hosted-API video fallback available if you set
`MINIMAX_API_KEY`. The deployment intentionally skips installing
torch/diffusers/transformers (see `azure/requirements.txt`) since the free/basic
tiers can't usefully run them anyway - image/local-video generation will show
the same clean "not installed"/"can't reach ComfyUI" errors as on any other
machine without a GPU, pointing you at `Settings > ComfyUI URL` if you want to
wire the deployed app up to a GPU machine you run elsewhere.

```powershell
az login   # sign in with the Azure account that should own the resources
./azure/deploy.ps1
```

Optional parameters: `-ResourceGroup`, `-Location`, `-WebAppName`, `-Sku` (e.g.
`-Sku B1` for the cheapest _paid_ tier - has "Always On" so the background job
worker doesn't get recycled during idle periods, unlike F1 which can unload the
app after ~20 minutes of no traffic).

**Region matters on restricted subscriptions.** Visual Studio/MSDN subscriptions
with an active spending limit often default to 0 App Service quota in most
regions - `eastus` and `westus2` both failed here with "Operation cannot be
completed without additional quota" (for both F1 and B1). `westeurope` worked
(confirmed by an existing App Service already running successfully there on
the same subscription) and is the script's default. If your deployment hits
the same quota error, either try another region or request a quota increase /
remove the spending limit in the Azure Portal.

Deployed and verified live at **https://mediagenstudio-1160.azurewebsites.net**
(resource group `mediagenstudio-rg`, region `westeurope`) - confirmed the GUI,
static assets, `/api/status`, `/api/config`, and the job queue/background
worker (submit -> process -> clean error message) all work end-to-end on the
real deployment, not just locally.

To tear everything down: `az group delete --name mediagenstudio-rg --yes --no-wait`.
