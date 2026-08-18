"""Video generation for MiniMax-H3 ("H3"): text/image-to-video+audio.

Two backends:
  - "diffusers": experimental local inference via diffusers' ModularPipeline,
    per https://huggingface.co/MiniMaxAI/MiniMax-H3. Needs a very recent
    diffusers build and, per the model card, ~4 GPUs for practical speed.
  - "api": calls MiniMax's hosted video-generation API as a fallback so the
    app is usable without a multi-GPU rig. Endpoint paths come from the
    model card's public docs links; verify the exact request/response schema
    against your MiniMax account's API docs (platform.minimax.io) since this
    is not independently verified end-to-end here.
"""
import base64
import os
import threading
import time

import requests

_pipeline = None
_pipeline_lock = threading.Lock()
_loaded_key = None


def _pipeline_key(cfg):
    v = cfg["video_model"]
    return (v.get("local_path") or v.get("repo_id"), v.get("variant"))


def _friendly_error(exc, model_path):
    msg = str(exc)
    lowered = msg.lower()
    if isinstance(exc, AttributeError) or "has no attribute" in lowered:
        return RuntimeError(
            "Your installed diffusers version doesn't yet support MiniMax-H3's "
            "pipeline classes - this model's local diffusers support is very new/"
            "bleeding-edge. Run 'pip install -U diffusers' (may need the git version: "
            "pip install git+https://github.com/huggingface/diffusers), or switch the "
            "video backend to 'api' in Settings, or use the ComfyUI workflow linked "
            "from https://huggingface.co/Comfy-Org/MiniMax-H3 instead."
        )
    if "gated" in lowered or "403" in msg or "access to model" in lowered:
        return RuntimeError(
            f"'{model_path}' is a gated model. Visit https://huggingface.co/{model_path}, "
            "log in, and click 'Agree and access repository', then set a Hugging Face "
            "access token (huggingface.co/settings/tokens) in Settings or the HF_TOKEN "
            "env var."
        )
    if "401" in msg or "authorization" in lowered or "unauthorized" in lowered:
        return RuntimeError(
            "Hugging Face rejected the request (401 unauthorized). Set a valid access "
            "token in Settings or the HF_TOKEN env var - create one at "
            "https://huggingface.co/settings/tokens."
        )
    return exc


def _load_pipeline(cfg, log=None):
    global _pipeline, _loaded_key
    key = _pipeline_key(cfg)
    with _pipeline_lock:
        if _pipeline is not None and _loaded_key == key:
            return _pipeline
        try:
            from diffusers import ModularPipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers is not installed, or your version does not yet "
                "support MiniMax-H3's ModularPipeline. Install/upgrade with "
                "'pip install -U diffusers', or switch the video backend to "
                "'api' in Settings."
            ) from exc

        v = cfg["video_model"]
        model_path = v.get("local_path") or v["repo_id"]
        token = cfg.get("hf_token") or None
        if log:
            log(f"Loading MiniMax-H3 ({v.get('variant')}) from '{model_path}'. "
                f"This is a 33B model - expect a very large download and multi-GPU memory needs.")
        try:
            pipe = ModularPipeline.from_pretrained(model_path, token=token)
        except Exception as exc:
            raise _friendly_error(exc, model_path) from exc
        _pipeline = pipe
        _loaded_key = key
        return pipe


def generate_local(cfg, prompt, image=None, duration=6, seed=None, log=None, progress_cb=None):
    pipe = _load_pipeline(cfg, log=log)  # raises a clear RuntimeError if diffusers is missing/too old
    import torch

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator = generator.manual_seed(int(seed))

    if progress_cb:
        progress_cb(1, 1)

    kwargs = dict(prompt=prompt, num_frames=int(duration * 24), generator=generator)
    if image is not None:
        kwargs["image"] = image

    output = pipe(**kwargs)
    return output


def generate_via_api(cfg, prompt, image_b64=None, duration=6, aspect_ratio="16:9",
                      seed=None, log=None, progress_cb=None):
    api_cfg = cfg["minimax_api"]
    api_key = os.environ.get(api_cfg.get("api_key_env", "MINIMAX_API_KEY"), "").strip()
    if not api_key:
        raise RuntimeError(
            f"Set the {api_cfg.get('api_key_env', 'MINIMAX_API_KEY')} environment "
            "variable with a MiniMax API key to use the hosted API backend."
        )

    base_url = api_cfg.get("base_url", "https://api.minimax.io").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "MiniMax-H3",
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
    }
    if image_b64:
        payload["first_frame_image"] = image_b64
    if seed is not None:
        payload["seed"] = int(seed)

    if log:
        log("Submitting job to MiniMax hosted API (video-generation-v2-create)...")
    resp = requests.post(f"{base_url}/v1/video-generation-v2-create", json=payload,
                          headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    base_resp = data.get("base_resp", {})
    if base_resp.get("status_code") not in (0, None):
        raise RuntimeError(f"MiniMax API error: {base_resp.get('status_msg', data)}")

    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"MiniMax API did not return a task_id: {data}")

    if progress_cb:
        progress_cb(10, 100)

    for attempt in range(180):
        time.sleep(5)
        status_resp = requests.get(
            f"{base_url}/v1/query/video-generation-v2",
            params={"task_id": task_id}, headers=headers, timeout=30,
        )
        status_resp.raise_for_status()
        status = status_resp.json()
        state = status.get("status")
        if progress_cb:
            progress_cb(min(10 + attempt, 95), 100)
        if state == "Success":
            file_id = status.get("file_id")
            file_resp = requests.get(
                f"{base_url}/v1/files/retrieve", params={"file_id": file_id},
                headers=headers, timeout=30,
            )
            file_resp.raise_for_status()
            return file_resp.json()["file"]["download_url"]
        if state == "Fail":
            raise RuntimeError(f"MiniMax video generation failed: {status}")

    raise TimeoutError("Timed out waiting for MiniMax video generation to finish.")


def image_path_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
