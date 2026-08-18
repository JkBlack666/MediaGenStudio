"""Drive a locally-running ComfyUI instance to generate MiniMax-H3 video+audio.

This is genuine 100% local execution: ComfyUI (with its native MiniMaxH3* nodes,
shipped since ComfyUI PR #15224) runs entirely on your machine/GPU using the
model files from https://huggingface.co/Comfy-Org/MiniMax-H3. No cloud API call
is involved here - this module only talks HTTP to your own ComfyUI server.

Confirmed empirically (2026-08-18): the public `diffusers` package (through
0.39.0, latest on PyPI) has no MiniMax-H3 pipeline classes at all, despite the
model card's diffusers usage snippet - so ComfyUI is currently the only
actually-working local backend for H3, not an alternative to one.

The graph built below mirrors ComfyUI's official "Image to Video (MiniMax H3)"
workflow template (video_minimax_h3_t2v.json / video_minimax_h3_i2v.json at
https://github.com/Comfy-Org/workflow_templates), flattened into the API
node-graph format ComfyUI's /prompt endpoint expects.
"""
import time
import uuid

import requests

CLIENT_ID = uuid.uuid4().hex


def _base_url(cfg):
    return cfg["video_model"].get("comfyui_url", "http://127.0.0.1:8188").rstrip("/")


def is_reachable(base_url):
    try:
        resp = requests.get(f"{base_url}/system_stats", timeout=3)
        return resp.ok
    except requests.RequestException:
        return False


def _upload_image(base_url, image_path):
    with open(image_path, "rb") as f:
        files = {"image": (image_path.split("/")[-1].split("\\")[-1], f)}
        resp = requests.post(f"{base_url}/upload/image", files=files, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["name"], data.get("subfolder", "")


def build_graph(params):
    """Build the flat ComfyUI API-format node graph for an H3 FL2VA generation."""
    v = params["video_model_cfg"]
    turbo = v.get("turbo", True)
    steps = v.get("steps_turbo", 8) if turbo else v.get("steps_full", 20)
    model_source = "134" if turbo else "127"

    graph = {
        "119": {"class_type": "VAELoader", "inputs": {"vae_name": v["vae_name"]}},
        "120": {"class_type": "VAELoader", "inputs": {"vae_name": v["audio_vae_name"]}},
        "127": {"class_type": "UNETLoader", "inputs": {"unet_name": v["unet_name"], "weight_dtype": "default"}},
        "128": {"class_type": "CLIPLoader", "inputs": {"clip_name": v["clip_name"], "type": "minimax", "device": "default"}},
        "134": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["127", 0], "lora_name": v["lora_name"], "strength_model": 1,
        }},
        "131": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["128", 0], "vae": ["119", 0],
            "prompt": params["prompt"],
            "width": params["width"], "height": params["height"], "length": params["length"],
        }},
        "123": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "124": {"class_type": "BasicScheduler", "inputs": {
            "model": [model_source, 0], "steps": steps, "scheduler": "simple", "denoise": 1,
        }},
        "129": {"class_type": "RandomNoise", "inputs": {"noise_seed": params["seed"]}},
        "126": {"class_type": "BasicGuider", "inputs": {"model": [model_source, 0], "conditioning": ["131", 0]}},
        "125": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["129", 0], "guider": ["126", 0], "sampler": ["123", 0],
            "sigmas": ["124", 0], "latent_image": ["131", 1],
        }},
        "122": {"class_type": "VAEDecode", "inputs": {"samples": ["125", 0], "vae": ["119", 0]}},
        "121": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["125", 0], "vae": ["120", 0]}},
        "130": {"class_type": "CreateVideo", "inputs": {"images": ["122", 0], "audio": ["121", 0], "fps": 24, "bit_depth": 8}},
        "92": {"class_type": "SaveVideo", "inputs": {
            "video": ["130", 0], "filename_prefix": "MediaGenStudio/h3", "format": "auto", "codec": "auto",
        }},
    }

    if params.get("first_frame_name"):
        graph["_first_frame_load"] = {"class_type": "LoadImage", "inputs": {"image": params["first_frame_name"]}}
        graph["131"]["inputs"]["first_frame"] = ["_first_frame_load", 0]

    return graph


def _queue(base_url, graph):
    resp = requests.post(f"{base_url}/prompt", json={"prompt": graph, "client_id": CLIENT_ID}, timeout=30)
    if not resp.ok:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"ComfyUI rejected the workflow: {detail}")
    return resp.json()["prompt_id"]


def _find_video_output(history_entry):
    for node_output in history_entry.get("outputs", {}).values():
        for key in ("videos", "gifs", "images"):
            for item in node_output.get(key, []):
                if isinstance(item, dict) and "filename" in item:
                    return item
    return None


# ~0.4 megapixels per the official workflow's resolution table, multiple-of-32.
_ASPECT_DIMS = {
    "16:9": (864, 480), "9:16": (480, 864), "1:1": (656, 656),
    "4:3": (768, 576), "3:4": (576, 768), "21:9": (992, 416),
}


def aspect_ratio_to_dims(aspect_ratio):
    return _ASPECT_DIMS.get(aspect_ratio, _ASPECT_DIMS["16:9"])


def generate(cfg, prompt, image_path=None, duration=6, seed=None, log=None, progress_cb=None,
             width=None, height=None, aspect_ratio="16:9", timeout_seconds=1800):
    base_url = _base_url(cfg)
    if not is_reachable(base_url):
        raise RuntimeError(
            f"Can't reach a ComfyUI server at {base_url}. This backend runs H3 fully "
            "locally through ComfyUI (not a cloud API) - install ComfyUI, add the "
            "MiniMax-H3 model files per https://huggingface.co/Comfy-Org/MiniMax-H3, "
            "start it (python main.py), then try again. Or switch to the 'api' backend "
            "in Settings if you don't have ComfyUI/a GPU set up."
        )

    if width is None or height is None:
        width, height = aspect_ratio_to_dims(aspect_ratio)

    length = max(5, round(duration * 24))
    length = length + (5 - (length % 17)) % 17  # snap to H3's 17-frame block grid

    params = {
        "prompt": prompt, "width": width, "height": height, "length": length,
        "seed": seed if seed is not None else int(time.time()),
        "video_model_cfg": cfg["video_model"],
    }
    if image_path:
        name, _ = _upload_image(base_url, image_path)
        params["first_frame_name"] = name

    graph = build_graph(params)
    if log:
        log(f"Queuing MiniMax-H3 workflow on local ComfyUI at {base_url}...")
    prompt_id = _queue(base_url, graph)

    deadline = time.time() + timeout_seconds
    elapsed = 0
    while time.time() < deadline:
        time.sleep(5)
        elapsed += 5
        resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=15)
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI workflow failed: {status.get('messages')}")
            video = _find_video_output(entry)
            if video:
                if progress_cb:
                    progress_cb(100, 100)
                view_resp = requests.get(
                    f"{base_url}/view",
                    params={"filename": video["filename"], "subfolder": video.get("subfolder", ""),
                            "type": video.get("type", "output")},
                    timeout=60,
                )
                view_resp.raise_for_status()
                return view_resp.content
        if progress_cb:
            progress_cb(min(95, int(elapsed / timeout_seconds * 100)), 100)
        if log:
            log(f"Waiting on ComfyUI ({elapsed}s elapsed)...")

    raise TimeoutError(f"ComfyUI didn't finish the H3 workflow within {timeout_seconds}s.")
