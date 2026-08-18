"""Config loading/merging for MediaGenStudio.

config.json (gitignored, holds local machine settings) is created from
config.example.json on first run if missing, then merged over defaults.
"""
import json
import copy
from pathlib import Path

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"
EXAMPLE_FILE = BASE_DIR / "config.example.json"

DEFAULTS = {
    # "auto" resolves against the actual machine at load time (see
    # resolve_hardware below) instead of hard-coding a GPU that may not exist.
    "device": "auto",
    "dtype": "auto",
    "hf_token": "",
    "image_model": {
        "repo_id": "black-forest-labs/FLUX.1-Krea-dev",
        "local_path": "",
        "offload": "auto",
    },
    "video_model": {
        "repo_id": "MiniMaxAI/MiniMax-H3",
        "local_path": "",
        "variant": "FL2VA",
        # "comfyui": genuinely local, drives a local ComfyUI server (default -
        # this is the only backend that actually runs H3 on your own machine
        # today; "diffusers" support for H3 doesn't exist in any public
        # diffusers release yet, confirmed 2026-08-18 against 0.39.0).
        # "api": MiniMax's hosted cloud API (not local, needs MINIMAX_API_KEY).
        "backend": "comfyui",
        "comfyui_url": "http://127.0.0.1:8188",
        "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "vae_name": "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae_name": "minimax_h3_audio_vae_fp32.safetensors",
        "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "turbo": True,
        "steps_full": 20,
        "steps_turbo": 8,
    },
    "minimax_api": {
        "base_url": "https://api.minimax.io",
        "api_key_env": "MINIMAX_API_KEY",
    },
}


def resolve_hardware(cfg):
    """Resolve 'auto' device/dtype/offload settings against the real machine.

    Returns dict with concrete device/dtype_name/offload ready to hand to
    torch/diffusers. Safe to call even if torch isn't installed yet (falls
    back to cpu/float32/none in that case; callers still need torch to
    actually run anything).
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if cuda_available else "cpu"

    dtype_name = cfg.get("dtype", "auto")
    if dtype_name == "auto":
        dtype_name = "bfloat16" if device == "cuda" else "float32"

    offload = cfg.get("image_model", {}).get("offload", "auto")
    if offload == "auto":
        # cpu_offload only makes sense when shuttling layers to/from a GPU.
        offload = "model" if device == "cuda" else "none"

    return {"device": device, "dtype_name": dtype_name, "offload": offload, "cuda_available": cuda_available}


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _ensure_config_file():
    if CONFIG_FILE.exists():
        return
    if EXAMPLE_FILE.exists():
        CONFIG_FILE.write_text(EXAMPLE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        CONFIG_FILE.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")


def load_config():
    _ensure_config_file()
    try:
        user_cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        user_cfg = {}
    return _deep_merge(DEFAULTS, user_cfg)


def save_config(cfg):
    merged = _deep_merge(DEFAULTS, cfg)
    CONFIG_FILE.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged
