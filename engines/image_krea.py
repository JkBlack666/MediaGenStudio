"""Local inference engine for Krea's text-to-image models ("Krea").

Supports two distinct model families:
  - Krea 2 ("K2") - krea/Krea-2-Turbo or krea/Krea-2-Raw, Krea.ai's own 13B
    architecture with native `Krea2Pipeline` support in diffusers>=0.39.
    This is the actual "Krea 2" model and the default here.
  - FLUX.1 Krea [dev] - black-forest-labs/FLUX.1-Krea-dev, a FLUX fine-tune
    collaboration between Black Forest Labs and Krea, using `FluxPipeline`.
    Kept as an alternative repo_id you can set in Settings.

Torch/diffusers are imported lazily so the server can still start (and show a
clear error in the GUI) on machines without those installed/CUDA available.
"""
import threading

import config as config_module

_pipeline = None
_pipeline_lock = threading.Lock()
_loaded_key = None


def _is_krea2(model_path):
    return "krea-2" in model_path.lower()


def _is_turbo(model_path):
    return "turbo" in model_path.lower()


def _pipeline_key(full_cfg):
    img = full_cfg["image_model"]
    hw = config_module.resolve_hardware(full_cfg)
    return (img.get("local_path") or img.get("repo_id"), hw["offload"], hw["device"], hw["dtype_name"])


def is_loaded(full_cfg):
    return _pipeline is not None and _loaded_key == _pipeline_key(full_cfg)


def _friendly_error(exc, model_path):
    msg = str(exc)
    lowered = msg.lower()
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
    if "404" in msg or "not found" in lowered:
        return RuntimeError(
            f"Model repo '{model_path}' was not found. Check the repo id / local path "
            "in Settings."
        )
    return exc


def _load_pipeline(full_cfg, log=None):
    global _pipeline, _loaded_key
    key = _pipeline_key(full_cfg)
    with _pipeline_lock:
        if _pipeline is not None and _loaded_key == key:
            return _pipeline
        try:
            import torch
            if _is_krea2(full_cfg["image_model"].get("local_path") or full_cfg["image_model"]["repo_id"]):
                from diffusers import Krea2Pipeline as PipelineClass
            else:
                from diffusers import FluxPipeline as PipelineClass
        except ImportError as exc:
            raise RuntimeError(
                "torch/diffusers are not installed. Run "
                "'pip install -r requirements.txt' in the project's virtualenv."
            ) from exc

        img = full_cfg["image_model"]
        hw = config_module.resolve_hardware(full_cfg)
        model_path = img.get("local_path") or img["repo_id"]
        token = full_cfg.get("hf_token") or None
        dtype = getattr(torch, hw["dtype_name"], torch.float32)

        if log:
            log(f"Loading '{model_path}' ({PipelineClass.__name__}) on {hw['device']} "
                f"({hw['dtype_name']}). First load downloads several GB+ and can take a while.")
        try:
            pipe = PipelineClass.from_pretrained(model_path, torch_dtype=dtype, token=token)
        except Exception as exc:
            raise _friendly_error(exc, model_path) from exc

        if hw["offload"] == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif hw["offload"] == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(hw["device"])

        _pipeline = pipe
        _loaded_key = key
        return pipe


def generate(full_cfg, prompt, negative_prompt=None, width=1024, height=1024,
             steps=None, guidance_scale=None, seed=None, progress_cb=None, log=None):
    pipe = _load_pipeline(full_cfg, log=log)  # raises a clear RuntimeError if torch/diffusers/auth are missing
    import torch

    img = full_cfg["image_model"]
    model_path = img.get("local_path") or img["repo_id"]
    if steps is None or guidance_scale is None:
        # Sensible per-checkpoint defaults (matching the official model cards).
        if _is_krea2(model_path) and _is_turbo(model_path):
            default_steps, default_guidance = 8, 0.0
        elif _is_krea2(model_path):
            default_steps, default_guidance = 52, 3.5
        else:
            default_steps, default_guidance = 28, 4.5
        steps = default_steps if steps is None else steps
        guidance_scale = default_guidance if guidance_scale is None else guidance_scale

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator = generator.manual_seed(int(seed))

    def _callback(pipeline, step, timestep, kwargs):
        if progress_cb:
            progress_cb(step + 1, steps)
        return kwargs

    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
        callback_on_step_end=_callback,
    )
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    result = pipe(**kwargs)
    return result.images[0]
