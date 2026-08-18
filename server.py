"""MediaGenStudio server: FastAPI backend + static GUI for local Krea (image)
and MiniMax-H3 (video) generation.

Run with: python server.py
"""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config as config_module
import jobs
from engines import image_krea, video_h3

BASE_DIR = Path(__file__).parent

app = FastAPI(title="MediaGenStudio")
app.mount("/outputs", StaticFiles(directory=str(jobs.OUTPUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/api/status")
def status():
    info = {"torch_available": False, "cuda_available": False, "gpu_count": 0, "gpus": []}
    try:
        import torch
        info["torch_available"] = True
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
        for i in range(info["gpu_count"]):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "name": props.name,
                "vram_gb": round(props.total_memory / (1024 ** 3), 1),
            })
    except ImportError:
        pass
    hw = config_module.resolve_hardware(config_module.load_config())
    info["resolved"] = hw
    return info


@app.get("/api/config")
def get_config():
    return config_module.load_config()


@app.post("/api/config")
def set_config(cfg: dict):
    return config_module.save_config(cfg)


# ---------------------------------------------------------------------------
# Image generation (Krea / FLUX)
# ---------------------------------------------------------------------------

class ImageGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance_scale: float = 4.5
    seed: Optional[int] = None


@app.post("/api/image/generate")
def image_generate(req: ImageGenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is required")
    job = jobs.create_job("image", req.model_dump())
    return job


# ---------------------------------------------------------------------------
# Video generation (MiniMax-H3)
# ---------------------------------------------------------------------------

@app.post("/api/video/generate")
async def video_generate(
    prompt: str = Form(...),
    duration: int = Form(6),
    aspect_ratio: str = Form("16:9"),
    seed: Optional[int] = Form(None),
    backend: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    if not prompt.strip():
        raise HTTPException(400, "prompt is required")

    image_path = None
    if image is not None and image.filename:
        ext = Path(image.filename).suffix or ".png"
        image_path = jobs.UPLOAD_DIR / f"{jobs.uuid.uuid4().hex[:12]}{ext}"
        with open(image_path, "wb") as f:
            f.write(await image.read())

    params = {
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "seed": seed,
        "backend": backend,
        "image_path": str(image_path) if image_path else None,
    }
    job = jobs.create_job("video", params)
    return job


@app.get("/api/jobs")
def get_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    cancelled = jobs.cancel_job(job_id)
    return {"cancelled": cancelled}


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

def _run_image_job(job):
    params = job["params"]
    cfg = config_module.load_config()

    def progress_cb(step, total):
        jobs.update_job(job["id"], progress=int(step / total * 100), message=f"Step {step}/{total}")

    def log(msg):
        jobs.update_job(job["id"], message=msg)

    image = image_krea.generate(
        cfg,
        prompt=params["prompt"],
        negative_prompt=params.get("negative_prompt"),
        width=params.get("width", 1024),
        height=params.get("height", 1024),
        steps=params.get("steps", 28),
        guidance_scale=params.get("guidance_scale", 4.5),
        seed=params.get("seed"),
        progress_cb=progress_cb,
        log=log,
    )
    out_name = f"{job['id']}.png"
    image.save(jobs.OUTPUT_DIR / out_name)
    jobs.update_job(job["id"], status="done", progress=100, message="Done",
                     result={"type": "image", "url": f"/outputs/{out_name}"})


def _run_video_job(job):
    params = job["params"]
    cfg = config_module.load_config()
    backend = params.get("backend") or cfg["video_model"].get("backend", "api")

    def progress_cb(step, total):
        jobs.update_job(job["id"], progress=int(step / total * 100), message=f"{step}/{total}")

    def log(msg):
        jobs.update_job(job["id"], message=msg)

    if backend == "api":
        image_b64 = None
        if params.get("image_path"):
            image_b64 = video_h3.image_path_to_b64(params["image_path"])
        url = video_h3.generate_via_api(
            cfg,
            prompt=params["prompt"],
            image_b64=image_b64,
            duration=params.get("duration", 6),
            aspect_ratio=params.get("aspect_ratio", "16:9"),
            seed=params.get("seed"),
            log=log,
            progress_cb=progress_cb,
        )
        jobs.update_job(job["id"], status="done", progress=100, message="Done (MiniMax API)",
                         result={"type": "video", "url": url})
    else:
        from PIL import Image
        image = Image.open(params["image_path"]) if params.get("image_path") else None
        output = video_h3.generate_local(
            cfg, prompt=params["prompt"], image=image,
            duration=params.get("duration", 6), seed=params.get("seed"),
            log=log, progress_cb=progress_cb,
        )
        out_name = f"{job['id']}.mp4"
        out_path = jobs.OUTPUT_DIR / out_name
        import imageio
        frames = getattr(output, "frames", output)
        imageio.mimsave(str(out_path), frames[0] if isinstance(frames, (list, tuple)) and len(frames) == 1 else frames, fps=24)
        jobs.update_job(job["id"], status="done", progress=100, message="Done (local)",
                         result={"type": "video", "url": f"/outputs/{out_name}"})


jobs.register_worker("image", _run_image_job)
jobs.register_worker("video", _run_video_job)
jobs.start_worker()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
