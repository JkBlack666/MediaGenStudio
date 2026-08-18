// MediaGenStudio frontend: tabs, form submission, job polling.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// --- Tabs ---------------------------------------------------------------
$$(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab-btn").forEach((b) => b.classList.remove("active"));
    $$(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "jobs") loadJobs();
  });
});

// --- GPU status badge -----------------------------------------------------
async function loadStatus() {
  const badge = $("#gpu-badge");
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const hw = data.resolved || {};
    if (data.cuda_available && data.gpus.length) {
      const g = data.gpus[0];
      badge.textContent = `${data.gpu_count}x GPU · ${g.name} (${g.vram_gb} GB)`;
      badge.className = "badge ok";
    } else if (data.torch_available) {
      badge.textContent = `No GPU detected · running on ${hw.device || "cpu"} (${hw.dtype_name || "float32"})`;
      badge.className = "badge warn";
    } else {
      badge.textContent = "torch not installed";
      badge.className = "badge warn";
    }
  } catch (e) {
    badge.textContent = "status unavailable";
  }
}
loadStatus();

// --- Job polling helper ---------------------------------------------------
function pollJob(jobId, onUpdate) {
  const interval = setInterval(async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) return;
    const job = await res.json();
    onUpdate(job);
    if (job.status === "done" || job.status === "error" || job.status === "cancelled") {
      clearInterval(interval);
    }
  }, 1500);
  return interval;
}

function renderResultBox(container, job) {
  const pct = job.progress || 0;
  let html = `<div>Status: <strong>${job.status}</strong> — ${job.message || ""}</div>
    <progress value="${pct}" max="100"></progress>`;
  if (job.status === "done" && job.result) {
    if (job.result.type === "image") {
      html += `<img src="${job.result.url}" alt="generated image" />`;
    } else if (job.result.type === "video") {
      html += `<video src="${job.result.url}" controls></video>`;
    }
  }
  if (job.status === "error") {
    html += `<div style="color:var(--err)">${job.message || "Unknown error"}</div>`;
  }
  container.innerHTML = html;
}

// --- Image form ------------------------------------------------------------
$("#image-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const submitBtn = form.querySelector("button[type=submit]");
  const resultBox = $("#image-result");
  const fd = new FormData(form);
  const payload = {
    prompt: fd.get("prompt"),
    negative_prompt: fd.get("negative_prompt") || null,
    width: Number(fd.get("width")),
    height: Number(fd.get("height")),
    steps: Number(fd.get("steps")),
    guidance_scale: Number(fd.get("guidance_scale")),
    seed: fd.get("seed") ? Number(fd.get("seed")) : null,
  };
  submitBtn.disabled = true;
  resultBox.innerHTML = "<div>Queued...</div>";
  try {
    const res = await fetch("/api/image/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    const job = await res.json();
    pollJob(job.id, (j) => {
      renderResultBox(resultBox, j);
      if (j.status !== "queued" && j.status !== "running") submitBtn.disabled = false;
    });
  } catch (err) {
    resultBox.innerHTML = `<div style="color:var(--err)">${err.message}</div>`;
    submitBtn.disabled = false;
  }
});

// --- Video form ------------------------------------------------------------
$("#video-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const submitBtn = form.querySelector("button[type=submit]");
  const resultBox = $("#video-result");
  const fd = new FormData(form);
  submitBtn.disabled = true;
  resultBox.innerHTML = "<div>Queued...</div>";
  try {
    const res = await fetch("/api/video/generate", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const job = await res.json();
    pollJob(job.id, (j) => {
      renderResultBox(resultBox, j);
      if (j.status !== "queued" && j.status !== "running") submitBtn.disabled = false;
    });
  } catch (err) {
    resultBox.innerHTML = `<div style="color:var(--err)">${err.message}</div>`;
    submitBtn.disabled = false;
  }
});

// --- Jobs list ---------------------------------------------------------------
async function loadJobs() {
  const list = $("#jobs-list");
  list.innerHTML = "Loading...";
  const res = await fetch("/api/jobs");
  const jobsData = await res.json();
  if (!jobsData.length) {
    list.innerHTML = "<p>No jobs yet.</p>";
    return;
  }
  list.innerHTML = jobsData.map((j) => {
    const date = new Date(j.created_at * 1000).toLocaleString();
    let preview = "";
    if (j.status === "done" && j.result) {
      preview = j.result.type === "image"
        ? `<img src="${j.result.url}" style="max-width:200px" />`
        : `<video src="${j.result.url}" style="max-width:200px" controls></video>`;
    }
    return `<div class="job-card">
      <div class="job-head">
        <span>${j.kind} · ${date}</span>
        <span class="status-pill status-${j.status}">${j.status}</span>
      </div>
      <div>${(j.params.prompt || "").slice(0, 140)}</div>
      ${preview}
    </div>`;
  }).join("");
}
$("#refresh-jobs").addEventListener("click", loadJobs);

// --- Settings ------------------------------------------------------------
async function loadSettings() {
  const res = await fetch("/api/config");
  const cfg = await res.json();
  const form = $("#settings-form");
  form.image_repo_id.value = cfg.image_model.local_path || cfg.image_model.repo_id;
  form.image_offload.value = cfg.image_model.offload;
  form.video_repo_id.value = cfg.video_model.local_path || cfg.video_model.repo_id;
  form.video_variant.value = cfg.video_model.variant;
  form.video_backend.value = cfg.video_model.backend;
  form.minimax_base_url.value = cfg.minimax_api.base_url;
  form.hf_token.value = cfg.hf_token || "";
}
loadSettings();

$("#settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = {
    hf_token: fd.get("hf_token") || "",
    image_model: { repo_id: fd.get("image_repo_id"), offload: fd.get("image_offload") },
    video_model: { repo_id: fd.get("video_repo_id"), variant: fd.get("video_variant"), backend: fd.get("video_backend") },
    minimax_api: { base_url: fd.get("minimax_base_url") },
  };
  const res = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const status = $("#settings-status");
  status.textContent = res.ok ? "Saved." : "Failed to save settings.";
  setTimeout(() => (status.textContent = ""), 3000);
});
