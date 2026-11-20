"""AudioLDM local: same engine for the browser UI and command-line benchmark."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".cache" / "gradio"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import json
import gc
from dataclasses import dataclass
import platform
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import psutil
import soundfile as sf
import torch
from diffusers import AudioLDMPipeline, AudioLDM2Pipeline
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

@dataclass(frozen=True)
class ModelConfig:
    """Checkpoint settings shared by loading, generation, and the interface."""
    label: str
    repo_id: str
    revision: str
    pipeline: type = AudioLDMPipeline
    use_safetensors: bool = True
    guidance_scale: float = 2.5


DEFAULT_MODEL = "audioldm2"
MODELS = {
    "small": ModelConfig(
        "Small v2", "cvssp/audioldm-s-full-v2",
        "feeb3d14203495a4b6ac0893cbdedb2159b4819c",
    ),
    "medium": ModelConfig(
        "Medium Full", "cvssp/audioldm-m-full",
        "9d4998f93dfdafd050695fc82a4c8773ade7bd88",
    ),
    "large": ModelConfig(
        "Large Full", "cvssp/audioldm-l-full",
        "318f5a31753ef5c81552ccdd3eabcd46f32d99ae",
        use_safetensors=False,
    ),
    "audioldm2": ModelConfig(
        "AudioLDM 2", "cvssp/audioldm2",
        "c8e7e189d324425c05c4c2f81214041ef4107983",
        pipeline=AudioLDM2Pipeline, guidance_scale=3.5,
    ),
}
RESULTS = ROOT / "results"
MODEL_FILES = (
    "model_index.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "tokenizer/merges.txt", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "unet/config.json", "vae/config.json", "vocoder/config.json",
)
SAFETENSORS_WEIGHTS = (
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "vocoder/model.safetensors",
)
PYTORCH_WEIGHTS = (
    "text_encoder/pytorch_model.bin",
    "unet/diffusion_pytorch_model.bin",
    "vae/diffusion_pytorch_model.bin",
    "vocoder/pytorch_model.bin",
)


AUDIOLDM2_FILES = MODEL_FILES + SAFETENSORS_WEIGHTS + (
    "feature_extractor/preprocessor_config.json",
    "language_model/config.json", "language_model/model.safetensors",
    "projection_model/config.json", "projection_model/diffusion_pytorch_model.safetensors",
    "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
    "tokenizer/special_tokens_map.json", "tokenizer/tokenizer.json",
    "tokenizer_2/special_tokens_map.json", "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer.json", "tokenizer_2/tokenizer_config.json",
)


def prepare_model(model=DEFAULT_MODEL):
    """Fetch explicit Hub paths, avoiding Diffusers' Windows download filters.

    Check cache first, so a complete model can also be used without internet.
    Failed/partial downloads are resumed by huggingface_hub on the next attempt.
    """
    config = MODELS[model]
    repo_id, revision = config.repo_id, config.revision
    weights = SAFETENSORS_WEIGHTS if config.use_safetensors else PYTORCH_WEIGHTS
    required_files = AUDIOLDM2_FILES if model == "audioldm2" else MODEL_FILES + weights
    model_dir = None
    for filename in required_files:
        try:
            path = hf_hub_download(repo_id, filename, revision=revision, local_files_only=True)
        except LocalEntryNotFoundError:
            print(f"Downloading {repo_id}: {filename}", flush=True)
            try:
                path = hf_hub_download(repo_id, filename, revision=revision)
            except Exception as exc:
                raise RuntimeError(
                    f"Model download is incomplete. Could not download {filename} for {repo_id} from Hugging Face. "
                    "Check your internet connection and retry; completed files are kept. "
                    f"Details: {exc}"
                ) from exc
        if filename == "model_index.json":
            model_dir = Path(path).parent
    missing = [name for name in required_files if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Model cache is incomplete. Missing files: {', '.join(missing)}")
    return model_dir


def hardware():
    info = {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "ram_available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
    }
    if info["cuda_available"]:
        gpu = torch.cuda.get_device_properties(0)
        info.update(gpu=gpu.name, vram_gb=round(gpu.total_memory / 1024**3, 2))
    return info


def resolve_mode(mode):
    """Choose a processing mode once, using GPU capacity rather than free memory."""
    mode = {"GPU ahorro": "GPU low memory", "GPU directa": "GPU direct"}.get(mode, mode)
    if mode == "Auto":
        mode = "CPU"
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            mode = "GPU direct" if vram_gb >= 8 else "GPU low memory"
    if mode not in ("GPU low memory", "GPU direct", "CPU"):
        raise ValueError("Invalid processing mode.")
    if mode != "CPU" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot detect CUDA. Select CPU or run install.ps1.")
    return mode


def configure_memory(pipe, mode, model):
    """Use direct execution or the low-memory setup needed by a smaller GPU."""
    if mode != "GPU low memory":
        pipe.to("cpu" if mode == "CPU" else "cuda")
        return

    if model == "audioldm2":
        # CLAP calls get_text_features, bypassing the offload hook on forward.
        pipe._exclude_from_cpu_offload = ["text_encoder"]
        pipe.model_cpu_offload_seq = (
            "text_encoder_2->projection_model->language_model->unet->vae->vocoder"
        )
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if model in ("large", "audioldm2") and vram_gb <= 4:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()


class Generator:
    def __init__(self):
        self.pipe = None
        self.mode = None
        self.model = None

    def load(self, mode, model=DEFAULT_MODEL):
        mode = resolve_mode(mode)
        if model not in MODELS:
            raise ValueError("Invalid model selection.")
        config = MODELS[model]
        if self.pipe is not None and self.mode == mode and self.model == model:
            return 0.0
        if self.pipe is not None:
            # Hooks installed by accelerate must go away with the old pipeline.
            self.pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.mode = self.model = None
        start = time.perf_counter()
        # Full precision avoids non-finite audio observed on the GTX 1650.
        # Component offloading still limits VRAM in the low-memory mode.
        gtx16 = mode != "CPU" and "GTX 16" in torch.cuda.get_device_name(0)
        dtype = torch.float32 if mode == "CPU" or gtx16 else torch.float16
        model_dir = prepare_model(model)
        pipe = config.pipeline.from_pretrained(
            str(model_dir), torch_dtype=dtype, use_safetensors=config.use_safetensors,
            low_cpu_mem_usage=True, local_files_only=True,
        )
        configure_memory(pipe, mode, model)
        self.pipe, self.mode, self.model = pipe, mode, model
        return time.perf_counter() - start

    def generate(self, prompt, mode="Auto", duration=5, steps=25, seed=42,
                 progress_callback=None, *, model=DEFAULT_MODEL):
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Enter a sound description.")
        duration, steps, seed = float(duration), int(steps), int(seed)
        if not 2 <= duration <= 10 or not 10 <= steps <= 200:
            raise ValueError("Use 2 to 10 seconds and 10 to 200 steps.")
        if not 0 <= seed <= 2147483647:
            raise ValueError("Seed must be between 0 and 2147483647.")
        total_start = time.perf_counter()
        load_s = self.load(mode, model)
        mode = self.mode
        gpu = mode != "CPU"
        if gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        rng = torch.Generator(device="cpu" if not gpu else "cuda").manual_seed(seed)

        def on_step(step, timestep, latents):
            if progress_callback:
                progress_callback(min((step + 1) / steps, 1))

        config = MODELS[model]
        guidance_scale = config.guidance_scale
        with torch.inference_mode():
            audio = self.pipe(
                prompt, audio_length_in_s=duration, num_inference_steps=steps,
                guidance_scale=guidance_scale, num_waveforms_per_prompt=1, generator=rng,
                callback=on_step, callback_steps=1,
            ).audios[0]
        if gpu:
            torch.cuda.synchronize()
        inference_s = time.perf_counter() - start
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-7:
            raise RuntimeError("The model produced silent or invalid audio. No result was saved.")
        RESULTS.mkdir(parents=True, exist_ok=True)
        name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        wav = RESULTS / f"{name}.wav"
        sf.write(wav, audio, self.pipe.vocoder.config.sampling_rate, subtype="PCM_16")
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": config.repo_id, "model_revision": config.revision,
            "prompt": prompt, "mode": mode,
            "duration_s": duration, "steps": steps, "seed": seed,
            "guidance_scale": guidance_scale,
            "load_s": round(load_s, 2), "inference_s": round(inference_s, 2),
            "total_s": round(time.perf_counter() - total_start, 2),
            "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if gpu else None,
            "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3) if gpu else None,
            "hardware": hardware(), "wav": wav.name,
        }
        report_path = wav.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(wav), str(report_path), report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AudioLDM local diagnostics and benchmark")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--prompt", default="A dog barking")
    parser.add_argument("--mode", choices=["Auto", "GPU low memory", "GPU direct", "CPU", "GPU ahorro", "GPU directa"], default="Auto")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1, help="Runs reusing the loaded model (1 to 10)")
    parser.add_argument("--download-only", action="store_true", help="Check and complete the model cache without generating audio")
    args = parser.parse_args()
    if args.download_only:
        print(prepare_model(args.model))
    elif args.diagnose:
        print(json.dumps(hardware(), indent=2, ensure_ascii=False))
    else:
        if not 1 <= args.repeat <= 10:
            parser.error("--repeat must be between 1 and 10")
        engine = Generator()
        for _ in range(args.repeat):
            _, _, report = engine.generate(
                args.prompt, args.mode, args.duration, args.steps, args.seed, model=args.model,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
