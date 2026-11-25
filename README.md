# AudioLDM Text-to-Sound Demo

A local Python demo that generates sound from a text description. The interface provides four examples, an audio player, and a downloadable benchmark report.

## Start on Windows

Open PowerShell in this folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

On a fresh x64 Windows computer, the launcher downloads its own Python and installs the dependencies. Setup and the first model download require internet access. Subsequent launches reuse the local environment and model cache.

For an explicit installation, run `install.ps1`. Use `install.ps1 -Device CPU` or `install.ps1 -Device GPU` to select the PyTorch build.

## Start on Linux

On x86_64 Linux with glibc, such as Ubuntu or Debian, open a terminal in this folder and run:

```bash
bash start.sh
```

On first launch, the script downloads its own Python 3.12 and installs the dependencies under `.cache/runtime/linux-x86_64`. It selects a compatible NVIDIA GPU automatically and uses CPU otherwise. Setup requires internet access. Commands `curl`, `tar`, `sha256sum`, `flock`, `awk`, and `getconf` must be available.

To install explicitly for CPU or GPU:

```bash
bash install.sh cpu
bash install.sh gpu
```

Choose one command. GPU installation requires an NVIDIA driver compatible with CUDA 12.4. In auto mode, the installer uses a conservative driver and GPU architecture check, then verifies CUDA with a real tensor operation. It falls back to CPU if that check fails.

The Linux runtime has its own directory, separate from the Windows runtime. Each computer should prepare its own environment. Models remain in the shared `.cache/huggingface` folder.

To download the selected model ahead of time:

```bash
./.cache/runtime/linux-x86_64/venv/bin/python engine.py --model audioldm2 --download-only
```

To use a different local port, run `bash start.sh --port 7861`.

## Use the demo

1. Choose a model and enter a sound description or select an example.
2. Leave **Processing mode** on **Auto**.
3. Set the duration, generation steps, and seed.
4. Click **Generate sound**, listen to the result, and optionally download its report.

The models are **Small v2**, **Medium Full**, **Large Full**, and **AudioLDM 2**. AudioLDM 2 is selected by default. Each model keeps its own cached weights.

Auto selects CPU when CUDA is unavailable, direct GPU execution on GPUs with at least 8 GB of VRAM, and memory-saving GPU execution on smaller GPUs. The manual processing modes remain available. The summary and JSON report show the actual mode used.

## Code structure

- `demo.py` builds the Gradio interface and connects the button to the generator.
- `engine.py` contains the model configurations, download logic, pipeline loading, generation, and WAV/JSON saving.
- `install.ps1` and `start.ps1` prepare and launch the Windows demo.
- `install.sh` and `start.sh` provide the same workflow on Linux.

In `engine.py`, `ModelConfig` gives each checkpoint named settings. `prepare_model()` completes its cache. `Generator.load()` loads or reuses the pipeline, and `Generator.generate()` produces the audio and report. `resolve_mode()` and `configure_memory()` contain the hardware-specific setup.

## Command-line usage

With the managed environment:

```powershell
.\.cache\runtime\venv\Scripts\python.exe .\engine.py --diagnose
.\.cache\runtime\venv\Scripts\python.exe .\engine.py --model audioldm2 --download-only
.\.cache\runtime\venv\Scripts\python.exe .\engine.py --model audioldm2 --prompt "A dog barking" --duration 5 --steps 25 --seed 42
```

For an older installation, use `.\.venv-audioldm\Scripts\python.exe`. Model keys are `small`, `medium`, `large`, and `audioldm2`. `--mode` accepts `Auto`, `GPU direct`, `GPU low memory`, or `CPU`. Add `--repeat 2` to compare initial and subsequent runs with one loaded pipeline.

## Share the project

Commit the source files, scripts, `requirements.txt`, and `README.md`. The `.gitignore` excludes model caches, runtime environments, and generated results. Each computer prepares its own environment. To reuse downloaded weights, copy `.cache/huggingface` separately.

Model weights use the [CC BY-NC-SA 4.0 license](https://huggingface.co/cvssp/audioldm2). The pipeline implementation comes from [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/v0.30.3/en/api/pipelines/audioldm2).

