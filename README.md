# AudioLDM Text-to-Sound Demo

A local Python demo that generates sound from a text description. The interface provides four examples, an audio player, and a downloadable benchmark report.

## Start on Windows

Open PowerShell in this folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

On a fresh x64 Windows computer, the launcher downloads its own Python and installs the dependencies. Setup and the first model download require internet access. Subsequent launches reuse the local environment and model cache.

For an explicit installation, run `install.ps1`. Use `install.ps1 -Device CPU` or `install.ps1 -Device GPU` to select the PyTorch build.

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
- `install.ps1` prepares Python and dependencies. `start.ps1` opens the demo.

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

