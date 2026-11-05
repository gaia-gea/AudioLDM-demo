# Local AudioLDM Demo

Text description → AudioLDM Small v2 → WAV clip. A Python interface built with Gradio and accessed through your local browser. No Conda or model training is required.

## Windows with NVIDIA

Requirements: 64-bit Python 3.12 and an NVIDIA driver compatible with CUDA 12.4. You do not need to install the CUDA Toolkit separately to use these PyTorch wheels.

From the project root, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\audioldm_demo\install.ps1
powershell -ExecutionPolicy Bypass -File .\audioldm_demo\start.ps1
```

The application opens [the local interface](http://127.0.0.1:7860). To stop it, press Ctrl+C in its terminal.

The initial installation and first model load download several GB. Additional disk space is needed for the environment and model weights. The weights are stored in `audioldm_demo/.cache/huggingface` and reused on subsequent loads. No public link is created.

## First test

1. Select **GPU low memory**, a duration of 5 seconds, 25 steps, and seed 42.
2. Generate `A dog barking in a quiet park`.
3. Repeat without changing the mode. The first run includes downloading/loading the model; the second measures generation with the model already loaded.
4. Listen to the WAV and download the JSON report. Successful execution alone does not establish perceptual quality.
5. On GPUs with more memory, compare **GPU direct**. If GPU memory runs out, switch back to **GPU low memory**, shorten the clip, and close other applications using the GPU.

GPU low memory also uses system RAM. A GPU with 4 GB of VRAM is not guaranteed to support every setting. Reducing steps reduces computation; shortening the clip can also reduce memory usage. Execution times depend on the computer.

GTX 16-series cards use float32 because half precision produced invalid audio on the tested GTX 1650. Other CUDA GPUs use float16 by default.

## Diagnostics and command-line testing

```powershell
.\.venv-audioldm\Scripts\python.exe .\audioldm_demo\engine.py --diagnose
.\.venv-audioldm\Scripts\python.exe .\audioldm_demo\engine.py --prompt "A dog barking in a quiet park" --steps 25 --duration 5
```

Add `--repeat 2` to measure the first generation and another generation with the model already loaded in the same process.

Each generation saves a WAV and a JSON report in `audioldm_demo/results`. The report includes GPU, RAM, software versions, seed, steps, timings, and peak CUDA memory allocated/reserved by PyTorch. These measurements do not represent total GPU memory usage. Available RAM in the diagnostics is a point-in-time reading, not a peak measurement.

## Running on other computers

Copy `audioldm_demo` into a working folder and run `install.ps1` there. Create a new virtual environment on each computer. You can copy the model cache to avoid downloading the weights again. Keep the description, seed, duration, steps, and processing mode identical when comparing timings. The same seed does not guarantee identical audio across all hardware and software versions.

The GPU installation targets NVIDIA GPUs compatible with CUDA 12.4. It does not automatically cover AMD/Intel GPUs, Apple devices, or every newer NVIDIA GPU. For other architectures, select a compatible PyTorch wheel using the official documentation. CPU execution is selected explicitly; a GPU failure does not silently switch execution to CPU.

For a CPU-only installation:

```powershell
powershell -ExecutionPolicy Bypass -File .\audioldm_demo\install.ps1 -Device CPU
```

Select CPU in the interface or add `--mode CPU` to the benchmark command. Generation may take considerably longer.

On Linux with Python 3.12, create an environment with `python3.12 -m venv .venv-audioldm`, activate it, install PyTorch from the appropriate index, and then run `pip install -r audioldm_demo/requirements.txt`. Start the application with `python audioldm_demo/demo.py`.

## References

- [Model and CC BY-NC-SA 4.0 license](https://huggingface.co/cvssp/audioldm-s-full-v2)
- [Pipeline documentation](https://huggingface.co/docs/diffusers/v0.30.3/en/api/pipelines/audioldm)
- [PyTorch wheels](https://pytorch.org/get-started/previous-versions/)

A version of Diffusers that includes this pipeline is pinned in the requirements. See [VALIDATION.md](VALIDATION.md) for measurements from the tested computer.

## Incomplete model cache on Windows

If a `config.json` error appears, restart the application after updating the code. The loader checks all 13 required files and downloads only missing files, retaining existing weights. It uses explicit Hugging Face paths to avoid the Diffusers download filters that mix Windows path separators. The model is pinned to revision `feeb3d14203495a4b6ac0893cbdedb2159b4819c`.

You can also complete the download without opening the interface:

```powershell
.\.venv-audioldm\Scripts\python.exe .\audioldm_demo\engine.py --download-only
```

If the connection to Hugging Face is interrupted, restore the connection and retry. Completed files are retained.
