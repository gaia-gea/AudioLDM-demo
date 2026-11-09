# Local AudioLDM Demo

Text description → AudioLDM → WAV clip. A Python interface built with Gradio and accessed through your local browser. No Conda or model training is required.

## Windows setup

On each x64 Windows computer, open PowerShell in the folder containing `start.ps1` and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

On a fresh copy, the launcher downloads a local Python 3.12 runtime and installs the required packages. It stores the environment under `.cache/runtime`. The first setup needs internet access and disk space for the runtime and packages. Later launches reuse the local environment.

The installer selects GPU mode for compatible NVIDIA hardware and uses CPU on other systems. To set up explicitly for CPU, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Device CPU
```

Use `-Device GPU` to request the CUDA 12.4 build for a compatible NVIDIA GPU. The application opens [the local interface](http://127.0.0.1:7860). To stop it, press Ctrl+C in the terminal.

## Choosing a model

Choose **Small v2**, **Medium Full**, or **Large Full** in the **Model** dropdown. Small v2 is the default. The next generation uses the selected model, and the benchmark report records its checkpoint.

The AudioLDM source clone contains the code. The demo downloads the pretrained weights for the selected model from Hugging Face and reuses its local cache. Medium Full requires about 2.6 GB and Large Full about 3.9 GB of downloads before cache reuse.

To download a model before presenting, run its command in PowerShell from the demo folder after setup:

```powershell
.\.cache\runtime\venv\Scripts\python.exe .\engine.py --model small --download-only
.\.cache\runtime\venv\Scripts\python.exe .\engine.py --model medium --download-only
.\.cache\runtime\venv\Scripts\python.exe .\engine.py --model large --download-only
```

With its weights cached, the demo can generate audio offline. For an older environment, use `.\.venv-audioldm\Scripts\python.exe` or run `install.ps1` to set up the managed environment.

On GPUs with 4 GB of VRAM or less, Large Full uses finer-grained CPU offloading in **GPU low memory** mode. This can increase generation time.

## Diagnostics and command-line testing

```powershell
.\.cache\runtime\venv\Scripts\python.exe .\engine.py --diagnose
.\.cache\runtime\venv\Scripts\python.exe .\engine.py --prompt "A dog barking in a quiet park" --steps 25 --duration 5
```

Add `--repeat 2` to measure the first generation and another generation with the model already loaded in the same process.

Each generation saves a WAV and a JSON report in `audioldm_demo/results`. The report includes GPU, RAM, software versions, seed, steps, timings, and peak CUDA memory allocated/reserved by PyTorch. These measurements do not represent total GPU memory usage. Available RAM in the diagnostics is a point-in-time reading, not a peak measurement.

## Running on other computers

Clone or copy the demo source and run `start.ps1` on each Windows computer. Let each computer prepare its own environment. You can copy `.cache/huggingface` to reuse downloaded weights. For a live presentation, complete setup and test one generation on the presentation computer beforehand.

On Linux with Python 3.12, create an environment with `python3.12 -m venv .venv-audioldm`, activate it, install PyTorch from the appropriate index, and then run `pip install -r requirements.txt`. Start the application with `python demo.py`.
