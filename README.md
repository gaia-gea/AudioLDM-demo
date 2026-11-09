# Local AudioLDM Demo

Text description → AudioLDM → WAV clip. A Python interface built with Gradio and accessed through your local browser. No Conda or model training is required.

## Windows with NVIDIA

Requirements: 64-bit Python 3.12 and an NVIDIA driver compatible with CUDA 12.4. You do not need to install the CUDA Toolkit separately to use these PyTorch wheels.

From the folder containing install.ps1 and start.ps1, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

The application opens [the local interface](http://127.0.0.1:7860). To stop it, press Ctrl+C in its terminal.


## Choosing a model

Choose **Small v2**, **Medium Full**, or **Large Full** in the **Model** dropdown. Small v2 is the default. The next generation uses the selected model, and the benchmark report records its checkpoint.

The AudioLDM source clone contains the code. The demo downloads the pretrained weights for the selected model from Hugging Face and reuses its local cache. Medium Full requires about 2.6 GB and Large Full about 3.9 GB of downloads before cache reuse.

To download Medium Full or Large Full ahead of time, run this from the folder containing `engine.py` in Git Bash:

```bash
./.venv-audioldm/Scripts/python.exe engine.py --model medium --download-only
./.venv-audioldm/Scripts/python.exe engine.py --model large --download-only
```

If your virtual environment is in the parent directory, use `../.venv-audioldm/Scripts/python.exe` instead. Start the demo and choose **GPU low memory** for the first test. On GPUs with 4 GB of VRAM or less, Large Full moves smaller parts of the model between RAM and GPU to reduce VRAM use. This can increase generation time.


## Diagnostics and command-line testing

```powershell
.\.venv-audioldm\Scripts\python.exe .\engine.py --diagnose
.\.venv-audioldm\Scripts\python.exe .\engine.py --prompt "A dog barking in a quiet park" --steps 25 --duration 5
```

Add `--repeat 2` to measure the first generation and another generation with the model already loaded in the same process.

Each generation saves a WAV and a JSON report in `audioldm_demo/results`. The report includes GPU, RAM, software versions, seed, steps, timings, and peak CUDA memory allocated/reserved by PyTorch. These measurements do not represent total GPU memory usage. Available RAM in the diagnostics is a point-in-time reading, not a peak measurement.

## Running on other computers

Copy `audioldm_demo` into a working folder and run `install.ps1` there. Create a new virtual environment on each computer. You can copy the model cache to avoid downloading the weights again.

For a CPU-only installation:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Device CPU
```

Select CPU in the interface or add `--mode CPU` to the benchmark command. Generation may take considerably longer.

On Linux with Python 3.12, create an environment with `python3.12 -m venv .venv-audioldm`, activate it, install PyTorch from the appropriate index, and then run `pip install -r audioldm_demo/requirements.txt`. Start the application with `python audioldm_demo/demo.py`.


