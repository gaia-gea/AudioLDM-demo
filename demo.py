"""Run with: python demo.py. Opens a local browser interface."""
from engine import DEFAULT_MODEL, MODELS, Generator, hardware
import argparse
import json
import traceback
import gradio as gr
import torch

engine = Generator()


def generate(prompt, mode, duration, steps, seed, model, progress=gr.Progress()):
    progress(0, desc=f"Preparing {MODELS[model].label} (downloads missing files on first use)…")
    try:
        wav, report_path, report = engine.generate(
            prompt, mode, duration, steps, seed, model=model,
            progress_callback=lambda value: progress(value, desc="Generating sound…"),
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise gr.Error("Not enough GPU memory. Select GPU low memory, shorten the clip and close other applications using the GPU.") from None
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"Could not generate audio: {exc}") from exc
    summary = (
        f"**Model:** {MODELS[model].label} · "
        f"**Processing:** {report['mode']} · "
        f"**Generation:** {report['inference_s']} s · "
        f"**Model loading/downloading:** {report['load_s']} s · "
        f"**Peak VRAM allocated by PyTorch:** {report['peak_vram_allocated_gb']} GB\n\n"
        "The loaded model is reused until you change the model or processing mode. Listen to the result to assess quality."
    )
    return wav, summary, report_path


def build_demo():
    info = hardware()
    with gr.Blocks(title="AudioLDM · Text to Sound") as demo:
        gr.Markdown("# Text to Sound\nDescribe a sound, generate a clip and listen to the result.")
        prompt = gr.Textbox(label="Sound description", value="A dog barking", lines=3,
                            info="Start with concrete descriptions in English. This generates sounds rather than reading text aloud.")
        gr.Examples(examples=[["A metal bell ringing repeatedly"],
                              ["A crowd applauding indoors"],
                              ["A dog barking"],
                              ["Steady heavy rain falling on a roof"]], inputs=prompt)
        model = gr.Dropdown(
            choices=[(config.label, key) for key, config in MODELS.items()],
            value=DEFAULT_MODEL,
            label="Model",
            info="The selected model downloads its weights on first use.",
        )
        mode = gr.Dropdown(choices=["Auto", "GPU low memory", "GPU direct", "CPU"],
                           value="Auto", label="Processing mode",
                           info="Auto selects the processing mode for this computer. You can also select a mode manually.")
        with gr.Row():
            duration = gr.Slider(2, 10, value=5, step=0.5, label="Sound duration (seconds)")
            steps = gr.Slider(10, 200, value=25, step=5, label="Generation steps")
            seed = gr.Number(value=42, minimum=0, maximum=2147483647, precision=0, label="Seed")
        button = gr.Button("Generate sound", variant="primary")
        audio = gr.Audio(label="Result", type="filepath", interactive=False)
        summary = gr.Markdown()
        report = gr.File(label="Download benchmark report", interactive=False)
        with gr.Accordion("Hardware diagnostics", open=False):
            gr.Code(json.dumps(info, indent=2, ensure_ascii=False), language="json")
        button.click(generate, inputs=[prompt, mode, duration, steps, seed, model], outputs=[audio, summary, report],
                     concurrency_limit=1, concurrency_id="model")
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_demo().queue(max_size=4).launch(server_name="127.0.0.1", server_port=args.port,
                                         inbrowser=True, share=False)
