# Gemma Studio

I built Gemma Studio to solve a simple problem: I wanted a unified workspace for text chat, visual analysis, image generation, and photo editing without giving up control over my data. 

Most AI suites rely heavily on cloud APIs, which means subscription fees, network lag, and sending personal photos or prompts to external servers. I wanted to see how far I could push a 100% on-device experience using consumer hardware. By combining Google's open-weights Gemma 3 models with Latent Consistency Model (LCM) diffusion pipelines, Gemma Studio runs entirely offline on my local machine.

## Why Local & Private?

Every prompt, uploaded photo, and generated image stays strictly in local memory. There are no API keys, no network calls to external servers, and zero cloud telemetry. This makes it ideal for handling sensitive documents, personal pictures, or creative work that you don't want floating around on someone else's server.

## Features

* **Gemma Chat Playground:** Fast multi-turn text generation and coding help powered by `gemma3:1b` running locally on CPU.
* **Photo Detective:** Visual analysis and document inspection using `gemma3:4b` to answer questions about local images without uploading them anywhere.
* **Dream Factory:** A conversational image generator where Gemma 3 refines my prompt ideas through chat, then passes them to a local Stable Diffusion pipeline accelerated with LCM LoRA (rendering crisp images in just 8 to 12 steps).
* **Magic Canvas:** Targeted photo editing using natural language commands (like *"change the mug color to neon blue"*), combining vision understanding with image-to-image diffusion.

## How It's Built

* **Language & Vision Models:** Google Gemma 3 (`gemma3:1b` & `gemma3:4b` running via Ollama)
* **Image Synthesis:** Hugging Face Diffusers (Stable Diffusion v1.5 + LCM LoRA)
* **Interface:** Streamlit
* **Runtime Environment:** Python & PyTorch

## Getting Started

1. Pull the required Gemma 3 models in Ollama:
   ollama pull gemma3:1b
   ollama pull gemma3:4b

2. Install Python dependencies:
   pip install streamlit torch diffusers transformers peft accelerate pillow requests

3. Run the app:
   streamlit run app.py

## License

Distributed under the MIT License.
