import os
import tempfile
import streamlit as st
import ollama
from PIL import Image
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# -------------------------------------------------------------------
# CPU OPTIMIZATION SETUP
# -------------------------------------------------------------------
if not torch.cuda.is_available():
    torch.set_num_threads(max(1, os.cpu_count() // 2))

# Page Configuration
st.set_page_config(
    page_title="Gemma Fun Studio 🚀", 
    page_icon="🌌", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# CUSTOM NIGHT SKY STYLING (CSS)
# -------------------------------------------------------------------
st.markdown("""
<style>
    /* Night Sky Deep Space Gradient */
    .stApp {
        background: linear-gradient(180deg, #090A0F 0%, #111827 50%, #1F2937 100%);
        color: #F3F4F6;
    }
    
    /* Playful Title */
    .main-title {
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(45deg, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    /* Subtitle Tagline */
    .sub-title {
        font-size: 1.1rem;
        color: #9CA3AF;
        font-weight: 500;
        margin-bottom: 25px;
    }

    /* Card Containers */
    div[data-testid="stExpander"], div.stCard {
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        border: 1px solid #374151;
        background-color: #1F2937;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        color: white;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }

    /* Fun Button Styling */
    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(90deg, #8B5CF6 0%, #EC4899 100%);
        color: white !important;
        font-weight: bold;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# HEADER SECTION
# -------------------------------------------------------------------
st.markdown('<p class="main-title">✨ Gemma Local AI Studio ✨</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">🌌 Fast, 100% Private, On-Device Creative Playground</p>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("🎮 Arcade Controls")
mode = st.sidebar.radio(
    "Choose Your Magic Mode:",
    [
        "💬 Gemma Chat Playground",
        "🔍 Photo Detective (Vision)",
        "🎨 Dream Factory (Image Gen)",
        "✏️ Magic Canvas (Photo Editor)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.title("⚡ Engine Turbo Boost")
selected_model = st.sidebar.selectbox(
    "Gemma Engine Variant:",
    ["gemma3:1b", "gemma3:4b"],
    help="gemma3:1b runs ultra-fast (~30+ tokens/sec) on laptop CPUs. gemma3:4b offers deeper reasoning."
)

st.sidebar.markdown("---")
st.sidebar.info("🔒 Zero Cloud Data | 100% Offline Active")

if st.sidebar.button("🧹 Reset Playground Memory"):
    st.session_state.text_messages = []
    st.session_state.vision_messages = []
    st.session_state.gen_messages = []
    st.session_state.edit_messages = []
    st.rerun()

# STANDARD NEGATIVE PROMPT TO ELIMINATE BLUR & ARTIFACTS
NEGATIVE_PROMPT = "blurry, low resolution, low quality, distorted, ugly, out of focus, duplicate, overexposed, noise"

# -------------------------------------------------------------------
# PIPELINE CACHING
# -------------------------------------------------------------------
@st.cache_resource
def get_txt2img_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=dtype
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe.to(device)

@st.cache_resource
def get_img2img_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=dtype
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe.to(device)


# -------------------------------------------------------------------
# MODE 1: CHAT PLAYGROUND
# -------------------------------------------------------------------
if mode == "💬 Gemma Chat Playground":
    # Whimsical Soft Pastel Purple Subheader
    st.markdown(
        '<h3 style="color: #C084FC; font-family: \'Comic Sans MS\', \'Trebuchet MS\', sans-serif; '
        'text-shadow: 2px 2px 8px rgba(192, 132, 252, 0.4);">💬 Chat & Brainstorming Lounge</h3>', 
        unsafe_allow_html=True
    )

    if "text_messages" not in st.session_state:
        st.session_state.text_messages = []

    for msg in st.session_state.text_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me anything... (e.g. 'Tell me a joke about AI!')"):
        st.session_state.text_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"⚡ Gemma ({selected_model}) is cooking up a response..."):
                try:
                    res = ollama.chat(model=selected_model, messages=st.session_state.text_messages)
                    reply = res['message']['content']
                    st.markdown(reply)
                    st.session_state.text_messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Oops! Something broke: {str(e)}")


# -------------------------------------------------------------------
# MODE 2: PHOTO DETECTIVE
# -------------------------------------------------------------------
elif mode == "🔍 Photo Detective (Vision)":
    st.markdown(
        '<h3 style="color: #C084FC; font-family: \'Comic Sans MS\', \'Trebuchet MS\', sans-serif; '
        'text-shadow: 2px 2px 8px rgba(192, 132, 252, 0.4);">🔍 Visual Inspector & Question Answering</h3>', 
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader("📂 Drop an image here to investigate:", type=["png", "jpg", "jpeg"])

    if "vision_messages" not in st.session_state:
        st.session_state.vision_messages = []

    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="🎯 Target Image", width=380)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        for msg in st.session_state.vision_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if vision_prompt := st.chat_input("What do you want to inspect in this photo?"):
            st.session_state.vision_messages.append({"role": "user", "content": vision_prompt})
            with st.chat_message("user"):
                st.markdown(vision_prompt)

            with st.chat_message("assistant"):
                with st.spinner("🕵️ Gemma Vision is inspecting every pixel..."):
                    try:
                        messages_payload = []
                        for idx, m in enumerate(st.session_state.vision_messages):
                            if idx == 0 and m["role"] == "user":
                                messages_payload.append({"role": m["role"], "content": m["content"], "images": [tmp_path]})
                            else:
                                messages_payload.append({"role": m["role"], "content": m["content"]})

                        res = ollama.chat(model="gemma3:4b", messages=messages_payload)
                        reply = res['message']['content']
                        st.markdown(reply)
                        st.session_state.vision_messages.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"Vision Error: {str(e)}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)


# -------------------------------------------------------------------
# MODE 3: DREAM FACTORY
# -------------------------------------------------------------------
elif mode == "🎨 Dream Factory (Image Gen)":
    st.markdown(
        '<h3 style="color: #C084FC; font-family: \'Comic Sans MS\', \'Trebuchet MS\', sans-serif; '
        'text-shadow: 2px 2px 8px rgba(192, 132, 252, 0.4);">🎨 Turn Ideas Into Visuals (Iterative Creation)</h3>', 
        unsafe_allow_html=True
    )

    if "gen_messages" not in st.session_state:
        st.session_state.gen_messages = []

    for msg in st.session_state.gen_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg:
                st.image(msg["image"], use_container_width=True)

    if gen_input := st.chat_input("Describe what to draw or follow up (e.g. 'A cyber-punk cat in Tokyo'):"):
        st.session_state.gen_messages.append({"role": "user", "content": gen_input})
        with st.chat_message("user"):
            st.markdown(gen_input)

        with st.chat_message("assistant"):
            final_prompt = f"{gen_input}, high quality, crisp details, sharp focus, 8k resolution"
            
            user_prompts = [m["content"] for m in st.session_state.gen_messages if m["role"] == "user"]
            if len(user_prompts) > 1:
                with st.spinner("🧠 Gemma is weaving your follow-up ideas into the prompt..."):
                    context_query = f"Synthesize these cumulative image requests into a single clear prompt: '{' -> '.join(user_prompts)}'."
                    res = ollama.chat(model=selected_model, messages=[{'role': 'user', 'content': context_query}])
                    final_prompt = f"{res['message']['content']}, high quality, sharp focus, detailed photography"
                    st.info(f"✨ **Synthesized Prompt:** {final_prompt}")

            with st.spinner("🎨 Painting your imagination (8-Step LCM Pass)..."):
                try:
                    pipe = get_txt2img_pipe()
                    output_image = pipe(
                        prompt=final_prompt, 
                        negative_prompt=NEGATIVE_PROMPT,
                        num_inference_steps=8, 
                        guidance_scale=1.0
                    ).images[0]
                    
                    st.image(output_image, caption=f"✨ Rendered Art", use_container_width=True)
                    st.session_state.gen_messages.append({
                        "role": "assistant", 
                        "content": f"Here is your creation for: *{gen_input}*", 
                        "image": output_image
                    })
                except Exception as e:
                    st.error(f"Creation Error: {str(e)}")


# -------------------------------------------------------------------
# MODE 4: MAGIC CANVAS
# -------------------------------------------------------------------
elif mode == "✏️ Magic Canvas (Photo Editor)":
    st.markdown(
        '<h3 style="color: #C084FC; font-family: \'Comic Sans MS\', \'Trebuchet MS\', sans-serif; '
        'text-shadow: 2px 2px 8px rgba(192, 132, 252, 0.4);">✏️ Iterative Photo Re-Imagining & Color Magic</h3>', 
        unsafe_allow_html=True
    )
    
    uploaded_base = st.file_uploader("📂 Upload Base Canvas:", type=["png", "jpg", "jpeg"])
    
    col1, col2 = st.columns(2)
    with col1:
        strength = st.slider(
            "🎛️ Transformation Power:", 
            0.3, 0.85, 0.65, 
            help="Set around 0.65 to easily change object colors!"
        )
    with col2:
        steps = st.slider(
            "💎 Quality Enhancement Steps:", 
            6, 16, 12, 
            help="Higher steps create crisp, non-blurry results."
        )

    if "edit_messages" not in st.session_state:
        st.session_state.edit_messages = []

    if uploaded_base:
        init_img = Image.open(uploaded_base).convert("RGB").resize((512, 512))
        st.image(init_img, caption="🖼️ Original Base Canvas (512x512)", width=320)

        for msg in st.session_state.edit_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "image" in msg:
                    st.image(msg["image"], use_container_width=True)

        if edit_instruction := st.chat_input("What change should we make? (e.g. 'Change the mug color to neon red'):"):
            st.session_state.edit_messages.append({"role": "user", "content": edit_instruction})
            with st.chat_message("user"):
                st.markdown(edit_instruction)

            with st.chat_message("assistant"):
                previous_images = [m["image"] for m in st.session_state.edit_messages if "image" in m]
                current_canvas = previous_images[-1] if previous_images else init_img

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    current_canvas.save(tmp.name)
                    tmp_path = tmp.name

                try:
                    with st.spinner("🎯 Gemma Vision mapping target elements..."):
                        gemma_prompt = (
                            f"Analyze this image and describe the entire scene, but explicitly declare this target modification: "
                            f"'{edit_instruction}'. Keep all unmentioned background elements unchanged."
                        )
                        
                        res = ollama.chat(model="gemma3:4b", messages=[{'role': 'user', 'content': gemma_prompt, 'images': [tmp_path]}])
                        gemma_context = res['message']['content']
                        
                        final_edit_prompt = f"a photo of {edit_instruction}, {gemma_context}, highly detailed, sharp focus, 8k"
                        st.info(f"✨ **Editing Target:** {final_edit_prompt}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                with st.spinner("🪄 Applying magic transformations..."):
                    try:
                        pipe = get_img2img_pipe()
                        result_img = pipe(
                            prompt=final_edit_prompt, 
                            negative_prompt=NEGATIVE_PROMPT,
                            image=current_canvas, 
                            strength=strength, 
                            num_inference_steps=steps,
                            guidance_scale=1.0
                        ).images[0]

                        st.image(result_img, caption="✨ Magic Canvas Output", use_container_width=True)
                        st.session_state.edit_messages.append({
                            "role": "assistant", 
                            "content": f"Applied magic: *{edit_instruction}*", 
                            "image": result_img
                        })
                    except Exception as e:
                        st.error(f"Editing Error: {str(e)}")
    else:
        st.info("💡 Upload an image above to unlock the magic editing chat!")