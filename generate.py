import streamlit as st
import time
import torch
import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Load model and processor
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    return processor, model

processor, model = load_model()

# App title
st.title("🎵 MusicGen - Generate Music from Text")

# User input
prompt = st.text_input("Enter a music prompt (e.g., 'rock music with guitar and drums')", 
                       value="rock music with guitar and drums")

generate_btn = st.button("Generate Music")

if generate_btn and prompt:
    with st.spinner("Generating music..."):
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        
        start = time.time()
        audio_values = model.generate(**inputs, max_new_tokens=500)
        end = time.time()

        # Save to file
        output_file = "musicgen_out.wav"
        sampling_rate = model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(output_file, rate=sampling_rate, data=audio_values[0, 0].numpy())

        st.success(f"Music generated in {end - start:.2f} seconds!")
        st.audio(output_file, format="audio/wav")

        # Option to download
        with open(output_file, "rb") as f:
            st.download_button("Download Music", f, file_name="generated_music.wav", mime="audio/wav")
