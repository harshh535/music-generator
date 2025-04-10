import scipy
import time
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    # text=["80s pop track with bassy drums and synth"],
    text=["rock music with guitar and drums"],
    padding=True,
    return_tensors="pt",
)

start = time.time()
audio_values = model.generate(**inputs, max_new_tokens=500)
print(time.time() - start) # Log time taken in generation

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out2.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())