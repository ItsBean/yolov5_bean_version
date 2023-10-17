# Install the bark library (unmentioned but assumed to be done through pip)
# pip install bark

from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# Download and load all models from Bark
preload_models()

# Read from a .txt file
with open("transcript.txt", "r") as file:
    text_prompt = file.read()

# Convert text to speech
speech_array = generate_audio(text_prompt)

# Save the speech to a .wav file
write_wav("transcript.wav", SAMPLE_RATE, speech_array)
