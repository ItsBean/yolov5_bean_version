import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_model_and_processor():
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    model.config.forced_decoder_ids = None
    return processor, model


def resample_audio(waveform, sample_rate, target_sample_rate=16000):
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def split_audio(waveform, chunk_size=int(480000/2)):  # 480000 samples = 30 seconds for 16kHz audio
    num_chunks = int(len(waveform[0]) / chunk_size) + 1
    return [waveform[:, i * chunk_size:i * chunk_size + chunk_size] for i in range(num_chunks)]


def transcribe_mp3(mp3_path, processor, model):
    try:
        waveform, sample_rate = torchaudio.load(mp3_path)
        waveform, sample_rate = resample_audio(waveform, sample_rate)

        audio_chunks = split_audio(waveform)
        transcriptions = []

        for chunk in audio_chunks:
            input_features = processor(chunk.squeeze().numpy(), sampling_rate=sample_rate,
                                       return_tensors="pt").input_features
            predicted_ids = model.generate(input_features)
            transcription_chunk = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcriptions.append(transcription_chunk[0])

        return ' '.join(transcriptions)
    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred: {e}"


if __name__ == "__main__":
    processor, model = load_model_and_processor()
    mp3_path = "/home/wxf/Downloads/Interview Use Bloom's to Think Critically.mp4"
    transcription = transcribe_mp3(mp3_path, processor, model)
    print(transcription)
    # save to txt
    with open("transcript.txt", "w") as f:
        f.write(transcription)
