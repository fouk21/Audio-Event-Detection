import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import os

model_name = "ALM/wav2vec2-large-audioset"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

model.eval()

def classify_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label_id = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_label_id]
    return predicted_label

if __name__ == "__main__":
    test_audio_path = "path_to_your_audio_file.wav"
    if os.path.exists(test_audio_path):
        label = classify_audio(test_audio_path)
        print(f"Predicted label: {label}")
    else:
        print(f"Audio file {test_audio_path} not found.")
