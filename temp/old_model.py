import warnings
import torch
from torch import nn
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import librosa
from typing import List, Union
from pydub import AudioSegment
import os
from functools import lru_cache
import torchaudio
import random
import numpy as np

warnings.filterwarnings("ignore")

class AudioFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=22050, n_mels=64):  # Reduced mel bands
        super().__init__()
        self.sample_rate = sample_rate
        
        # Simplified feature extraction - just mel spectrograms
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,  # Reduced FFT size
            hop_length=512
        )
        
        # Lighter conv network
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))  # Smaller output size
        )
        
        # Smaller output dimension
        self.output_dim = 32 * 16 * 16

    def _convert_to_wav(self, audio_path):
        """Convert audio file to WAV if needed"""
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        if file_ext != '.wav':
            try:
                # Convert to WAV using pydub
                audio = AudioSegment.from_file(audio_path)
                # Create temp wav in same directory as source
                dir_path = os.path.dirname(os.path.abspath(audio_path))
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                wav_path = os.path.join(dir_path, f"{base_name}_temp.wav")
                audio.export(wav_path, format='wav')
                return wav_path, True  # Second value indicates if file is temporary
            except Exception as e:
                print(f"Error converting audio: {str(e)}")
                return audio_path, False
        return audio_path, False

    def forward(self, audio_paths):
        # Ensure input is a list
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        elif isinstance(audio_paths, (list, tuple)):
            audio_paths = list(audio_paths)
        else:
            raise ValueError(f"Expected string or list of strings, got {type(audio_paths)}")

        valid_paths = []
        for path in audio_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                print(f"Warning: File not found - {path}")

        if not valid_paths:
            raise ValueError("No valid audio files found in the input")

        features_list = []
        temp_files = []

        for path in valid_paths:
            try:
                # Convert to WAV if needed
                wav_path, is_temp = self._convert_to_wav(path)
                if is_temp:
                    temp_files.append(wav_path)

                # Load and process audio
                waveform, sr = torchaudio.load(wav_path)
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        sr, self.sample_rate
                    )
                    waveform = resampler(waveform)
                
                # Convert to mono
                if waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Simple feature extraction
                mel_spec = self.mel_spec(waveform)
                mel_spec = (mel_spec + 1e-9).log2()
                features = self.feature_encoder(mel_spec.unsqueeze(0))
                features_list.append(features)
                
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                continue  # Skip failed files instead of using zero tensor
                
            finally:
                # Cleanup temp files
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        print(f"Error removing temporary file {temp_file}: {str(e)}")
        
        if not features_list:
            raise ValueError("No features could be extracted from any of the input files")

        batch_features = torch.stack(features_list)
        return batch_features.flatten(start_dim=2)

class GRU_Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2):  # Reduced complexity
        super().__init__()
        
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_classes),
            nn.LayerNorm(num_classes)
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden)

class AudioClassifier(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        
        # Only handle device at top level
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create components without device
        self.extractor = AudioFeatureExtractor()
        self.genre_classifier = GRU_Classifier(self.extractor.output_dim, num_classes=8)
        self.emotion_classifier = GRU_Classifier(self.extractor.output_dim, num_classes=6)
        self.usecase_classifier = GRU_Classifier(self.extractor.output_dim, num_classes=5)
        
        # Move entire model to device at once
        self.to(self.device)

    def forward(self, audio_paths):
        """
        Process multiple audio files in batch

        Args:
            audio_paths: str or List[str] - Path or list of paths to audio files

        Returns:
            dict: Dictionary containing batch predictions for each classifier
                 Each key contains a tensor of shape (batch_size, num_classes)
        """
        features = self.extractor(audio_paths)
        
        # Process features in batch
        genre_logits = self.genre_classifier(features)
        emotion_logits = self.emotion_classifier(features)
        usecase_logits = self.usecase_classifier(features)
        
        # Return batch predictions
        return {
            'genre': torch.softmax(genre_logits, dim=1),
            'emotion': torch.softmax(emotion_logits, dim=1),
            'usecase': torch.softmax(usecase_logits, dim=1)
        }

def main():
    try:
        model = AudioClassifier()
        
        # Example with multiple audio files
        audio_files = [
            'test_data.mp3'
        ]
        
        # Filter existing files
        audio_files = [f for f in audio_files if os.path.exists(f)]
        if not audio_files:
            print("No valid audio files found!")
            return

        print(f"Processing {len(audio_files)} audio files...")
        with torch.no_grad():
            model.eval()
            outputs = model(audio_files)
            
            # Print batch predictions
            for i, file in enumerate(audio_files):
                print(f"\nResults for {os.path.basename(file)}:")
                for name, probs in outputs.items():
                    print(f"\n{name.capitalize()} probabilities:")
                    for j, prob in enumerate(probs[i]):
                        print(f"  Class {j}: {prob:.4f}")
                print("-" * 50)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()