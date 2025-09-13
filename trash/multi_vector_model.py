import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import warnings
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        
        super(AudioClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Add an adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        # Calculate the size of the output from the convolutional layers
        self._to_linear = None
        self.convs(torch.randn(1, 1, 64, 5290))
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        # Dropout
        self.dropout = nn.Dropout(0.3)

    def convs(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # sigmoid at the end
        x = F.sigmoid(x)
        return x

class PretrainedAudioClassifier(nn.Module):
    
    def __init__(self, genre_classes=8, emotion_classes=6, usecase_classes=5, device=None):
        super(PretrainedAudioClassifier, self).__init__()
        # Define a transformation pipeline
        self.transform = T.MelSpectrogram(
            sample_rate=16000, 
            n_fft=1024, 
            hop_length=512, 
            n_mels=64
        )
        # Classifiers
        self.genre_classifier = AudioClassifier(num_classes=genre_classes)
        self.emotion_classifier = AudioClassifier(num_classes=emotion_classes)
        self.usecase_classifier = AudioClassifier(num_classes=usecase_classes)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def convert_mp3_to_wav(self, mp3_path):
        """
        Convert MP3 file to WAV format.
        """
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            wav_path = mp3_path.replace(".mp3", ".wav")
            audio.export(wav_path, format="wav")
            return wav_path
        except CouldntDecodeError as e:
            raise ValueError(f"Failed to decode {mp3_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to convert {mp3_path} to WAV: {str(e)}")

    def preprocess_audio(self, audio, sample_rate):
        # Move audio to the same device as the model
        audio = audio.to(self.device)
        # Standardize audio length to 30 seconds (480,000 samples at 16kHz)
        max_length = sample_rate * 30  # 30 seconds
        if audio.shape[1] > max_length:
            audio = audio[:, :max_length]
        else:
            padding = max_length - audio.shape[1]
            audio = F.pad(audio, (0, padding))
        # Convert raw audio to Mel spectrogram
        mel_spec = self.transform(audio)
        # Apply logarithmic scale for better dynamic range
        log_mel_spec = torch.log(mel_spec + 1e-9)
        return log_mel_spec  # Shape: [1, n_mels, time_steps]
    
    def extract_features(self, audio_paths, sr=16000):
        """
        Extract Mel spectrogram features for a list of audio files.
        """
        features_list = []

        for audio_path in audio_paths:
            try:
                # Check if file exists
                if not os.path.isfile(audio_path):
                    print(f"File does not exist: {audio_path}")
                    continue

                # Convert MP3 to WAV if necessary
                if (audio_path.endswith(".mp3")):
                    audio_path_converted = self.convert_mp3_to_wav(audio_path)
                    delete_wav = True
                else:
                    audio_path_converted = audio_path
                    delete_wav = False

                audio, sample_rate = torchaudio.load(audio_path_converted)
                # Convert to mono if necessary
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                # Resample if necessary
                if sample_rate != sr:
                    resampler = T.Resample(orig_freq=sample_rate, new_freq=sr)
                    audio = resampler(audio)
                # Move audio to device and preprocess
                feature = self.preprocess_audio(audio.to(self.device), sr)

                features_list.append(feature)

                # Delete the converted WAV file if it was created
                if delete_wav:
                    os.remove(audio_path_converted)

            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                continue

        if features_list:
            # Stack features into a tensor
            features_batch = torch.stack(features_list, dim=0)  # Shape: [batch_size, 1, n_mels, time_steps]
            return features_batch
        else:
            return None

    def forward(self, audio_paths):
        """
        Predict genre, emotion, and use case for a list of audio files in parallel.
        """
        # Extract features for all audio files
        features_batch = self.extract_features(audio_paths)

        if features_batch is None:
            return None

        # Features already have correct dimensions; no need to unsqueeze
        # Pass the batch of features through the classifiers
        genre_preds = self.genre_classifier(features_batch)
        emotion_preds = self.emotion_classifier(features_batch)
        usecase_preds = self.usecase_classifier(features_batch)

        # Concatenate predictions into a single tensor
        combined_preds = torch.cat((genre_preds, emotion_preds, usecase_preds), dim=1)

        return combined_preds

    def save(self, path):
        """
        Save the model's state_dict to the specified path.
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path, device=None):
        """
        Load the model's state_dict from the specified path.
        """
        # check if exists
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            return False
        
        #load state dict to self
        self.load_state_dict(torch.load(path, map_location=device or self.device))

# Example Usage
if __name__ == "__main__":
    classifier = PretrainedAudioClassifier()
    audio_files = ["classical.mp3","classical.mp3"]

    try:
        predictions = classifier.predict(audio_files)
        print("Combined Predictions:", predictions)
    except Exception as e:
        print(f"Error: {e}")
