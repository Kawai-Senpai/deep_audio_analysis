import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def load_audio(file_path, target_sr=16000, duration=10):
    """Load and preprocess audio file"""
    try:
        # Load audio file
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Cut or pad to desired duration
        desired_length = int(duration * target_sr)
        current_length = waveform.size(1)
        
        if current_length > desired_length:
            waveform = waveform[:, :desired_length]
        elif current_length < desired_length:
            padding = desired_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform
    
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def process_audio_batch(file_paths, device, target_sr=16000, duration=10):
    """Process a batch of audio files"""
    waveforms = []
    for path in file_paths:
        waveform = load_audio(path, target_sr, duration)
        if waveform is not None:
            waveforms.append(waveform)
    
    if not waveforms:
        return None
    
    # Stack waveforms into a batch
    waveform_batch = torch.stack(waveforms)
    return waveform_batch.squeeze(1).to(device)  # Remove unnecessary channel dim

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    
    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = F.softmax(attention_weights, dim=1)
        attended = torch.sum(hidden_states * attention_weights, dim=1)
        return attended, attention_weights

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        batch_size, time_steps, *dims = x.size()
        x_reshaped = x.contiguous().view(-1, *dims)
        y = self.module(x_reshaped)
        y = y.contiguous().view(batch_size, time_steps, -1)
        return y

class AudioEncoder(nn.Module):
    def __init__(self, embedding_size=384, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Spectrogram parameters
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = 80
        
        # Spectral processing
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0
        )
        self.mel_transform = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=sample_rate,
            n_stft=self.n_fft // 2 + 1
        )

        # Raw waveform processing
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Frequency processing branch
        self.freq_lstm = nn.LSTM(
            input_size=self.n_mels,
            hidden_size=256,
            num_layers=3,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )

        # Time domain processing branch
        self.time_lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )

        # Attention mechanisms
        self.freq_attention = AttentionLayer(512)  # 512 = bidirectional * hidden_size
        self.time_attention = AttentionLayer(512)

        # Transformer layers for feature interaction
        encoder_layer = TransformerEncoderLayer(
            d_model=1024,  # Combined features size
            nhead=8,
            dim_feedforward=2048,
            dropout=0.3
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=3)

        # Final embedding layers
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x):
        # x shape: (batch, samples)
        batch_size = x.size(0)
        
        # Process raw waveform
        x_raw = x.unsqueeze(1)  # Add channel dimension
        x_raw = self.conv1d(x_raw)
        x_raw = x_raw.transpose(1, 2)  # (batch, time, features)
        x_raw, _ = self.time_lstm(x_raw)
        x_raw_attended, _ = self.time_attention(x_raw)

        # Process spectral features
        x_spec = self.spec_transform(x)  # Compute spectrogram
        x_mel = self.mel_transform(x_spec)  # Convert to mel scale
        x_mel = x_mel.transpose(1, 2)  # (batch, time, mels)
        x_mel, _ = self.freq_lstm(x_mel)
        x_mel_attended, _ = self.freq_attention(x_mel)

        # Combine features
        combined = torch.cat([x_raw_attended, x_mel_attended], dim=1)
        
        # Transform combined features
        combined = combined.unsqueeze(1)  # Add sequence dimension for transformer
        combined = self.transformer(combined)
        combined = combined.squeeze(1)

        # Final embedding
        embedding = self.fc_layers(combined)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

class DescriptionEmbedder(nn.Module):
    def __init__(self, embedding_size=384, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = AudioEncoder(embedding_size=embedding_size)
        self.to(self.device)

    def forward(self, audio_paths):
        """
        Parameters:
        audio_paths: list of paths to audio files
        """
        # Process audio files into tensor batch
        audio_batch = process_audio_batch(audio_paths, self.device)
        if audio_batch is None:
            return None
            
        # Get embeddings
        return self.encoder(audio_batch)

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path, device=None):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return False
        self.load_state_dict(torch.load(path, map_location=device or self.device))
        return True

# Example Usage
if __name__ == "__main__":
    embedder = DescriptionEmbedder()

    # display model size
    print("Model size (MB):", sum(p.numel() for p in embedder.parameters()) / 1e6)

    audio_files = ["cache/classical.mp3", "cache/classical.mp3"]

    try:
        embeddings = embedder(audio_files)
        print("Audio Embeddings shape:", embeddings.shape if embeddings is not None else None)
    except Exception as e:
        print(f"Error: {e}")
