import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import warnings
from model_utils import extract_features

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

#! Model Definitions ---------------------------------------------------------------------
#? Audio Classifier ---------------------------------------------------------------------
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
        # Dropout
        self.dropout = nn.Dropout(0.3)
        self.dropout_conv = nn.Dropout2d(0.3)
        # Calculate the size of the output from the convolutional layers
        self._to_linear = None
        self.convs(torch.randn(1, 1, 64, 5290))
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def convs(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)
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
        return x

#! Wrapper Model -------------------------------------------------------------------------
class GenureClassifier(nn.Module):
    
    def __init__(self, genre_classes=10, device=None):
        super(GenureClassifier, self).__init__()
        # Define a transformation pipeline
        self.transform = T.MelSpectrogram(
            sample_rate=16000, 
            n_fft=1024, 
            hop_length=512, 
            n_mels=64
        )
        self.genre_classifier = AudioClassifier(num_classes=genre_classes)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, audio_paths):
        """
        Predict genre for a list of audio files in parallel.
        """
        # Extract features for all audio files using the utility function
        features_batch = extract_features(audio_paths, self.transform, self.device)

        if features_batch is None:
            return None

        # Pass the batch of features through the classifier
        genre_preds = self.genre_classifier(features_batch)
        return genre_preds

    def save(self, path):
        """
        Save the model's state_dict to the specified path.
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path, device=None):
        """
        Load the model's state_dict from the specified path.
        """
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            return False
        
        self.load_state_dict(torch.load(path, map_location=device or self.device))
