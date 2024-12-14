import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

#* Audio Model Ultilities -------------------------------------------------------
def convert_mp3_to_wav(mp3_path):
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

def preprocess_audio(audio, sample_rate, transform, device, duration=30):
    # Move audio to device
    audio = audio.to(device)
    # Standardize audio length to specified duration (samples at given sample rate)
    max_length = sample_rate * duration  # duration in seconds
    if audio.shape[1] > max_length:
        audio = audio[:, :max_length]
    else:
        padding = max_length - audio.shape[1]
        audio = F.pad(audio, (0, padding))
    # Convert raw audio to Mel spectrogram
    mel_spec = transform(audio)
    # Apply logarithmic scale for better dynamic range
    log_mel_spec = torch.log(mel_spec + 1e-9)
    return log_mel_spec

def extract_features(audio_paths, transform, device, sr=16000, duration=30):
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
            if audio_path.endswith(".mp3"):
                audio_path_converted = convert_mp3_to_wav(audio_path)
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
            feature = preprocess_audio(audio.to(device), sr, transform, device, duration)

            features_list.append(feature)

            # Delete the converted WAV file if it was created
            if delete_wav:
                os.remove(audio_path_converted)

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue

    if features_list:
        # Stack features into a tensor
        features_batch = torch.stack(features_list, dim=0)
        return features_batch
    else:
        return None
