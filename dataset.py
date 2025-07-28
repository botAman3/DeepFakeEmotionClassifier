import torch 
import torchaudio
import logging 
from torch.utils.data import Dataset 
from torchvision.transforms import resize 


class SpectrogramDataset(Dataset):

    def __init__(self , file_paths , labels , config):

        self.file_paths = file_paths
        self.labels = labels 
        self.config = config 
        self.max_len = config["target_sample_rate"] * config["max_duration_seconds"]


        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = config["target_sample_rate"],
            n_fft = config["n_fft"],
            hop_length = config["hop_length"],
            n_mels = config["n_mels"]
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.resize = Resize((config["vit_image_size"] , config["vit_image_size"]) , antialias=True)

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self , idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try :
            waveform , sr = torchaudio.load(file_path)

        except Exception as e :
            logging.warning(f"Error loading file {file_path}: {e}. Skipping.")
            return torch.zeros(3 , self.config["vit_image_size"] , self.config["vit_image_size"]) , torch.tensor(-1.0)

        if sr != self.config["target_sample_rate"]:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.config["target_sample_rate"])
            waveform = resampler(waveform)
        
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        else:
            pad_size = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        spec = self.mel_spectrogram(waveform)
        spec_db = self.amplitude_to_db(spec)
        
        mean = spec_db.mean()
        std = spec_db.std()
        spec_norm = (spec_db - mean) / (std + 1e-6)

        spec_resized = self.resize(spec_norm)
        spec_stacked = spec_resized.repeat(3, 1, 1)

        if torch.isnan(spec_stacked).any():
            logging.warning(f"NaN value detected in spectrogram for file {file_path}. Skipping.")
            return torch.zeros(3, self.config["vit_image_size"], self.config["vit_image_size"]), torch.tensor(-1.0)

        return spec_stacked, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    """Custom collate function to filter out samples that failed to load."""
    batch = [item for item in batch if item[1] != -1]
    if not batch: 
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

