import os
import h5py
import numpy as np
from typing import Any, Tuple
import torch
import random
from pytorch_lightning import LightningDataModule
import torchaudio
from torchaudio.functional import apply_codec
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Optional, Tuple

class MusdbMoisesdbDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        sr: int = 16000,
        segments: int = 10,
        num_stems: int = 4,
        snr_range: Tuple[int, int] = (-10, 10),
        num_samples: int = 1000,
    ) -> None:
        
        self.data_dir = data_dir
        self.segments = int(segments * sr)
        self.sr = sr
        self.num_stems = num_stems
        self.snr_range = snr_range
        self.num_samples = num_samples
        
        self.instruments = [
            "bass", 
            "bowed_strings", 
            "drums", 
            "guitar",
            "other", 
            "other_keys", 
            "other_plucked", 
            "percussion", 
            "piano", 
            "vocals", 
            "wind"
        ]

    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            select_stems = random.randint(1, self.num_stems)
            select_stems = random.choices(self.instruments, k=select_stems)
            ori_wav = []
            for stem in select_stems:
                h5path = random.choice(os.listdir(os.path.join(self.data_dir, stem)))
                datas = h5py.File(os.path.join(self.data_dir, stem, h5path), 'r')['data']
                random_index = random.randint(0, datas.shape[0]-1)
                music_wav = torch.FloatTensor(datas[random_index])
                start = random.randint(0, music_wav.shape[-1] - self.segments)
                music_wav = music_wav[:, start:start+self.segments]
                
                rescale_snr = random.randint(self.snr_range[0], self.snr_range[1])
                music_wav = music_wav * np.sqrt(10**(rescale_snr/10))
                ori_wav.append(music_wav)
            ori_wav = torch.stack(ori_wav).sum(0)
        else:
            h5path = random.choice(os.listdir(os.path.join(self.data_dir, "mixture")))
            datas = h5py.File(os.path.join(self.data_dir, "mixture", h5path), 'r')['data']
            random_index = random.randint(0, datas.shape[0]-1)
            music_wav = torch.FloatTensor(datas[random_index])
            start = random.randint(0, music_wav.shape[-1] - self.segments)
            ori_wav = music_wav[:, start:start+self.segments]
  
        return ori_wav
    

class MusdbMoisesdbEval(Dataset):
    def __init__(
        self,
        data_dir: str
    ) -> None:
        self.data_path = os.listdir(data_dir)
        self.data_path = [os.path.join(data_dir, i) for i in self.data_path]
        
    def __len__(self) -> int:
        return len(self.data_path)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ori_wav = torchaudio.load(self.data_path[idx]+"/ori.wav")[0]
        
        return ori_wav, self.data_path[idx]
    
class MusdbMoisesdbDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        eval_dir: str,
        sr: int = 16000,
        segments: int = 10,
        num_stems: int = 4,
        snr_range: Tuple[int, int] = (-10, 10),
        num_samples: int = 1000,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            self.data_train = MusdbMoisesdbDataset(
                data_dir=self.hparams.train_dir,
                sr=self.hparams.sr,
                segments=self.hparams.segments,
                num_stems=self.hparams.num_stems,
                snr_range=self.hparams.snr_range,
                num_samples=self.hparams.num_samples,
            )
            
            self.data_val = MusdbMoisesdbEval(
                data_dir=self.hparams.eval_dir
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )