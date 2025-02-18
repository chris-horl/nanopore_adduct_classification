"""
Module for classes to handle dataloading
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import lightning as L

DATA_DIR =  "my_dir"

class NanoporeDataset(Dataset):
    """Dataset for storage of nanopore sequencing data.
    
    Attributes:
        X_sig: signal tensors containing 300 signal values
        X_seq: sequence tensor containing 17 bases
        y: label tensor
    """
    def __init__(self, X_sig: torch.Tensor, X_seq: torch.Tensor, y: torch.Tensor):
        self.X_sig = X_sig
        self.X_seq = X_seq
        self.y = y

    def __len__(self):
        assert len(self.X_sig) == len(self.X_seq)
        return len(self.X_sig)

    def __getitem__(self, index):
        # Get one individual data sample and index (for tracking the sample)
        return self.X_sig[index], self.X_seq[index], self.y[index], index

class NanoporeDataModule(L.LightningDataModule):
    """LightningDataModule to handle preprocessed nanopore data.

    Attributes:
      file_path: directory containing training and test data subdirectories
      batch_size: batch size for forward pass
      sample_size: number of samples to be used
      generator: random number generator for splitting and permutation
      all_labels: torch.tensor containing all labels
      all_sequences: torch.tensor containing all base-called sequences
      all_signals: torch.tensor conataining all signals
      all_data: dataset object containing all data as triples of the above
      used_data: permuted dataset of size sample_size
    """
    
    def __init__(self, file_path: str = DATA_DIR, sample_size: int = 12000, batch_size: int = 512):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.generator = torch.Generator()

    def __collect_nanopore_data(self, stage: str):
        """Data collection from individual npy files
        
        Loads shuffled data for each treatment class, merge into one dataset
        and shuffle for training/validation and testing

        Args:
          stage: fit | test (following Lightning docs)
        """
        
        if stage == "fit":
            dir = "train"
        elif stage == "test":
            dir = "test"

        labels_all_ls, sequences_all_ls, signals_all_ls = [], [], []

        # Gather data over all barcodes (= treatment groups)
        for i in range(1, 14):
            labels_np = np.load(f"{self.file_path + dir}/label/my_file.npy")
            labels_torch = torch.tensor(labels_np, dtype=torch.int64)
            labels_all_ls.append(labels_torch)
            sequences_np = np.load(f"{self.file_path + dir}/sequence/my_file.npy")
            sequences_torch = torch.tensor(sequences_np, dtype=torch.float32)
            sequences_all_ls.append(sequences_torch)
            signals_np = np.load(f"{self.file_path + dir}/signal/my_file.npy")
            signals_torch = torch.tensor(signals_np, dtype=torch.float32)
            signals_all_ls.append(signals_torch)

        # Shuffle and complete dataset
        self.all_labels = torch.cat(labels_all_ls)
        self.all_sequences = torch.cat(sequences_all_ls)
        self.all_signals = torch.cat(signals_all_ls)

        self.all_data = NanoporeDataset(X_sig=self.all_signals,
                                        X_seq=self.all_sequences,
                                        y=self.all_labels)
        sample_indices = torch.randperm(len(self.all_data),
                                            generator=self.generator,
                                            )[:self.sample_size]
        self.used_data = Subset(self.all_data, sample_indices)

    def setup(self, stage: str):
        """Assign and split shuffled data depending on stage
        
        Args:
          stage: fit | test
        """

        if stage == "fit":
            self.__collect_nanopore_data(stage="fit")
            self.train_data, self.val_data = random_split(
                 self.used_data, [.9, .1], generator = self.generator
            )

        elif stage == "test":
            self.__collect_nanopore_data(stage="test")
            self.test_data = self.used_data

        elif stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, generator=self.generator)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        pass

#<<<<< <<<<< <<<<< <<<<< <<<<< <<<<< Testing >>>>> >>>>> >>>>> >>>>> >>>>> >>>>>
def main():
    dm = NanoporeDataModule(sample_size=512)
    dm.setup("fit")
    breakpoint()

if __name__ == "__main__":
    main()
