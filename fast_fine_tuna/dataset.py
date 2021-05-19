from torch.utils.data import Dataset
import torch

class MainDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class MainDatasetDouble(Dataset):
    def __init__(self, encodings, labels_A, labels_B):
        self.encodings = encodings
        self.labels_A = labels_A
        self.labels_B = labels_B

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels_A'] = torch.tensor(self.labels_A[idx])
        item['labels_B'] = torch.tensor(self.labels_B[idx])
        return item

    def __len__(self):
        return len(self.labels_B)
