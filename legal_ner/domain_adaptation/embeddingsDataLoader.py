from torch.utils.data import Dataset
import torch
import numpy as np

class EmbeddingDataset(Dataset):

    def __init__(self, embeddings_path, labels_path):
        try:
            self.embeddings = torch.load(embeddings_path)
            self.labels = torch.load(labels_path)
        except: # debug
            self.embeddings = torch.randn((100, 1024))
            self.labels = torch.tensor(np.random.randint(0,10, (100,)), dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]