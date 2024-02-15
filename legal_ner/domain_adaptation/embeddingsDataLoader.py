from torch.utils.data import Dataset
import torch
import numpy as np

class EmbeddingDataset(Dataset):

    def __init__(self, embeddings_path, labels_path):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            self.embeddings = torch.load(embeddings_path, map_location=device)
            self.labels = torch.load(labels_path, map_location=device)
        except: # debug
            self.embeddings = torch.randn((100, 1024))
            self.labels = torch.tensor(np.random.randint(0,10, (100,)), dtype=torch.long)
            raise Exception("EmbeddingDataset: error loading embeddings or labels")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]