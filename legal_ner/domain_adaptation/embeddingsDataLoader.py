from torch.utils.data import Dataset
import torch

class EmbeddingDataset(Dataset):
    def init(self, embeddings_path, labels_path):
        self.embeddings = torch.load(embeddings_path)
        self.labels = torch.load(labels_path)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]