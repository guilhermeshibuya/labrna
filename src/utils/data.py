from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np
import torch


class DailyClimateDataset(Dataset):
    def __init__(self, df, window_size=7, target_column=0, scaler=None):
        data = df.values

        if scaler is None:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            data = self.scaler.transform(data)

        self.X, self.y = self.create_sequences(data, window_size, target_column)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

    def create_sequences(self, data, window_size, target_column):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size][target_column])
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
