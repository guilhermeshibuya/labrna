import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import epoch_time


class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, X_train, y_train, X_val=None, y_val=None, patience=30, epochs=100, verbose=True):
        self.model.train()

        best_val_loss = float('inf')
        best_epoch = None
        patience_counter = 0
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            outputs = self.model(X_train).squeeze()
            loss = self.loss_fn(outputs, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            save_path = ""

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': train_loss,
            }, save_path)

            if (X_val and y_val) is not None:
                val_loss = self.evaluate()

                if verbose:
                    print(f"Epoch {epoch} / {epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print('Early stopping triggered')
                        print(f'Epoch {epoch}')
                        break
                if verbose:
                    print(f"Epoch {epoch} / {epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s - Loss: {train_loss:.4f}")

    def evaluate(self,  X_test, y_test):
        self.model.eval()

        with torch.no_grad():
            preds = self.model(X_test).squeeze()
            loss = self.loss_fn(preds, y_test)

        return preds.cpu().numpy(), y_test.cpu().numpy(), loss.items()

