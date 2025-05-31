import os
import time
import torch
from src.utils.utils import epoch_time


class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, train_loader, val_loader=None, patience=30, epochs=100, save_path=None, verbose=True):
        self.model.train()

        best_val_loss = float('inf')
        best_epoch = None
        patience_counter = 0
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            start_time = time.time()

            for x, y in train_loader:
                outputs = self.model(x).squeeze()
                loss = self.loss_fn(outputs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if save_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': epoch_loss,
                }, os.path.join(save_path, f"_epoch{epoch}"))

            val_loss = 0.0
            if val_loader is not None:
                for x, y in val_loader:
                    _, _, loss = self.evaluate(x, y)
                    val_loss += loss
                if verbose:
                    print(f"Epoch {epoch} / {epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch} / {epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s - Train Loss: {epoch_loss:.4f}")
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Best epoch {best_epoch}')
                        print(f'Early stopping triggered at epoch {epoch}')
                        break

        return self.model

    #     # def train(self, train_dataset, val_dataset=None, patience=30, epochs=100, save_path=None, verbose=True):
    #     #     self.model.train()
    #     #
    #     #     best_val_loss = float('inf')
    #     #     best_epoch = None
    #     #     patience_counter = 0
    #     #     for epoch in range(1, epochs + 1):
    #     #         epoch_loss = 0.0
    #     #         start_time = time.time()
    #     #
    #     #         outputs = self.model(train_dataset.X).squeeze()
    #     #         loss = self.loss_fn(outputs, train_dataset.y)
    #     #
    #     #         self.optimizer.zero_grad()
    #     #         loss.backward()
    #     #         self.optimizer.step()
    #     #
    #     #         epoch_loss = loss.item()
    #     #
    #     #         end_time = time.time()
    #     #         epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #     #
    #     #         if save_path is not None:
    #     #             torch.save({
    #     #                 'epoch': epoch,
    #     #                 'model_state_dict': self.model.state_dict(),
    #     #                 'optimizer_state_dict': self.optimizer.state_dict(),
    #     #                 'loss': epoch_loss,
    #     #             }, os.path.join(save_path, f"_epoch{epoch}"))
    #     #
    #     #         val_loss = 0.0
    #     #         if val_dataset is not None:
    #     #             for x, y in val_dataset:
    #     #                 _, _, loss = self.evaluate(x, y)
    #     #                 val_loss += loss
    #     #             if verbose:
    #     #                 print(f"Epoch {epoch} / {epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
    #     #         else:
    #     #             if verbose:
    #     #                 print(f"Epoch {epoch} / {epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s - Train Loss: {epoch_loss:.4f}")
    #     #         if val_loss is not None:
    #     #             if val_loss < best_val_loss:
    #     #                 best_val_loss = val_loss
    #     #                 patience_counter = 0
    #     #                 best_epoch = epoch
    #     #             else:
    #     #                 patience_counter += 1
    #     #                 if patience_counter >= patience:
    #     #                     print(f'Best epoch {best_epoch}')
    #     #                     print(f'Early stopping triggered at epoch {epoch}')
    #     #                     break
    #     #
    #     #     return self.modelel
    #

    def evaluate(self,  X_test, y_test):
        self.model.eval()

        with torch.no_grad():
            preds = self.model(X_test).squeeze()
            loss = self.loss_fn(preds, y_test)

        return preds.cpu().numpy(), y_test.cpu().numpy(), loss.item()

