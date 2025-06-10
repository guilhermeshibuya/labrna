import os
import time
import torch


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def train(self, train_loader, val_loader, patience=30, epochs=100, save_path=None, verbose=True):
        self.model.train()

        best_model_state = None
        best_val_loss = float('inf')
        best_epoch = None
        patience_counter = 0

        log_history = []
        total_time = 0.0

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            start_time = time.time()

            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(f'{name} grad mean: {param.grad.mean()}')

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(x)

                loss = self.loss_fn(outputs.squeeze(), y.squeeze())
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            epoch_loss = epoch_loss / len(train_loader)

            end_time = time.time()
            epoch_time = end_time - start_time
            total_time += epoch_time

            val_loss = None
            if val_loader is not None:
                val_loss, _, _ = self.evaluate(val_loader)
                val_loss = val_loss.item()
                if verbose:
                    print(f"Epoch {epoch} / {epochs} | Epoch Time: {epoch_time:.4f}s - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch} / {epochs} | Epoch Time: {epoch_time:.4f}s - Train Loss: {epoch_loss:.4f}")
            if val_loss is not None:
                log_history.append(
                    {
                        'epoch': epoch,
                        'train_loss': epoch_loss,
                        'val_loss': val_loss
                    }
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch
                    best_model_state = self.model.state_dict()
                    if save_path is not None:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': epoch_loss,
                        }, os.path.join(save_path, 'best.pth'))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Best epoch {best_epoch}')
                        print(f'Early stopping triggered at epoch {epoch}')
                        break
            if save_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': epoch_loss,
                }, os.path.join(save_path, "last.pth"))

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"Best model loaded (epoch {best_epoch})")
        if verbose:
            print('Total time: ', total_time)
        return self.model, log_history

    def evaluate(self,  val_loader):
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                preds = self.model(x).squeeze()
                loss = self.loss_fn(preds, y.squeeze())
                total_loss += loss.item()

                all_preds.append(preds.cpu().numpy().reshape(-1))
                all_targets.append(y.cpu().numpy().reshape(-1))
        avg_loss = total_loss / len(val_loader)
        return torch.tensor(avg_loss), torch.cat([torch.tensor(p) for p in all_preds]), torch.cat([torch.tensor(t) for t in all_targets])


