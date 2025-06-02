import matplotlib.pyplot as plt


def plot_train_val_loss(log_history):
    epochs = []
    train_losses = []
    val_losses = []

    for log in log_history:
        epochs.append(log['epoch'])
        train_losses.append(log['train_loss'])
        val_losses.append(log['val_loss'])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Val Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

