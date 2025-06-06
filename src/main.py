import torch.optim
from torch.utils.data import DataLoader
from torcheval.metrics import R2Score
import matplotlib.pyplot as plt
import seaborn as sns
from model.model import RNN
import pandas as pd
from src.utils.data import DailyClimateDataset
from utils.plot import plot_train_val_loss, plot_predictions_and_labels
from utils.trainer import Trainer

df_train = pd.read_csv(
    '../datasets/DailyDelhiClimateTrain.csv',
    parse_dates=['date'],
    index_col='date'
)

df_test = pd.read_csv(
    '../datasets/DailyDelhiClimateTest.csv',
    parse_dates=['date'],
    index_col='date'
)

# print(df_train['meanpressure'].corr(df_train['meantemp']))

# corr_matrix = df_train.corr()
#
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f")
# plt.title('Matriz de correlacao')
# plt.show()

df_train.drop(columns=['meanpressure'], inplace=True)
df_test.drop(columns=['meanpressure'], inplace=True)

val_ratio = 0.2
val_size = int(len(df_train) * val_ratio)
train_part = df_train[:val_size]
val_part = df_train[-val_size:]

SEQUENCE_LENGTH = 7

train_dataset = DailyClimateDataset(train_part, window_size=SEQUENCE_LENGTH)
val_dataset = DailyClimateDataset(val_part, window_size=SEQUENCE_LENGTH, scaler=train_dataset.scaler)
test_dataset = DailyClimateDataset(df_test, window_size=SEQUENCE_LENGTH, scaler=train_dataset.scaler)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

LEARNING_RATE = 1e-3
EPOCHS = 300
PATIENCE = 10
HIDDEN_SIZE = 16
OUTPUT_SIZE = 1
NUM_LAYERS = 1
INPUT_SIZE = train_dataset.X.shape[2]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, 'tanh')
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

trainer = Trainer(model, optimizer, loss_fn)
trained_model, history = trainer.train(train_loader, val_loader, PATIENCE, EPOCHS)

metric = R2Score()
model.eval()
with torch.no_grad():
    outputs = model(test_dataset.X).squeeze()
    target = test_dataset.y.squeeze()
    test_loss = loss_fn(outputs, target)
    metric.update(outputs, target)
    test_metric = metric.compute()
    print(f'Test Loss: {test_loss.item():.4f}')
    print(f'R2: {test_metric}')

y_pred = outputs.cpu().detach().numpy()
y_true_test = test_dataset.y.numpy()

dates = df_test.index[SEQUENCE_LENGTH:]
plot_train_val_loss(history)
plot_predictions_and_labels(dates, y_pred, y_true_test)
