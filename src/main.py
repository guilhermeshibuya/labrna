import torch.optim
from torch.utils.data import DataLoader
from src.model.model import WeatherRNN
import pandas as pd
from src.utils.data import DailyClimateDataset
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

df_train.drop(columns=['meanpressure'], inplace=True)
df_test.drop(columns=['meanpressure'], inplace=True)

val_ratio = 0.2
val_size = int(len(df_train) * val_ratio)
train_part = df_train[:val_size]
val_part = df_train[-val_size:]

train_dataset = DailyClimateDataset(train_part)
val_dataset = DailyClimateDataset(val_part, scaler=train_dataset.scaler)
test_dataset = DailyClimateDataset(df_test, scaler=train_dataset.scaler)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

LEARNING_RATE = 1e-3
EPOCHS = 300
PATIENCE = 10
HIDDEN_SIZE = 16
INPUT_SIZE = train_dataset.X.shape[2]

model = WeatherRNN(INPUT_SIZE, HIDDEN_SIZE, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()


trainer = Trainer(model, optimizer, loss_fn)
trained_model, history = trainer.train(train_loader, val_loader, PATIENCE, EPOCHS)

