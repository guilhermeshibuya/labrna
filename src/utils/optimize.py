import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import optuna
from typing import Literal
import pandas as pd

from model.model import RNN
from utils.data import DailyClimateDataset
from utils.seed import define_seed
from utils.trainer import Trainer


def objective(trial):
    df_train = pd.read_csv(
        '../../datasets/DailyDelhiClimateTrain.csv',
        parse_dates=['date'],
        index_col='date'
    )

    df_test = pd.read_csv(
        '../../datasets/DailyDelhiClimateTest.csv',
        parse_dates=['date'],
        index_col='date'
    )

    df_train.drop(columns=['meanpressure'], inplace=True)
    df_test.drop(columns=['meanpressure'], inplace=True)

    define_seed(42)

    val_ratio = 0.2
    val_size = int(len(df_train) * val_ratio)
    train_part = df_train[:-val_size]
    val_part = df_train[-val_size:]

    sequence_length = trial.suggest_int("sequence_length", 7, 63, step=7)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32])

    train_dataset = DailyClimateDataset(train_part, window_size=sequence_length)
    val_dataset = DailyClimateDataset(val_part, window_size=sequence_length, scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_size = train_dataset.X.shape[2]
    output_size = 1
    patience = 20
    epochs = 300

    hidden_size = trial.suggest_int("hidden_size", 8, 128)
    num_layers = trial.suggest_int("num_layers", 1, 6)
    activation = trial.suggest_categorical("activation", ["tanh", "sigmoid", "relu"])
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = RNN(input_size, hidden_size, output_size, num_layers, activation)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    trainer = Trainer(model, optimizer, loss_fn)
    _, history = trainer.train(train_loader, val_loader, patience, epochs)

    val_losses = [entry['val_loss'] for entry in history]
    return min(val_losses)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best Hyperparameters:", study.best_params)
