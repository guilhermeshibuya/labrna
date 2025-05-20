from src.model.model import WeatherRNN
import numpy as np
import torch
import pandas as pd

data_train = pd.read_csv(
    '../datasets/DailyDelhiClimateTrain.csv',
    parse_dates=['date'],
    index_col='date'
)
