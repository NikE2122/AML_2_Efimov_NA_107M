import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(data_frame):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_frame[['Temperature']])
    data_frame['Scaled_Temperature'] = scaled_data
    return data_frame

train_files = os.listdir('train')
test_files = os.listdir('test')

# Предобработка тренировочных данных
for file in train_files:
    df = pd.read_csv(f'train/{file}')
    preprocessed_df = preprocess_data(df)
    preprocessed_df.to_csv(f'train/preprocessed_{file}', index=False)

# Предобработка тестовых данных
for file in test_files:
    df = pd.read_csv(f'test/{file}')
    preprocessed_df = preprocess_data(df)
    preprocessed_df.to_csv(f'test/preprocessed_{file}', index=False)