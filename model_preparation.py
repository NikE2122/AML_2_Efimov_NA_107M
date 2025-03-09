import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import joblib

# Обучаемая модель
model = LinearRegression()

# Обучение модели на данных
train_files = os.listdir('train')
all_train_data = pd.DataFrame()

for file in train_files:
    df = pd.read_csv(f'train/{file}')
    all_train_data = pd.concat([all_train_data, df])

# Обучаем модель на предобработанных данных
X_train = all_train_data[['Day']]
y_train = all_train_data['Scaled_Temperature']
model.fit(X_train, y_train)

# Сохраняем модель
joblib.dump(model, 'temperature_model.pkl')