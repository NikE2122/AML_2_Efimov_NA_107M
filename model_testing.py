import pandas as pd
import joblib
import os

# Загружаем модель
model = joblib.load('temperature_model.pkl')

test_files = os.listdir('test')

# Тестирование модели и вывод результатов
for file in test_files:
    df = pd.read_csv(f'test/{file}')
    X_test = df[['Day']]
    predictions = model.predict(X_test)
    df['Predicted_Temperature'] = predictions
    print(f'Predictions for {file}:')
    print(df[['Day', 'Temperature', 'Predicted_Temperature']])