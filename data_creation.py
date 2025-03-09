import numpy as np
import pandas as pd
import os

def generate_temperature_data(num_days=100, noise_level=1.0, anomalies=None):
    np.random.seed(42)
    days = np.arange(num_days)
    base_temperature = 20 + 10 * np.sin(days / 10)  # периодическая температура
    noise = noise_level * np.random.randn(num_days)  # шум
    temperature_data = base_temperature + noise

    # Вставка аномалий
    if anomalies is not None:
        for day, anomaly_value in anomalies.items():
            temperature_data[day] = anomaly_value

    return days, temperature_data

def save_data(days, temperature_data, filename):
    df = pd.DataFrame({'Day': days, 'Temperature': temperature_data})
    df.to_csv(filename, index=False)

# Создаем директории для train и test
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Генерация и сохранение наборов данных
# Набор данных 1
days, temperatures = generate_temperature_data(num_days=100, noise_level=2.0)
save_data(days, temperatures, 'train/train_data_1.csv')

# Набор данных 2
days, temperatures = generate_temperature_data(num_days=100, noise_level=5.0, 
                                               anomalies={20: 50, 80: -10})
save_data(days, temperatures, 'train/train_data_2.csv')

# Набор данных 3
days, temperatures = generate_temperature_data(num_days=100, noise_level=0.5)
save_data(days, temperatures, 'test/test_data_1.csv')

# Набор данных 4
days, temperatures = generate_temperature_data(num_days=100, noise_level=4.0, 
                                               anomalies={10: -20, 90: 30})
save_data(days, temperatures, 'test/test_data_2.csv')