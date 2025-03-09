#!/bin/bash

# Запуск создания данных
python data_creation.py

# Запуск предобработки данных
python model_preprocessing.py

# Запуск подготовки модели
python model_preparation.py

# Запуск тестирования модели
python model_testing.py