from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np


class DataLoader:
    def __init__(self):
        # Wczytanie danych
        digits = load_digits()
        X, y = digits.data, digits.target

        # Skalowanie danych (bardzo ważne dla stabilności uczenia!)
        # Wartości pikseli to 0-16, dzieląc przez 16 sprowadzamy je do zakresu [0, 1]
        X = X / 16.0

        # Podział 60% train, 40% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=42
        )

        # Podział 40% temp na 20% val i 20% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_val_data(self):
        return self.X_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_all_data(self):
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)