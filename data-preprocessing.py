import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load datasets
data_train = pd.read_csv('/content/fashionmnist/fashion-mnist_train.csv')
data_test = pd.read_csv('/content/fashionmnist/fashion-mnist_test.csv')

# Check for missing values
print(data_train.isnull().sum().sum())
print(data_test.isnull().sum().sum())

# Normalize and reshape data
X_train = data_train.drop('label', axis=1) / 255.0
y_train = data_train['label']
X_test = data_test.drop('label', axis=1) / 255.0
y_test = data_test['label']

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print("Preprocessing done!")

# Print data shapes
print("x_train shape:", X_train.shape, "\ty_train shape:", y_train.shape)
print("x_test shape:", X_test.shape, "\ty_test shape:", y_test.shape)
