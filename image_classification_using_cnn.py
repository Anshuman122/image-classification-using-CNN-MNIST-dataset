

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.inspection import permutation_importance

"""**Downloading the Dataset from Kaggle**"""

from google.colab import files
files.upload()

!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d zalando-research/fashionmnist

!unzip fashionmnist.zip -d fashionmnist > /dev/null

"""**Reading the CSV Files**"""

data_train = pd.read_csv('/content/fashionmnist/fashion-mnist_train.csv')
data_test = pd.read_csv('/content/fashionmnist/fashion-mnist_test.csv')

"""**Shape of the Training and Testing Data**"""

x_train = np.array(data_train.iloc[:, 1:])
y_train = np.array(data_train.iloc[:, 0])

x_test = np.array(data_test.iloc[:, 1:])
y_test = np.array(data_test.iloc[:, 0])

print("x_train shape:", x_train.shape, "\ty_train shape:", y_train.shape)
print("x_test shape:", x_test.shape, "\ty_test shape:", y_test.shape)

"""**Missing Values**"""

print(data_train.isnull().sum().sum())
print(data_test.isnull().sum().sum())

"""**Normalization**"""

X_train = data_train.drop('label', axis=1) / 255.0
y_train = data_train['label']
X_test = data_test.drop('label', axis=1) / 255.0
y_test = data_test['label']

"""**Reshaping the Data**"""

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

"""**One-Hot Encoding of Labels**"""

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print("Preprocessing done!")

"""# **Exploratory Data Analysis (EDA)**

**Shape of the Data**
"""

print('Training data shape:', data_train.shape)
print('Test data shape:', data_test.shape)

"""**Statistics of the Data**"""

data_train.describe()

"""**Display First Few Rows**"""

data_train.head(10)

"""**Distribution of Classes**"""

labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

def get_classes_distribution(data):
    label_counts = data["label"].value_counts().sort_index()
    total_samples = len(data)
    for i in range(len(label_counts)):
        label = labels[label_counts.index[i]]
        count = label_counts.values[i]
        percent = (count / total_samples) * 100
        print("{:<20s}:   {} or {:.2f}%".format(label, count, percent))

def plot_label_per_class(data, dataset_name):
    label_counts = data['label'].value_counts().sort_index()
    plt.figure(figsize=(12, 5))
    bars = plt.barh(range(10), label_counts.values, color=colors)
    plt.yticks(range(10), [labels[i] for i in range(10)])
    plt.xlabel('Count')
    plt.title(f'Number of labels for each class in {dataset_name}')

    for bar, label in zip(bars, label_counts.values):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{label}', va='center')

    plt.show()

get_classes_distribution(data_train)
plot_label_per_class(data_train, 'Training Set')

get_classes_distribution(data_test)
plot_label_per_class(data_test, 'Test Set')

"""**Pixel Intensity Distribution**"""

plt.figure(figsize=(10, 6))
plt.hist(pixel_values, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Pixel Values')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

"""**Pixel Intensity Distribution of 10 Random Pixels**"""

pixel_columns = ['pixel10', 'pixel200', 'pixel350', 'pixel400', 'pixel470', 'pixel480', 'pixel500', 'pixel600', 'pixel700', 'pixel735']
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for i, pixel in enumerate(pixel_columns):
    sns.histplot(data_train[pixel], color='blue', bins=100, kde=True, ax=axes[i], alpha=0.4)
    axes[i].set_title(f'Distribution of {pixel}')
    axes[i].set_xlabel('Pixel Intensity')
    axes[i].set_ylabel('Density')

plt.tight_layout()
plt.show()

"""**Pixel Intensity Histogram of an Image**"""

image_index = 67
selected_image = X_train[image_index].reshape(28, 28)
pixel_values = selected_image.flatten()

fig, axes = plt.subplots(1, 2, figsize=(5, 5))
axes[0].imshow(selected_image, cmap='Blues')
axes[0].set_title('Selected Image')
axes[0].axis('off')

sns.histplot(pixel_values, bins=50, color='blue', alpha=0.7, ax=axes[1])
axes[1].set_title('Pixel Intensity Histogram')
axes[1].set_xlabel('Pixel Intensity')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

"""**Mean Images per Class**"""

mean_images = data_train.groupby('label').mean()

plt.figure(figsize=(15, 15))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(mean_images.iloc[i].values.reshape(28, 28), cmap='coolwarm')
    plt.title(f"Class: {i}")
    plt.axis('off')
plt.show()

"""**Pairwise Pixel Correlation**"""

subset = data_train.iloc[:, 1:100]  # Taking a subset for simplicity
correlation_matrix = subset.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='viridis')
plt.title('Correlation Heatmap of First 100 Pixels')
plt.show()

"""**Visualize Sample Images**"""

labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
num_classes = len(labels)
plt.figure(figsize=(14, 5))

for class_label in range(num_classes):
    index = np.where(np.argmax(y_train, axis=1) == class_label)[0][0]
    sample_image = X_train[index].reshape(28, 28)
    plt.subplot(1, num_classes, class_label + 1)
    plt.imshow(sample_image, cmap='Blues')
    plt.title(labels[class_label])
    plt.axis('off')

plt.tight_layout()
plt.show()

"""# **Feature Visualization Using PCA**

**Perform PCA and Visualize the Features**
"""

features = data_train.iloc[:, 1:-1]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
data_train['pca-one'] = pca_result[:, 0]
data_train['pca-two'] = pca_result[:, 1]

labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
data_train['label_name'] = data_train['label'].map(labels)

plt.figure(figsize=(14, 8))
sns.scatterplot(x='pca-one', y='pca-two', hue='label_name', palette='tab10', data=data_train, legend='full', alpha=0.6)
plt.title('PCA of Fashion-MNIST')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best', title='Class')
plt.show()

"""**Explained Variance Ratio**"""

pca = PCA(n_components=10)
pca_result = pca.fit_transform(scaled_features)

plt.figure(figsize=(10, 6))
explained_variance = pca.explained_variance_ratio_
plt.bar(range(1, 11), explained_variance, alpha=0.7, align='center', label='individual explained variance')
plt.step(range(1, 11), np.cumsum(explained_variance), where='mid', label='cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.title('Explained Variance Ratio by Principal Components')
plt.show()

"""**Pairplot of PCA Components**"""

pca = PCA(n_components=4)
pca_result = pca.fit_transform(scaled_features)
data_train['pca-one'] = pca_result[:, 0]
data_train['pca-two'] = pca_result[:, 1]
data_train['pca-three'] = pca_result[:, 2]
data_train['pca-four'] = pca_result[:, 3]

sns.pairplot(data_train, vars=['pca-one', 'pca-two', 'pca-three', 'pca-four'], hue='label_name', palette='tab10')
plt.suptitle('Pairplot of PCA Components', y=1.02)
plt.show()

"""**Distribution of PCA Components**"""

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(data_train['pca-one'], kde=True, color='b')
plt.title('Distribution of Principal Component 1')
plt.xlabel('Principal Component 1')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(data_train['pca-two'], kde=True, color='r')
plt.title('Distribution of Principal Component 2')
plt.xlabel('Principal Component 2')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

"""# **Model Implementation (CNN)**

**Model Training & Testing**
"""

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
precision = precision_score(np.argmax(y_test, axis=1), y_pred, average='weighted')
recall = recall_score(np.argmax(y_test, axis=1), y_pred, average='weighted')
f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

"""**Classification Report**"""

y_true = data_test.iloc[:, 0]
target_names = ["Class {} ({}) :".format(i,labels[i]) for i in range(10)]
print(classification_report(y_true, y_pred , target_names=target_names))

"""**Predicted labels Visualization**"""

y_hat = model.predict(X_test)

plt.figure(figsize=(15, 8))
for i, index in enumerate(np.random.choice(X_test.shape[0], size=20, replace=False)):
    plt.subplot(4, 5, i + 1)
    image = np.squeeze(X_test[index]).reshape((28, 28))
    plt.imshow(image, cmap='gray')
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    if predict_index == true_index:
        plt.title(labels[predict_index], color='green')
    else:
        plt.title(labels[predict_index], color='red')
    plt.axis('off')

plt.show()

"""# **Feature Importance Visualization**"""

def compute_accuracy(X, y):
    y_pred = np.argmax(model.predict(X), axis=1)
    return accuracy_score(y, y_pred)

baseline_accuracy = compute_accuracy(X_val, y_val)
print("Baseline Accuracy:", baseline_accuracy)
importance_scores = []

for layer in model.layers:
    if 'conv' in layer.name:
        print("Processing layer:", layer.name)
        original_weights = layer.get_weights()
        layer.set_weights([np.zeros_like(w) for w in original_weights])
        accuracy_after_zeroing = compute_accuracy(X_val, y_val)
        importance_score = baseline_accuracy - accuracy_after_zeroing
        importance_scores.append(importance_score)
        layer.set_weights(original_weights)

plt.figure(figsize=(8, 6))
plt.bar(range(len(importance_scores)), importance_scores, color=colors)
plt.title('Permutation Importance')
plt.xlabel('Convolutional Layer')
plt.ylabel('Importance Score')
plt.xticks(range(len(importance_scores)), [layer.name for layer in model.layers if 'conv' in layer.name], rotation=45)
plt.show()
