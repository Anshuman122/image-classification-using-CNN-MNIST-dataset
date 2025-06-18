import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Visualize training
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Model evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Model predictions visualization
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

# Permutation importance calculation
def compute_accuracy(X, y):
    y_pred = np.argmax(model.predict(X), axis=1)
    return accuracy_score(y, y_pred)

baseline_accuracy = compute_accuracy(X_test, np.argmax(y_test, axis=1))
print("Baseline Accuracy:", baseline_accuracy)

importance_scores = []
for layer in model.layers:
    if 'conv' in layer.name:
        original_weights = layer.get_weights()
        layer.set_weights([np.zeros_like(w) for w in original_weights])
        accuracy_after_zeroing = compute_accuracy(X_test, np.argmax(y_test, axis=1))
        importance_score = baseline_accuracy - accuracy_after_zeroing
        importance_scores.append(importance_score)
        layer.set_weights(original_weights)

plt.figure(figsize=(8, 6))
plt.bar(range(len(importance_scores)), importance_scores, color='blue')
plt.title('Permutation Importance of Convolutional Layers')
plt.xlabel('Convolutional Layer')
plt.ylabel('Importance Score')
plt.xticks(range(len(importance_scores)), [layer.name for layer in model.layers if 'conv' in layer.name], rotation=45)
plt.show()
