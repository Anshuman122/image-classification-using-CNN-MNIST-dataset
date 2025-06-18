from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.inspection import permutation_importance

# PCA analysis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], -1))

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=np.argmax(y_train, axis=1), palette='tab10')
plt.title('PCA of Fashion-MNIST Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Plot classification report
y_true = data_test.iloc[:, 0]
target_names = ["Class {} ({}) :".format(i, labels[i]) for i in range(10)]
print(classification_report(y_true, y_pred, target_names=target_names))
