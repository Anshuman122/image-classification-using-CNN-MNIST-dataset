# 👗 Fashion-MNIST Image Classification with CNN

## Project Description
This project revolves around developing a CNN-based AI model to classify Zalando's Fashion‑MNIST grayscale images into 10 clothing categories (e.g., T-shirt/top, Trouser, Pullover, etc.). The goal is to demonstrate how machine learning and convolutional neural networks can be practically applied in real-world fashion scenarios.

## Dataset Selection
Fashion‑MNIST consists of 60,000 training and 10,000 test examples—each a 28×28 pixel grayscale image labeled with one of 10 clothing classes. It’s a direct, drop-in replacement for the classic MNIST dataset, making it ideal for benchmarking ML algorithms.

## Data Preprocessing & Cleaning
- **Missing values**: None detected—no imputation needed.  
- **Normalization**: Pixel values (0–255) scaled to 0–1.  
- **Reshaping**: Images flattened to arrays of 784 pixels.  
- **One‑hot encoding**: Class labels converted to 10‑dimensional binary vectors.

## Exploratory Data Analysis (EDA)
- **Class distribution**: Balanced across all 10 categories.  
- **Pixel intensity**: Dominated by 0 (dark pixels); lighter pixels less frequent.  
- **Random pixel analysis**: Some locations show brighter values, indicating diverse patterns.  
- **Mean images**: Class‑wise averages highlight distinctive visual features.  
- **Pairwise correlations**: Diagonal dominance and periodic patterns confirmed.  
- **Visualization**: Sample images help verify data quality and label accuracy.

## Feature Visualization
- **PCA (2D projection)** reveals distinct clusters for some categories (e.g., Coat vs. Pullover) and overlap in similar classes (e.g., Shirt vs. T-shirt/top).  
- **Explained variance**: The first principal component captures ~25%; the first 10 explain ~60%, enabling meaningful dimensionality reduction.

## Model Selection
We chose a **Convolutional Neural Network (CNN)** because:
1. **Spatial hierarchies** – CNNs learn local-to-global image patterns.  
2. **Parameter sharing** – Efficiently reduces model size and improves scalability.  
3. **Translation invariance** – Recognizes features irrespective of their position.  
4. **Proven track record** – Effective on tasks like MNIST/CIFAR-10, making it ideal for Fashion-MNIST.

## Training & Testing
Built a CNN in TensorFlow/Keras consisting of convolution, pooling, flatten, dense, and dropout layers for overfitting prevention.  
**Evaluation metrics** on the test set:
- **Accuracy**: 91.03%  
- **Precision**: 90.98%  
- **Recall**: 91.03%  
- **F1-score**: 90.89%

## Activation Visualizations
- **Layer 1 (32 filters)** captures edges and textures.  
- **Layer 2 (64 filters)** extracts mid-level shapes.  
- **Layer 3 (128 filters)** learns abstract, high-level features.
Visualization of training vs. validation accuracy over epochs shows solid learning with no major overfitting.

## Feature Importance
Permutation importance was applied to evaluate how each convolutional layer contributes to model performance.  
- High importance scores were found in middle-to-late layers, indicating deep features are key to classification.

## Results & Findings
- **Model performance**: High metrics demonstrate the robustness of the CNN.  
- **EDA insights**: Guided preprocessing and feature selection.  
- **Feature importance**: Highlighted critical layers for further tuning.
Overall, the project validates CNN effectiveness in fashion item recognition and paves the way for future enhancements in computer vision applications.

## Conclusion
This work successfully addressed the Fashion‑MNIST classification task using CNNs, thorough preprocessing, EDA, and feature interpretation. With strong accuracy and interpretability, it lays a solid foundation for further research and applications in computer vision, especially in fashion tech.


