# 🤖 ML Model Explorer - Streamlit App

This interactive Streamlit application lets you explore and compare different machine learning models on popular datasets. Users can select from multiple classifiers and datasets, tweak model parameters, and view performance results along with 2D PCA visualizations.

---

## 🌟 Features

- **Dataset Selection**
  - Iris
  - Wine
  - Breast Cancer

- **Classifier Selection**
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest

- **Interactive Parameter Tuning**
  - Adjust parameters like `K`, `C`, `n_estimators`, and `max_depth` using sliders.

- **Visualization**
  - Visualize dataset in 2D using Principal Component Analysis (PCA).
  - Color-coded scatter plot of class distributions.

- **Performance Metrics**
  - Displays classifier accuracy.
  - Optionally view code used for training and evaluation.

---

## 🧠 Model Flow

1. User selects dataset and classifier.
2. App loads dataset using `sklearn.datasets`.
3. User adjusts model parameters via sidebar.
4. App trains the selected model using an 80/20 train-test split.
5. Accuracy of the model is displayed.
6. Dataset is projected to 2D using PCA and visualized.

---

## 🚀 How to Run

### 🛠️ Prerequisites

Install the required libraries:

```bash
pip install streamlit scikit-learn matplotlib numpy
```

### ▶️ Run the App

```bash
streamlit run webapp.py
```

---

## 📁 File Structure

```
webapp.py          # Main Streamlit app file
README.md          # Project description and instructions (you are here)
```


## 📚 Acknowledgements

- Datasets from `sklearn.datasets`
- Built using:
  - Streamlit
  - Scikit-learn
  - Matplotlib
  - NumPy

---

## 📝 Author

Developed by [@Mrusmangoraya](https://github.com/Mrusmangoraya)

