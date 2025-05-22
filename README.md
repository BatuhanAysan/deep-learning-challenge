# 📊 Alphabet Soup Charity – Deep Learning Model Report

## 🧭 Overview of the Analysis

The purpose of this analysis is to develop a deep learning model capable of predicting whether an applicant funded by the nonprofit organization **Alphabet Soup** is likely to be successful. By training a binary classifier using TensorFlow/Keras, the organization can better allocate funding resources to applicants with a higher probability of success.

---

## 🔎 Data Preprocessing

### 🎯 Target Variable:
- `IS_SUCCESSFUL` (1: successful use of funds, 0: not successful)

### 🧠 Feature Variables:
All other columns after dropping irrelevant ones were used, including:
- `APPLICATION_TYPE`
- `AFFILIATION`
- `CLASSIFICATION`
- `USE_CASE`
- `ORGANIZATION`
- `INCOME_AMT`
- `ASK_AMT` *(log-transformed)*
- One-hot encoded categorical variables via `pd.get_dummies`

### 🔻 Dropped Columns:
- `EIN` – unique identifier, not informative  
- `NAME` – textual data, not useful for classification

### 🧹 Rare Category Handling:
- Application types with <500 entries were grouped as `"Other"`
- Classifications with <1000 entries were grouped as `"Other"`

---

## 🧠 Neural Network Architecture

### Final Model Summary:
- **Input Layer:** 116 features (after one-hot encoding)
- **Hidden Layer 1:** 64 neurons, ReLU activation
- **Hidden Layer 2:** 32 neurons, ReLU activation
- **Hidden Layer 3:** 16 neurons, ReLU activation
- **Output Layer:** 1 neuron, Sigmoid activation
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Epochs:** 100 (EarlyStopping applied)
- **Validation Split:** 0.2

### 📉 Regularization:
- `EarlyStopping` used with `patience=5` and `restore_best_weights=True` to prevent overfitting

---

## 📈 Results

### ✅ Training Accuracy (Final Epochs):
- Peaked around **0.75+**

### ✅ Test Accuracy:
- **0.7301** (~73.01%)

### 📉 Loss:
- Test set loss: **0.5544**

---

## 🧪 Optimization Strategies Attempted

| Attempt | Description |
|--------|-------------|
| 1 | ReLU activation instead of Tanh |
| 2 | Increased hidden layers (2 → 3) |
| 3 | Adjusted units: 64 → 32 → 16 |
| 4 | Dropout removed for better learning |
| 5 | Log transformation of `ASK_AMT` |
| 6 | EarlyStopping to avoid overfitting |
| 7 | Validation split introduced to improve generalization |

Despite these optimizations, the highest test accuracy achieved was **~73%**.

---

## 🧠 Recommendation

Given that the deep learning model could not reach the desired accuracy of 75% even after multiple optimizations, a better approach may be to apply **ensemble tree-based methods** like:

- Random Forest
- Gradient Boosting (e.g., XGBoost)

These models generally perform better on **tabular, categorical-heavy datasets** like this one, and require less tuning than deep neural networks.
