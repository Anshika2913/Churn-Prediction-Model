# 🧠 Customer Churn Prediction using ANN

A deep learning model built using Artificial Neural Networks (ANN) to predict whether a bank customer is likely to churn. This helps banks take proactive steps to retain customers.

---

## 🔍 Project Overview

Customer churn is a major challenge for banks. This project uses an Artificial Neural Network to classify which customers are likely to leave the bank based on historical data. The model aims to assist customer retention teams in identifying at-risk clients early.

---

## 📊 Features

- Deep Learning with ANN (Keras/TensorFlow)
- Data Preprocessing and Feature Scaling
- Model Training and Evaluation
- Streamlit App for user-friendly interface (optional)
- Bank customer dataset with real-world features

---

## 🧠 Model Details

- Framework: **Keras** with **TensorFlow** backend
- Layers: Input layer → Hidden layers (ReLU) → Output layer (Sigmoid)
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

---

## 🧩 Technologies Used
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- TensorFlow & Keras
- Streamlit (optional)

---
## 🗃️ Dataset

- Source: [Kaggle - Churn Modeling Dataset](https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers)
- Records: 10,000 customers
- Features:
  - Credit Score
  - Geography
  - Gender
  - Age
  - Tenure
  - Balance
  - Number of Products
  - Has Credit Card
  - Is Active Member
  - Estimated Salary
- Target: `Exited` (0 = No churn, 1 = Churn)

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bank-churn-ann.git
   cd bank-churn-ann
2. **Create virtual environment**
      python -m venv venv
      venv\Scripts\activate
3. **Install dependencies**
      pip install -r requirements.txt
4. **Run the model**
      python churn_ann_model.py
5. **Run the Streamlit app**
      streamlit run app.py
