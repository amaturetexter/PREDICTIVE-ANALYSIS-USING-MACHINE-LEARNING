# PREDICTIVE ANALYSIS USING MACHINE LEARNING

---

**Company**: CODETECH IT SOLUTIONS

**Name**: SALILA PUNNESHETTY

**Intern ID**: *CT04DH2206*

**Domain**: *DATA ANALYTICS*

**Duration**: 4 Weeks

**Mentor**: NEELA SANTOSH KUMAR

---
# ❤️Heart Disease Prediction using Machine Learning


> **Internship Task-2 – CODTECH IT SOLUTIONS**

> **Project Title**: Predictive Analysis using Machine Learning  

> 📊 **Dataset Used**: [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

---

## Objective

To build a supervised machine learning model that can predict the likelihood of heart disease based on clinical features such as age, cholesterol, blood pressure, and chest pain type using the Heart Disease UCI dataset.

---

## 🗂️ Folder Structure

task-2-predictive-analysis-ml/

├── data/

│ └── heart_disease.csv # Dataset

├── models/

│ └── model.pkl # Trained logistic regression model

├── notebooks/

│ └── predictive_analysis.ipynb # Jupyter notebook with step-by-step code

├── outputs/

│ └── evaluation_report.txt # Evaluation results (accuracy, report, confusion matrix)

├── src/

│ ├── data_loader.py # Loads dataset

│ ├── preprocessing.py # Cleans, encodes, and splits data

│ ├── train_model.py # Trains and saves ML model

│ └── evaluate_model.py # Evaluates model performance

└── README.md


---

## 🔩Technologies Used

- Python 3.12
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- joblib

---

## 🔁 Workflow Overview

### 1. **Load Data**
Using `data_loader.py` to import and view the dataset.

### 2. **Preprocess**
- One-hot encode categorical variables
- Standard scale numeric features
- Split data into training and testing sets

### 3. **Train**
- Logistic Regression model is trained on the preprocessed data
- Model saved as `model.pkl` inside the `models/` folder

### 4. **Evaluate**
- Generates accuracy score, confusion matrix, and classification report
- Stores results in `outputs/evaluation_report.txt`

---
## 📝 Task Description
This project is a part of the Task-2 requirement of the Data Analytics Virtual Internship offered by CODETECH IT SOLUTIONS. The goal was to develop a predictive machine learning model to identify patients at risk of heart disease based on a range of medical features using a real-world dataset from Kaggle.

The dataset includes clinical data such as age, sex, cholesterol levels, blood pressure, chest pain type, fasting blood sugar, ECG results, and more. The target variable is HeartDisease (1 for presence, 0 for absence of disease). This classification task was implemented using the Logistic Regression algorithm from scikit-learn due to its efficiency, simplicity, and suitability for binary classification.

The solution follows a modular and industry-standard structure with clean separation of concerns. The process begins with the loading of data via data_loader.py, which reads the CSV file and returns a pandas DataFrame for further processing. The preprocessing stage in preprocessing.py performs one-hot encoding for categorical features, scales numerical values using StandardScaler, and splits the dataset into training and test sets.

In the train_model.py file, the Logistic Regression model is trained using the training data (X_train, y_train) and saved to disk as model.pkl using joblib. This approach ensures that the trained model can be reused or deployed without retraining from scratch.

The evaluation phase, coded in evaluate_model.py, uses test data to assess model performance. It computes the accuracy score, generates a classification report (precision, recall, F1-score), and a confusion matrix. These results are saved to outputs/evaluation_report.txt and also printed in the notebook. Evaluation results indicate whether the model is overfitting, underfitting, or generalizing well.

The predictive_analysis.ipynb notebook brings all components together for seamless execution. It walks through the entire flow — data import, exploration, preprocessing, training, and evaluation — step-by-step for transparency and reproducibility.

This task provided invaluable hands-on experience in working with real datasets and building end-to-end predictive pipelines using machine learning. It emphasized the importance of data cleaning, exploratory analysis, and proper validation. In addition, it strengthened knowledge of using industry best practices such as modular Python scripting, version control with Git, and result reproducibility.

Overall, this task has prepared me to approach real-world problems with a structured mindset and utilize machine learning tools effectively to derive meaningful insights and predictions.


## How to Run This Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/task-2-predictive-analysis-ml.git
cd task-2-predictive-analysis-ml
pip install -r requirements.txt
cd notebooks
jupyter notebook
[Open predictive_analysis.ipynb and run all cells sequentially.]
```
## OUTPUT:
<img width="834" height="488" alt="Image" src="https://github.com/user-attachments/assets/1f6531dc-6eb9-4fbd-997a-727e340aec1d" />

##  Conclusion
This machine learning model demonstrates good predictive accuracy for identifying heart disease based on basic patient health metrics. It can assist in early intervention and clinical decision-making.
