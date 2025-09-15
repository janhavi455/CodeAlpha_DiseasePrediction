# Disease Prediction from Medical Data 🩺

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-green)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

A machine learning project that predicts **Diabetes, Heart Disease, and Breast Cancer** from medical datasets.  
This repo demonstrates **end-to-end ML workflows**: preprocessing, training, evaluation, and an **interactive web app** for real-time disease prediction.


## 🚀 Features
- Preprocessing: Missing values, feature scaling, categorical encoding.
- ML Models: Logistic Regression, Random Forest, XGBoost.
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
- Streamlit App: Choose disease type → Enter medical data → Get prediction.
- Modular code: Easy to add more diseases in the future.


## 🗂 Repository Structure
```plaintext
CodeAlpha_DiseasePrediction/
│
├── data/ #preprocessed dataset for heart
│   ├── processed.cleaveland.data
│
├── models/                      
│   ├── breast_features.pkl
│   ├── breast_model.pkl   
│   ├── breast_scaler.pkl     
│   ├── diabetes_features.pkl
│   ├── diabetes_model.pkl  
│   ├── diabetes_scaler.pkl
│   ├── heart_features.pkl
│   ├── heart_model.pkl   
│   ├── heart_scaler.pkl
│
├── notebooks/                    
│   ├── data_prep_breastCancer.ipynb   # Jupyter notebook for exploration and experiments
│   ├── data_prep_diabetes.ipynb
│   ├── data_heart.ipynb
│   ├── data_prediction.ipynb
│
├── app.py                       # Streamlit app for frontend
├── requirements.txt
├── README.md
```


## 📊 Datasets Used
- **Diabetes Dataset** (PIMA Indian Diabetes Dataset)
- **Heart Disease Dataset** (UCI Cleveland Heart Disease Dataset)
- **Breast Cancer Dataset** (Wisconsin Breast Cancer Dataset)

Each dataset contains patient health metrics like blood pressure, glucose level, cholesterol, tumor size, etc., which are used to predict disease outcomes.


## ⚙️ Installation
1. **Clone the repo:**
```bash
git clone https://github.com/yourusername/disease-prediction-ml.git
cd disease-prediction-ml
```
2. **Install dependencies**
```
pip install -r requirements.txt
```
3. **Run the Streamlit app:**
```
streamlit run app.py
```


## Usage
Open the Streamlit app in browser.<br>
Select Disease Type (Diabetes / Heart Disease / Breast Cancer).<br>
Enter patient medical details.<br>
View the prediction (Positive / Negative).<br>


## Tech Stack
Python 3.10<br>
Data Handling: Pandas, NumPy<br>
ML Models: Scikit-learn, XGBoost<br>
Visualization: Matplotlib, Seaborn<br>
Web App: Streamlit<br>


## **Author**<br>
Janhavi Sakhare<br>
B.Tech Student | AI & Data Science Enthusiast<br>
GitHub: janhavi455<br>
