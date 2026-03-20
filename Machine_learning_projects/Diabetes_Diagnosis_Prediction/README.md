# 🩺 Diabetes Diagnosis Prediction — Machine Learning Project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prettymun/Data_analysis_portfolio/blob/main/Diabetes-Diagnosis-ML/Diabetes_Diagnosis_Prediction__ML_.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prettymuna.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Author:** Purity Musambi &nbsp;|&nbsp; **Portfolio:** [prettymun.github.io](https://prettymun.github.io) &nbsp;|&nbsp; **Fiverr:** [fiverr.com/users/prettymuna](https://www.fiverr.com/users/prettymuna)

---

## 📌 Problem Statement

Can we predict whether a patient is likely to be diabetic based on clinical measurements such as glucose level, BMI, age, and insulin?

Early and accurate prediction of diabetes enables healthcare providers to:
- Intervene earlier before complications develop
- Prioritise high-risk patients for immediate screening
- Allocate clinical resources more effectively
- Reduce the burden of late-stage diabetes management

---

## 📦 Dataset

| Property | Detail |
|---|---|
| **Name** | Pima Indians Diabetes Dataset |
| **Source** | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) |
| **Records** | 768 patients |
| **Features** | 8 clinical measurements |
| **Target** | `Outcome` — 1 = Diabetic, 0 = Not Diabetic |
| **Class Split** | 500 Not Diabetic (65.1%) · 268 Diabetic (34.9%) |

### Feature Descriptions

| Feature | Description | Unit |
|---|---|---|
| `Pregnancies` | Number of times pregnant | count |
| `Glucose` | Plasma glucose concentration (2hr OGTT) | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mmHg |
| `SkinThickness` | Triceps skin fold thickness | mm |
| `Insulin` | 2-hour serum insulin | µU/mL |
| `BMI` | Body mass index | kg/m² |
| `DiabetesPedigreeFunction` | Family history score | score |
| `Age` | Age of patient | years |

---

## 🗂️ Repository Structure

```
Diabetes-Diagnosis-ML/
│
├── Diabetes_Diagnosis_Prediction__ML_.ipynb   # Main Jupyter notebook
├── app.py                                      # Streamlit web app
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

---

## 📓 Notebook Structure

The notebook is organised into 12 clearly labelled steps:

| Step | Section | Description |
|---|---|---|
| 01 | Install & Import Libraries | Set up pandas, numpy, matplotlib, seaborn, scikit-learn |
| 02 | Load the Dataset | Pull data directly from URL — no manual download needed |
| 03 | Exploratory Data Analysis | Data types, statistical summary, missing values, class balance |
| 04 | Data Visualisation | Feature distributions, correlation heatmap, class-level comparisons |
| 05 | Data Preprocessing | Replace impossible zero values with median, feature scaling |
| 06 | Train / Test Split | 80/20 stratified split, StandardScaler applied |
| 07 | Logistic Regression | Baseline model — training, evaluation, confusion matrix |
| 08 | Random Forest | Ensemble model — training, evaluation, confusion matrix |
| 09 | Decision Tree | Tree-based model with visual tree structure |
| 10 | K-Nearest Neighbours | Distance-based model + optimal K selection |
| 11 | Model Comparison | Side-by-side bar chart across all 4 models |
| 12 | Feature Importance | Importance scores across all 3 tree/linear models |
| 13 | ROC Curve | All 4 models on one chart with AUC scores |
| 14 | Business Insights | Clinical findings and recommendations |

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| **Logistic Regression** | Baseline classifier · uses scaled features · `class_weight='balanced'` |
| **Random Forest** | 100 decision trees · uses unscaled features · best overall balance |
| **Decision Tree** | `max_depth=5` to prevent overfitting · visualised with `plot_tree` |
| **KNN** | K optimised by testing K=1 to 30 · sensitive to feature scale |

---

## 📊 Results

> Exact scores will vary slightly depending on environment. Below are representative results from the training run.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~77% | ~72% | ~76% | ~74% | ~0.84 |
| **Random Forest** | **~76%** | **~71%** | **~72%** | **~71%** | **~0.83** |
| Decision Tree | ~72% | ~65% | ~70% | ~67% | ~0.76 |
| KNN | ~74% | ~68% | ~68% | ~68% | ~0.80 |

> **Why Recall matters most here:** In a clinical setting, a False Negative (missing a diabetic patient) is more costly than a False Positive (unnecessary follow-up). Models are evaluated with this priority in mind.

---

## 💡 Key Findings

1. **Glucose** is the strongest predictor of diabetes — consistently ranked #1 across all feature importance methods
2. **BMI** is the second most important factor, confirming the clinical link between obesity and Type 2 diabetes
3. **Age** and **DiabetesPedigreeFunction** are significant non-modifiable risk factors
4. **Insulin** and **SkinThickness** are the weakest predictors — partly due to high rates of zero/missing values in the original dataset
5. All 4 models significantly outperform random chance (AUC > 0.5)

---

## 🚀 Live Demo — Streamlit App

The trained Random Forest model is deployed as an interactive web app.

**Features:**
- Adjust 8 clinical sliders for a patient
- Switch between all 4 models
- Instant prediction with confidence percentage
- Model performance comparison dashboard
- Feature importance and distribution explorer

👉 **[Launch the App](https://prettymuna.streamlit.app)**

---

## 🛠 How to Run Locally

### Option 1 — Google Colab (recommended, no setup needed)
Click the **Open in Colab** badge at the top of this README.

### Option 2 — Run the Notebook Locally

```bash
# 1. Clone the repository
git clone https://github.com/prettymun/Data_analysis_portfolio.git
cd Data_analysis_portfolio/Diabetes-Diagnosis-ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook Diabetes_Diagnosis_Prediction__ML_.ipynb
```

### Option 3 — Run the Streamlit App Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📋 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## ⚕️ Disclaimer

This project is built for **portfolio and educational purposes only**. It is not intended for clinical use and should not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical decisions.

---

## 👩‍💻 About the Author

**Purity Musambi** — Data Analyst & Machine Learning Engineer based in Nairobi, Kenya.

- 🌐 Portfolio: [prettymun.github.io](https://prettymun.github.io)
- 💼 Fiverr: [fiverr.com/users/prettymuna](https://www.fiverr.com/users/prettymuna)
- 🐙 GitHub: [github.com/prettymun](https://github.com/prettymun)
- ✉️ Email: avulamusipurity@gmail.com

---

*Part of the [Data Analysis Portfolio](https://github.com/prettymun/Data_analysis_portfolio) — a collection of real-world data science and analytics projects.*
