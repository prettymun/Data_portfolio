import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction | Purity Musambi",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark background */
  .stApp { background-color: #0a0a0f; }

  /* Main headings */
  h1, h2, h3 { color: #00e5a0 !important; font-family: 'Arial', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #12121e;
    border-right: 1px solid #1e1e2e;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background-color: #14141e;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem;
  }

  /* Prediction box — risk levels */
  .high-risk {
    background: rgba(255,100,80,0.12);
    border: 1px solid rgba(255,100,80,0.4);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
  }
  .low-risk {
    background: rgba(0,229,160,0.08);
    border: 1px solid rgba(0,229,160,0.35);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
  }
  .risk-label {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
  }
  .risk-sub {
    font-size: 0.85rem;
    color: #9999aa;
  }

  /* Info boxes */
  .info-box {
    background: #14141e;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #7c6aff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #c8c8d8;
  }

  /* Footer */
  .footer {
    text-align: center;
    color: #4a4a5a;
    font-size: 0.72rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #1e1e2e;
  }
</style>
""", unsafe_allow_html=True)


# ── LOAD AND PREPARE DATA (cached so it only runs once) ────────────────────
@st.cache_data
def load_and_train():
    # Load dataset
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)

    # Clean — replace impossible zeros with median
    df_clean = df.copy()
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        median_val = df_clean[col].replace(0, np.nan).median()
        df_clean[col] = df_clean[col].replace(0, median_val)

    # Split
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train all 4 models
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    dt.fit(X_train_scaled, y_train)

    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train_scaled, y_train)

    models = {
        'Logistic Regression': lr,
        'Random Forest':       rf,
        'Decision Tree':       dt,
        'KNN':                 knn,
    }

    # Predictions for metrics
    preds = {
        'Logistic Regression': lr.predict(X_test_scaled),
        'Random Forest':       rf.predict(X_test),
        'Decision Tree':       dt.predict(X_test_scaled),
        'KNN':                 knn.predict(X_test_scaled),
    }

    return df_clean, X, y, scaler, models, preds, X_test, X_test_scaled, y_test


df, X, y, scaler, models, preds, X_test, X_test_scaled, y_test = load_and_train()


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🩺 Diabetes Predictor")
    st.markdown("**By Purity Musambi**")
    st.markdown("[prettymun.github.io](https://prettymun.github.io)")
    st.divider()

    st.markdown("### ⚙️ Choose Model")
    selected_model = st.selectbox(
        "Select a classification model:",
        ['Random Forest', 'Logistic Regression', 'Decision Tree', 'KNN'],
        help="Random Forest gives the best balance of precision and recall"
    )

    st.divider()
    st.markdown("### 📋 Patient Details")
    st.caption("Adjust the sliders to enter patient clinical measurements")

    # Input sliders — ranges based on dataset min/max
    pregnancies    = st.slider("Pregnancies",       0,  17,  3,   help="Number of times pregnant")
    glucose        = st.slider("Glucose (mg/dL)",   50, 200, 120, help="Plasma glucose concentration (2hr OGTT)")
    blood_pressure = st.slider("Blood Pressure (mmHg)", 30, 122, 70, help="Diastolic blood pressure")
    skin_thickness = st.slider("Skin Thickness (mm)", 7, 99, 23,  help="Triceps skin fold thickness")
    insulin        = st.slider("Insulin (µU/mL)",   15, 846, 80,  help="2-hour serum insulin")
    bmi            = st.slider("BMI",               15.0, 67.0, 28.0, step=0.1, help="Body mass index")
    dpf            = st.slider("Diabetes Pedigree", 0.08, 2.42, 0.47, step=0.01, help="Diabetes pedigree function (family history score)")
    age            = st.slider("Age (years)",       21, 81, 33)

    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

    st.divider()
    st.markdown('<div class="footer">Dataset: Pima Indians Diabetes<br>768 patients · 8 features<br>© Purity Musambi 2024</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════

# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("# 🩺 Diabetes Diagnosis Prediction")
st.markdown("**Machine Learning project by Purity Musambi** · Logistic Regression · Random Forest · Decision Tree · KNN")
st.divider()

# ── TABS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Prediction",
    "📊 Model Performance",
    "📈 Data Insights",
    "ℹ️ About"
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Patient Risk Assessment")
    st.caption("Adjust patient values in the sidebar, then click **Predict**")

    if predict_btn:
        # Build input array
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        # Get selected model and predict
        model = models[selected_model]

        # Random Forest uses unscaled data
        if selected_model == 'Random Forest':
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
        else:
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

        confidence = probability[prediction] * 100

        # Display result
        col1, col2 = st.columns([1.2, 1])

        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="high-risk">
                  <div class="risk-label">⚠️ High Risk</div>
                  <div style="font-size:1.4rem; color:#ff6450; font-weight:700; margin:0.5rem 0;">
                    Diabetic — {confidence:.1f}% confidence
                  </div>
                  <div class="risk-sub">
                    This patient shows clinical indicators consistent with diabetes.<br>
                    Recommend immediate glucose tolerance testing.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="low-risk">
                  <div class="risk-label">✅ Low Risk</div>
                  <div style="font-size:1.4rem; color:#00e5a0; font-weight:700; margin:0.5rem 0;">
                    Not Diabetic — {confidence:.1f}% confidence
                  </div>
                  <div class="risk-sub">
                    Clinical indicators do not suggest diabetes at this time.<br>
                    Continue routine monitoring based on age and family history.
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            # Probability bar chart
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor('#0a0a0f')
            ax.set_facecolor('#14141e')
            bars = ax.barh(
                ['Not Diabetic', 'Diabetic'],
                [probability[0] * 100, probability[1] * 100],
                color=['#00e5a0', '#ff6450'],
                edgecolor='none', height=0.5
            )
            for bar, val in zip(bars, [probability[0]*100, probability[1]*100]):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', color='white', fontsize=10)
            ax.set_xlim(0, 115)
            ax.set_xlabel('Probability (%)', color='#9999aa', fontsize=9)
            ax.set_title('Prediction Probabilities', color='#00e5a0',
                         fontsize=10, fontweight='bold')
            ax.tick_params(colors='#9999aa')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e1e2e')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()

        # Patient summary table
        st.markdown("#### 📋 Patient Input Summary")
        input_df = pd.DataFrame({
            'Feature': ['Pregnancies', 'Glucose', 'Blood Pressure',
                        'Skin Thickness', 'Insulin', 'BMI',
                        'Diabetes Pedigree', 'Age'],
            'Value': [pregnancies, glucose, blood_pressure,
                      skin_thickness, insulin, bmi, dpf, age],
            'Unit': ['count', 'mg/dL', 'mmHg', 'mm', 'µU/mL',
                     'kg/m²', 'score', 'years']
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

        # Disclaimer
        st.markdown("""
        <div class="info-box">
        ⚕️ <strong>Clinical Disclaimer:</strong> This tool is a machine learning demonstration
        built for portfolio purposes. It should NOT be used as a substitute for professional
        medical diagnosis. Always consult a qualified healthcare provider.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("👈 Adjust the patient values in the sidebar and click **Predict** to see the result.")

        # Show dataset sample while waiting
        st.markdown("#### 📄 Dataset Preview (first 5 rows)")
        st.dataframe(df.head(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance Comparison")
    st.caption("All 4 models trained on 80% of the data and evaluated on the remaining 20% (154 patients)")

    # Metrics table
    metrics_data = {}
    for name, pred in preds.items():
        if name == 'Random Forest':
            auc = roc_auc_score(y_test, models[name].predict_proba(X_test)[:,1])
        else:
            auc = roc_auc_score(y_test, models[name].predict_proba(X_test_scaled)[:,1])
        metrics_data[name] = {
            'Accuracy':  round(accuracy_score(y_test, pred), 4),
            'Precision': round(precision_score(y_test, pred), 4),
            'Recall':    round(recall_score(y_test, pred), 4),
            'F1 Score':  round(f1_score(y_test, pred), 4),
            'ROC-AUC':   round(auc, 4),
        }

    metrics_df = pd.DataFrame(metrics_data).T
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='rgba(0,229,160,0.25)'),
                 use_container_width=True)

    st.divider()

    # KPI cards for selected model
    st.markdown(f"### 📌 {selected_model} — Detailed Metrics")
    pred = preds[selected_model]
    if selected_model == 'Random Forest':
        auc = roc_auc_score(y_test, models[selected_model].predict_proba(X_test)[:,1])
    else:
        auc = roc_auc_score(y_test, models[selected_model].predict_proba(X_test_scaled)[:,1])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{accuracy_score(y_test, pred):.2%}")
    c2.metric("Precision", f"{precision_score(y_test, pred):.2%}")
    c3.metric("Recall",    f"{recall_score(y_test, pred):.2%}")
    c4.metric("F1 Score",  f"{f1_score(y_test, pred):.2%}")
    c5.metric("ROC-AUC",   f"{auc:.4f}")

    st.divider()

    # Confusion matrix for selected model
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### Confusion Matrix — {selected_model}")
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_facecolor('#14141e')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Diabetic', 'Diabetic'],
                    yticklabels=['Not Diabetic', 'Diabetic'],
                    linewidths=0.5, ax=ax,
                    annot_kws={'size': 14, 'color': 'white'})
        ax.set_ylabel('Actual', color='#9999aa')
        ax.set_xlabel('Predicted', color='#9999aa')
        ax.tick_params(colors='#9999aa')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### All Models — Bar Comparison")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_facecolor('#14141e')
        model_names = list(metrics_data.keys())
        f1_scores   = [metrics_data[m]['F1 Score'] for m in model_names]
        colors = ['#7c6aff', '#00e5a0', '#ffb347', '#ff6450']
        bars = ax.bar(model_names, f1_scores, color=colors,
                      edgecolor='none', width=0.55, alpha=0.85)
        for bar, val in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    color='white', fontsize=9)
        ax.set_ylabel('F1 Score', color='#9999aa')
        ax.set_ylim(0, 1.0)
        ax.set_title('F1 Score Comparison', color='#00e5a0',
                     fontsize=11, fontweight='bold')
        ax.tick_params(colors='#9999aa', axis='x', rotation=10)
        ax.tick_params(colors='#9999aa', axis='y')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e1e2e')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Dataset Insights")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients",    len(df))
    col2.metric("Diabetic",          int(df['Outcome'].sum()))
    col3.metric("Not Diabetic",      int((df['Outcome'] == 0).sum()))
    col4.metric("Features",          len(df.columns) - 1)

    st.divider()

    # Feature importance from Random Forest
    st.markdown("#### 🏆 Feature Importance — Random Forest")
    st.caption("Which clinical measurements are most predictive of diabetes?")

    importances = pd.Series(
        models['Random Forest'].feature_importances_,
        index=X.columns
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#14141e')
    colors_imp = ['#00e5a0' if v == importances.max() else '#7c6aff'
                  for v in importances]
    bars = ax.barh(importances.index, importances.values,
                   color=colors_imp, edgecolor='none', alpha=0.85)
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', color='white', fontsize=9)
    ax.set_xlabel('Importance Score', color='#9999aa')
    ax.tick_params(colors='#9999aa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e1e2e')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Distribution by outcome
    st.markdown("#### 📊 Feature Distributions by Outcome")
    feature_select = st.selectbox("Select a feature to explore:",
                                  [c for c in df.columns if c != 'Outcome'])

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#14141e')
    ax.hist(df[df['Outcome'] == 0][feature_select], bins=30,
            alpha=0.7, color='#00e5a0', label='Not Diabetic', edgecolor='none')
    ax.hist(df[df['Outcome'] == 1][feature_select], bins=30,
            alpha=0.7, color='#ff6450', label='Diabetic', edgecolor='none')
    ax.set_xlabel(feature_select, color='#9999aa')
    ax.set_ylabel('Count', color='#9999aa')
    ax.set_title(f'{feature_select} Distribution by Outcome',
                 color='#00e5a0', fontsize=12, fontweight='bold')
    ax.legend(facecolor='#14141e', edgecolor='#1e1e2e', labelcolor='white')
    ax.tick_params(colors='#9999aa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e1e2e')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🩺 Problem Statement
        Can we predict whether a patient is likely to be diabetic
        based on clinical measurements such as glucose level, BMI,
        age, and insulin? Early prediction helps healthcare providers
        intervene earlier and allocate resources more effectively.

        #### 📦 Dataset
        - **Name:** Pima Indians Diabetes Dataset
        - **Source:** National Institute of Diabetes and Digestive and Kidney Diseases
        - **Records:** 768 patients
        - **Features:** 8 clinical measurements
        - **Target:** Binary — Diabetic (1) or Not Diabetic (0)

        #### 🛠 Tech Stack
        - **Python** — pandas, numpy
        - **Visualisation** — matplotlib, seaborn
        - **Machine Learning** — scikit-learn
        - **Deployment** — Streamlit
        """)

    with col2:
        st.markdown("""
        #### 🤖 Models Trained
        | Model | Best For |
        |---|---|
        | Logistic Regression | Fast baseline, interpretable |
        | Random Forest | Best overall performance |
        | Decision Tree | Visual explainability |
        | KNN | Non-parametric comparison |

        #### 💡 Key Findings
        - **Glucose** is the strongest predictor of diabetes
        - **BMI** is the second most important factor
        - **Age** and **family history (DPF)** are significant risk factors
        - Random Forest achieves the best balance of precision and recall
        - All models significantly outperform random chance (AUC > 0.5)

        #### ⚕️ Disclaimer
        This is a machine learning portfolio project.
        It is **not** intended for clinical use. Always consult
        a qualified healthcare provider for medical decisions.
        """)

    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#4a4a5a; font-size:0.8rem; padding:1rem;">
      Built by <a href="https://prettymun.github.io" style="color:#00e5a0;">Purity Musambi</a>
      &nbsp;·&nbsp; Data Analyst & ML Engineer
      &nbsp;·&nbsp; <a href="https://github.com/prettymun" style="color:#7c6aff;">GitHub</a>
      &nbsp;·&nbsp; <a href="https://www.fiverr.com/users/prettymuna" style="color:#7c6aff;">Fiverr</a>
    </div>
    """, unsafe_allow_html=True)
