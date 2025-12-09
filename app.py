import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# ------------------------------------------------
# Page config MUST be the first Streamlit command
# ------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    page_icon="💳",
)

# -------------------
# Load trained model
# -------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Features used by the model (order matters)
FEATURE_COLUMNS = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount', 'Time_hours'
]

# -------------------
# App header
# -------------------
st.title("💳 Credit Card Fraud Detection")
st.markdown(
    """
This tool helps you check whether a credit card transaction is likely **genuine** or **fraudulent**  
based on historical transaction patterns.

**Key features:**
- 🧑‍💻 Simple, user-friendly interface  
- ⚡ Real-time transaction checking  
- 📊 Prediction confidence score for each result  
    """
)

with st.expander("ℹ️ About this app"):
    st.write(
        """
The model looks at transformed transaction features (V1–V28), the transaction **Amount**,  
and the **Time (in hours)** since the first recorded transaction.  

**Classes:**
- `0` → Genuine transaction  
- `1` → Fraudulent transaction  
        """
    )

# -------------------
# Sidebar
# -------------------
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio(
    "Choose how you want to check transactions:",
    ["Single Transaction", "Batch (CSV Upload)"]
)


st.sidebar.markdown("---")



# -------------------
# Helper for prediction
# -------------------
def predict_single(features_dict):
    """
    features_dict: dict {col_name: value}
    """
    x = np.array([[features_dict[col] for col in FEATURE_COLUMNS]])
    proba = model.predict_proba(x)[0]
    pred = int(model.predict(x)[0])
    confidence = float(proba[pred])
    return pred, confidence


def predict_batch(df):
    """
    df: DataFrame containing FEATURE_COLUMNS
    """
    x = df[FEATURE_COLUMNS].values
    proba = model.predict_proba(x)
    preds = model.predict(x).astype(int)
    confidences = proba[np.arange(len(preds)), preds]
    return preds, confidences


# -------------------
# Single transaction mode
# -------------------
if input_mode == "Single Transaction":
    st.subheader("🔍 Check a Single Transaction")

    st.markdown("Enter the transaction details below:")

    # Layout inputs in 3 columns
    cols = st.columns(3)
    inputs = {}

    # Numeric inputs for V1–V28
    for i, feature in enumerate(FEATURE_COLUMNS):
        col_index = i % 3
        with cols[col_index]:
            # Provide sensible default 0.0 for transformed features
            if feature.startswith("V"):
                inputs[feature] = st.number_input(
                    feature,
                    value=0.0,
                    step=0.1,
                    format="%.4f"
                )
            elif feature == "Amount":
                inputs[feature] = st.number_input(
                    "Amount (in currency units)",
                    value=0.0,
                    min_value=0.0,
                    step=1.0,
                    format="%.2f"
                )
            elif feature == "Time_hours":
                inputs[feature] = st.number_input(
                    "Time since first transaction (hours)",
                    value=0.0,
                    min_value=0.0,
                    step=0.1,
                    format="%.3f"
                )

    check_button = st.button("🚀 Check Transaction")

    if check_button:
        pred, confidence = predict_single(inputs)
        label = "Genuine" if pred == 0 else "Fraudulent"

        if pred == 0:
            st.success(f"✅ The transaction is predicted as **{label}**.")
        else:
            st.error(f"⚠️ The transaction is predicted as **{label}**.")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence * 100:.2f} %"
        )

# -------------------
# Batch mode
# -------------------
else:
    st.subheader("📂 Check Multiple Transactions (Batch Mode)")

    st.markdown(
        """
Upload a CSV file containing the required columns:

`V1`–`V28`, `Amount`, and `Time_hours`  
(You can create `Time_hours` as `Time / 3600` from your original dataset.)
"""
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
            if missing_cols:
                st.error(
                    "The uploaded file is missing the following required columns:\n\n"
                    + ", ".join(missing_cols)
                )
            else:
                preds, confidences = predict_batch(df)

                df_result = df.copy()
                df_result["Prediction"] = np.where(preds == 0, "Genuine", "Fraudulent")
                df_result["Confidence"] = (confidences * 100).round(2)

                st.success("✅ Predictions generated successfully!")

                st.write("### Preview of results")
                st.dataframe(df_result.head(50))

                # Summary
                total = len(df_result)
                genuine_count = (preds == 0).sum()
                fraud_count = (preds == 1).sum()

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Transactions", total)
                col_b.metric("Predicted Genuine", genuine_count)
                col_c.metric("Predicted Fraudulent", fraud_count)

                # Download button
                csv_download = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download full results as CSV",
                    data=csv_download,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

