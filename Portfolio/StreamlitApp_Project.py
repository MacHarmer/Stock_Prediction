import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import joblib
import tarfile
import tempfile
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.pipeline import Pipeline
import shap
from joblib import load

warnings.simplefilter("ignore")

# ── Path Configuration ────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ── Load X_train baseline (used to fill non-input columns) ───────────────────
file_path = os.path.join(project_root, 'Portfolio/X_train.csv')
print(file_path)
dataset = pd.read_csv(file_path)
dataset = dataset.drop(['Unnamed: 0'], axis=1)
#dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]


# ── AWS Secrets ───────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── Model Configuration ───────────────────────────────────────────────────────
# Top 5 features from SHAP (C13, C14, C1, card1_mean_amt, TransactionAmt)
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_fraud.shap",
    "pipeline"  : "fraud_model_final.tar.gz",
    "keys"      : ['C13', 'C14', 'C1', 'card1_mean_amt', 'TransactionAmt'],
    "inputs"    : [
        {
            "name"    : "C13",
            "label"   : "C13 — Count of Addresses Associated with Card",
            "type"    : "number",
            "min"     : 0.0,
            "max"     : 3000.0,
            "default" : 1.0,
            "step"    : 1.0,
            "help"    : "Number of addresses linked to this payment card (higher = more suspicious)"
        },
        {
            "name"    : "C14",
            "label"   : "C14 — Count of Billing Address Matches",
            "type"    : "number",
            "min"     : 0.0,
            "max"     : 3000.0,
            "default" : 1.0,
            "step"    : 1.0,
            "help"    : "Number of times billing address was matched to this card"
        },
        {
            "name"    : "C1",
            "label"   : "C1 — Count of Payment Addresses",
            "type"    : "number",
            "min"     : 0.0,
            "max"     : 3000.0,
            "default" : 1.0,
            "step"    : 1.0,
            "help"    : "Number of payment addresses associated with this card"
        },
        {
            "name"    : "card1_mean_amt",
            "label"   : "Card Average Transaction Amount ($)",
            "type"    : "number",
            "min"     : 0.0,
            "max"     : 10000.0,
            "default" : 100.0,
            "step"    : 0.01,
            "help"    : "Historical average transaction amount for this card"
        },
        {
            "name"    : "TransactionAmt",
            "label"   : "Transaction Amount ($)",
            "type"    : "number",
            "min"     : 0.0,
            "max"     : 10000.0,
            "default" : 68.5,
            "step"    : 0.01,
            "help"    : "Dollar amount of this transaction"
        },
    ]
}

# ── AWS Session ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Load Pipeline ─────────────────────────────────────────────────────────────
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]
    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        pkl_file = [f for f in tar.getnames() if f.endswith('.pkl')][0]
    return joblib.load(pkl_file)

# ── Load SHAP Explainer ───────────────────────────────────────────────────────
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return load(f)

# ── Prediction ────────────────────────────────────────────────────────────────
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping  = {0: "✅ Legitimate", 1: "🚨 Fraud"}
        return mapping.get(int(pred_val), "Unknown"), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# ── SHAP Explanation ──────────────────────────────────────────────────────────
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )
    best_pipeline         = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-3])
    input_df              = pd.DataFrame(input_df)
    input_df_transformed  = preprocessing_pipeline.transform(input_df)
    feature_names         = best_pipeline[:-2].get_feature_names_out()
    input_df_transformed  = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values           = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1])
    st.pyplot(fig)

    top_feature = (
        pd.Series(shap_values[0, :, 1].values,
                  index=shap_values[0, :, 1].feature_names)
        .abs().idxmax()
    )
    st.info(f"**Key Fraud Driver:** The most influential factor in this prediction was **{top_feature}**.")

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection — IEEE-CIS", layout="wide")

st.title("🔒 Real-Time Fraud Detection")
st.markdown(
    "Enter transaction details below. The model will predict whether the transaction "
    "is **legitimate or fraudulent** based on the top features identified by SHAP analysis."
)

st.markdown("---")

with st.form("pred_form"):
    st.subheader("Transaction Inputs")
    st.caption("Fields are based on the top 5 most predictive features (SHAP analysis)")

    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                label=inp['label'],
                min_value=float(inp['min']),
                max_value=float(inp['max']),
                value=float(inp['default']),
                step=float(inp['step']),
                help=inp.get('help', '')
            )

    submitted = st.form_submit_button("🔍 Run Fraud Check", use_container_width=True)

# ── On Submit ─────────────────────────────────────────────────────────────────
if submitted:
    # Start from first row of X_train, override with user inputs
    original = dataset.iloc[0].to_dict()#orient='records')#[0]
    original.update(user_inputs)
    #input_df = pd.DataFrame([original])
    print(original)

    with st.spinner("Running prediction..."):
        res, status = call_model_api([original])

    if status == 200:
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Prediction Result", res)
            if "Fraud" in res:
                st.error("⚠️ This transaction has been flagged for review.")
            else:
                st.success("Transaction appears legitimate.")
        with col2:
            st.markdown("**Top Input Values:**")
            st.dataframe(
                pd.DataFrame([user_inputs]).T.rename(columns={0: "Value"}),
                use_container_width=True
            )
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)

st.markdown("---")
st.caption("IEEE-CIS Fraud Detection Model | Mac Harmer | INSC 30273")
