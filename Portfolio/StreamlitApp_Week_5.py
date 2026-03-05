import os, sys, warnings
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
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from imblearn.pipeline import Pipeline

import shap


# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import get_bitcoin_historical_prices

# Access the secrets
aws_id                = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret            = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token             = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket            = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint_bitcoin  = st.secrets["aws_credentials"]["AWS_ENDPOINT"]


# AWS Session Management
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

# ── Feature Engineering Helper ────────────────────────────────────────────────
# Computes the 4 technical indicators the model was trained on from a Close
# price series. The user provides a single Close price; we append it to the
# historical series so the rolling windows have enough data to produce values.

def compute_features(close_series: pd.Series) -> pd.DataFrame:
    """Given a Close price Series, return a DataFrame of the 4 model features."""
    df = pd.DataFrame({'Close': close_series})

    # EMA_10 — Exponential Moving Average (10-period)
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # RSI_10 — Relative Strength Index (10-period)
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=9, adjust=False).mean()
    avg_loss = loss.ewm(com=9, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI_10'] = 100 - (100 / (1 + rs))

    # ROC_10 — Rate of Change (10-period, %)
    df['ROC_10'] = df['Close'].pct_change(periods=10) * 100

    # MOM_10 — Momentum (10-period, USD)
    df['MOM_10'] = df['Close'].diff(periods=10)

    return df[['EMA_10', 'RSI_10', 'ROC_10', 'MOM_10']]


# Data & Model Configuration
df_prices = get_bitcoin_historical_prices()

# Dynamic bounds derived from historical prices
MIN_VAL     = 0.5  * df_prices.iloc[:, 0].min()
MAX_VAL     = 2.0  * df_prices.iloc[:, 0].max()
DEFAULT_VAL = df_prices.iloc[:, 0].mean()

# CHANGED: keys and inputs now reflect the 4 engineered features, not raw Close
FEATURE_COLS = ['EMA_10', 'RSI_10', 'ROC_10', 'MOM_10']

MODEL_INFO = {
    "endpoint": aws_endpoint_bitcoin,
    "explainer": 'explainer_bitcoin.shap',
    "pipeline":  'finalized_bitcoin_model.tar.gz',
    # Single user-facing input — Close price — features are computed from it
    "inputs": [
        {"name": "Close Price", "type": "number",
         "min": MIN_VAL, "max": MAX_VAL, "default": DEFAULT_VAL, "step": 100.0}
    ]
}


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
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(f"{joblib_file}")


def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return joblib.load(f)  # CHANGED: use joblib.load (saved with joblib.dump in notebook)


# Prediction Logic
def call_model_api(feature_df: pd.DataFrame):
    """Send the 4-feature row to the SageMaker endpoint and return a signal."""
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        # CHANGED: send the feature array (shape 1x4), not the raw Close series
        input_array = feature_df[FEATURE_COLS].values.astype(np.float32)
        raw_pred    = predictor.predict(input_array)
        pred_val    = int(pd.DataFrame(raw_pred).values[-1][0])
        mapping     = {-1: "SELL 📉", 0: "HOLD ⏸️", 1: "BUY 📈"}
        return mapping.get(pred_val, str(pred_val)), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# Local Explainability
def display_explanation(feature_df: pd.DataFrame, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    full_pipeline         = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    # CHANGED: preprocessing = all steps except sampler and model (last 2 steps)
    preprocessing_pipeline = Pipeline(steps=[
        s for s in full_pipeline.steps if s[0] not in ('sampler', 'model')
    ])

    # CHANGED: transform the 4-feature row directly (no more Close-price concat)
    input_array       = feature_df[FEATURE_COLS].values.astype(np.float32)
    input_transformed = preprocessing_pipeline.transform(input_array)
    shap_values       = explainer(pd.DataFrame(input_transformed, columns=FEATURE_COLS))

    # Handle both 2D (binary) and 3D (multi-class OvR) SHAP value shapes
    if len(shap_values.shape) == 3:
        vals        = shap_values[0, :, 1]   # class index 1 = Buy
        base_val    = explainer.expected_value[1]
    else:
        vals        = shap_values[0]
        base_val    = explainer.expected_value

    exp = shap.Explanation(
        values=vals.values if hasattr(vals, 'values') else vals,
        base_values=base_val,
        data=input_transformed[0],
        feature_names=FEATURE_COLS
    )

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(exp, show=False)
    st.pyplot(fig)

    top_feature = pd.Series(exp.values, index=exp.feature_names).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Deployment Compiler", layout="wide")
st.title("👨‍💻 ML Deployment Compiler")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols        = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                max_value=inp['max'],
                value=inp['default'],
                step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    # CHANGED: append user's Close price to historical series, then compute features
    close_price  = user_inputs["Close Price"]
    close_series = pd.concat([
        df_prices.iloc[:, 0],
        pd.Series([close_price])
    ], ignore_index=True)

    feature_df = compute_features(close_series)

    # Take the last row — that corresponds to the user's entered Close price
    input_row = feature_df.iloc[[-1]].reset_index(drop=True)

    st.write("**Computed Features:**")
    st.dataframe(input_row)

    res, status = call_model_api(input_row)

    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_row, session, aws_bucket)
    else:
        st.error(res)


