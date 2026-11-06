import streamlit as st
import numpy as np
import pandas as pd
import joblib
import hashlib
import time
import traceback
import os
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from samples import ATTACK_SAMPLE

st.set_page_config(
    page_title="CyberShield IDS",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] { background-color: #f4f6f9; }

.main .block-container {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 2rem 2.5rem;
    border: 1px solid #e0e4ea;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

[data-testid="stSidebar"] { background-color: #1a2233 !important; }
[data-testid="stSidebar"] * { color: #c8d4e8 !important; }

h1 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; color: #1a2233 !important; font-size: 1.8rem !important; margin-bottom: 0.25rem !important; }
h2, h3, h4 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; color: #1a2233 !important; }
p, label, div { color: #3a4a5c; }

.stButton > button {
    background-color: #ffffff !important; color: #1a2233 !important;
    border: 1.5px solid #1a2233 !important; border-radius: 6px;
    font-family: 'Inter', sans-serif; font-weight: 600;
    font-size: 0.88rem; padding: 0.45rem 1rem; transition: all 0.2s ease;
}
.stButton > button:hover {
    background-color: #1a2233 !important; color: #ffffff !important;
    border-color: #1a2233 !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: #f8f9fb !important; color: #1a2233 !important;
    border: 1px solid #d0d7e2 !important; border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.88rem !important;
}

[data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #e0e4ea; }
[data-baseweb="tab"] {
    background-color: transparent !important; border: none !important;
    color: #6b7a90 !important; font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important; font-size: 0.9rem !important; padding: 0.5rem 1rem !important;
}
[aria-selected="true"] {
    background-color: transparent !important; color: #1a2233 !important;
    border-bottom: 2px solid #1a2233 !important; font-weight: 600 !important;
}

.card { background-color: #ffffff; border: 1px solid #e0e4ea; border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }

.attack-banner { background-color: #fef2f2; border: 1px solid #fca5a5; border-left: 5px solid #dc2626; border-radius: 8px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.normal-banner { background-color: #f0fdf4; border: 1px solid #86efac; border-left: 5px solid #16a34a; border-radius: 8px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }

.stat-box { background-color: #ffffff; border: 1px solid #e0e4ea; border-radius: 10px; padding: 1.2rem; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
.stat-number { font-family: 'Inter', sans-serif; font-size: 2rem; font-weight: 700; color: #1a2233; display: block; }
.stat-label { font-size: 0.78rem; color: #8a96a8; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 0.25rem; display: block; }

.section-title { font-family: 'Inter', sans-serif; font-size: 0.78rem; font-weight: 600; color: #8a96a8; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #e8ecf2; padding-bottom: 0.5rem; margin-bottom: 1rem; }

.sample-btn-attack > button { background-color: #dc2626 !important; color: #ffffff !important; border-color: #dc2626 !important; }
.sample-btn-attack > button:hover { background-color: #b91c1c !important; border-color: #b91c1c !important; }
.sample-btn-normal > button { background-color: #16a34a !important; color: #ffffff !important; border-color: #16a34a !important; }
.sample-btn-normal > button:hover { background-color: #15803d !important; border-color: #15803d !important; }
</style>
""", unsafe_allow_html=True)

USERS_FILE = "users.json"
PREDICTIONS_FILE = "predictions.json"
CSV_FILE = "balanced_sampled.csv"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    default = {"admin": hashlib.sha256("password".encode()).hexdigest()}
    save_users(default)
    return default

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def load_predictions():
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            return json.load(f)
    return []

def save_prediction(entry):
    logs = load_predictions()
    logs.append(entry)
    logs = logs[-500:]
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(logs, f, indent=2)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "current_row_index" not in st.session_state:
    st.session_state.current_row_index = 0
if "lime_training_data" not in st.session_state:
    st.session_state.lime_training_data = None
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Dashboard"
if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = None
if "sample_load_count" not in st.session_state:
    st.session_state.sample_load_count = 0

label_mapping = {
    "Normal": 0, "Generic": 1, "Exploits": 2, "Fuzzers": 3,
    "DoS": 4, "Reconnaissance": 5, "Analysis": 6,
    "Backdoor": 7, "Shellcode": 8, "Worms": 9
}
inv_label_mapping = {v: k for k, v in label_mapping.items()}
target_names = [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])]

feature_names = [
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"
]

@st.cache_resource(show_spinner=False)
def load_all_models():
    cnn_model, lstm_model, scaler = None, None, None
    loaded = False
    for ext in [".keras", ".h5"]:
        try:
            cnn_model = load_model(f"cnn_model{ext}")
            lstm_model = load_model(f"lstm_model{ext}")
            loaded = True
            break
        except Exception:
            continue
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception:
        scaler = StandardScaler()
    return cnn_model, lstm_model, scaler, loaded

@st.cache_data(show_spinner=False)
def load_csv():
    try:
        return pd.read_csv(CSV_FILE)
    except Exception:
        return None

def get_lime_training_data(df):
    if st.session_state.lime_training_data is not None:
        return st.session_state.lime_training_data
    if df is not None:
        rows = []
        for _, row in df.iterrows():
            vals = []
            for f in feature_names:
                if f in row:
                    try:
                        vals.append(float(row[f]))
                    except:
                        vals.append(0.0)
                else:
                    vals.append(0.0)
            rows.append(vals)
        data = np.array(rows, dtype=float)
        st.session_state.lime_training_data = data
        return data
    return np.random.rand(200, len(feature_names))

def make_prediction(inputs_array, cnn_model, lstm_model, scaler, already_scaled=False):
    try:
        try:
            inputs_norm = inputs_array if already_scaled else scaler.transform(inputs_array)
        except:
            inputs_norm = inputs_array
        lstm_input = np.tile(inputs_norm, (10, 1)).reshape(1, 10, -1)
        cnn_probs = cnn_model.predict(inputs_norm.reshape(1, inputs_norm.shape[1], 1), verbose=0)
        cnn_idx = int(np.argmax(cnn_probs, axis=1)[0])
        cnn_conf = float(np.max(cnn_probs, axis=1)[0])
        cnn_result = inv_label_mapping.get(cnn_idx, "Unknown")
        lstm_probs = lstm_model.predict(lstm_input, verbose=0)
        lstm_idx = int(np.argmax(lstm_probs, axis=1)[0])
        lstm_conf = float(np.max(lstm_probs, axis=1)[0])
        lstm_result = inv_label_mapping.get(lstm_idx, "Unknown")
        return {
            "cnn_result": cnn_result, "cnn_confidence": cnn_conf,
            "lstm_result": lstm_result, "lstm_confidence": lstm_conf,
            "inputs_norm": inputs_norm, "success": True
        }
    except Exception as e:
        return {
            "cnn_result": "Error", "cnn_confidence": 0.0,
            "lstm_result": "Error", "lstm_confidence": 0.0,
            "inputs_norm": inputs_array, "success": False, "error": str(e)
        }

def generate_lime(inputs_norm, lime_training_data, cnn_model):
    try:
        explainer = LimeTabularExplainer(
            training_data=lime_training_data,
            feature_names=feature_names,
            class_names=target_names,
            mode="classification"
        )
        def cnn_predict_fn(data):
            try:
                return cnn_model.predict(data.reshape(data.shape[0], data.shape[1], 1), verbose=0)
            except:
                return np.ones((data.shape[0], len(target_names))) / len(target_names)
        explanation = explainer.explain_instance(data_row=inputs_norm[0], predict_fn=cnn_predict_fn)
        html = explanation.as_html()
        top_features = explanation.as_list()
        return html, top_features, True
    except Exception as e:
        return f"<p>LIME error: {str(e)}</p>", [], False

def sign_in_page():
    st.markdown("<h1>CyberShield IDS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7a90; margin-bottom:2rem;'>Advanced Intrusion Detection System</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sign In</div>', unsafe_allow_html=True)
        username = st.text_input("Username", key="login_user", placeholder="Enter username")
        password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
        if st.button("Sign In", use_container_width=True):
            users = load_users()
            ph = hashlib.sha256(password.encode()).hexdigest()
            if username in users and users[username] == ph:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Access granted.")
                time.sleep(0.8)
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.markdown('</div>', unsafe_allow_html=True)

def register_page():
    st.markdown("<h1>Create Account</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">New User Registration</div>', unsafe_allow_html=True)
        username = st.text_input("Username", key="reg_user")
        password = st.text_input("Password", type="password", key="reg_pass")
        confirm = st.text_input("Confirm Password", type="password", key="reg_conf")
        if st.button("Register", use_container_width=True):
            users = load_users()
            if username.strip() == "":
                st.error("Username cannot be empty.")
            elif username in users:
                st.error("Username already exists.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                users[username] = hashlib.sha256(password.encode()).hexdigest()
                save_users(users)
                st.success("Account created. Go to Sign In.")
        st.markdown('</div>', unsafe_allow_html=True)

def dashboard_page(models_loaded):
    st.markdown("<h1>Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#8a96a8; font-size:0.85rem; margin-bottom:1.5rem;'>Welcome, {st.session_state.username} &nbsp;|&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
    logs = load_predictions()
    total = len(logs)
    attacks = sum(1 for l in logs if l.get("cnn_result") != "Normal" or l.get("lstm_result") != "Normal")
    normal = total - attacks
    attack_rate = f"{(attacks / total * 100):.1f}%" if total > 0 else "0.0%"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-box"><span class="stat-number">{total}</span><span class="stat-label">Total Scanned</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box" style="border-left:4px solid #dc2626"><span class="stat-number" style="color:#dc2626">{attacks}</span><span class="stat-label">Attacks Detected</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box" style="border-left:4px solid #16a34a"><span class="stat-number" style="color:#16a34a">{normal}</span><span class="stat-label">Normal Traffic</span></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-box"><span class="stat-number">{attack_rate}</span><span class="stat-label">Attack Rate</span></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)
        if models_loaded:
            st.markdown("**CNN Model** — Loaded")
            st.markdown("**LSTM Model** — Loaded")
            st.markdown("**Scaler** — Loaded")
            st.markdown("**System** — Operational")
        else:
            st.error("Models failed to load. Check model files in app directory.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Attack Distribution</div>', unsafe_allow_html=True)
        if attacks > 0:
            attack_types = {}
            for l in logs:
                r = l.get("cnn_result", "Unknown")
                if r not in ("Normal", "Error", "Unknown"):
                    attack_types[r] = attack_types.get(r, 0) + 1
            if attack_types:
                dist_df = pd.DataFrame(list(attack_types.items()), columns=["Attack Type", "Count"])
                dist_df = dist_df.sort_values("Count", ascending=False)
                st.dataframe(dist_df, use_container_width=True, hide_index=True)
            else:
                st.info("No attack type data yet.")
        else:
            st.info("No attacks detected yet.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recent Activity</div>', unsafe_allow_html=True)
    if logs:
        recent = logs[-10:][::-1]
        rows = []
        for l in recent:
            is_attack = l.get("cnn_result") != "Normal" or l.get("lstm_result") != "Normal"
            rows.append({
                "Timestamp": l.get("timestamp", "-"),
                "Status": "ATTACK" if is_attack else "Normal",
                "CNN Result": l.get("cnn_result", "-"),
                "CNN Confidence": f"{l.get('cnn_confidence', 0):.2%}",
                "LSTM Result": l.get("lstm_result", "-"),
                "LSTM Confidence": f"{l.get('lstm_confidence', 0):.2%}",
                "Source": l.get("source", "manual")
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No predictions logged yet. Run a detection to see activity here.")
    st.markdown('</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Intrusion Detection", use_container_width=True):
            st.session_state.nav_page = "Intrusion Detection"
            st.rerun()
    with col2:
        if st.button("Project Information", use_container_width=True):
            st.session_state.nav_page = "Project Info"
            st.rerun()

def prediction_page(cnn_model, lstm_model, scaler, models_loaded, df):
    st.markdown("<h1>Intrusion Detection</h1>", unsafe_allow_html=True)
    if not models_loaded:
        st.error("Models not loaded. Place model files in the app directory.")
        return
    lime_training_data = get_lime_training_data(df)
    tab1, tab2 = st.tabs(["Manual Input", "Auto Detection"])

    with tab1:
        st.markdown('<div class="section-title">Quick Sample Load</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns([1, 5])
        with sc1:
            st.markdown('<div class="sample-btn-attack">', unsafe_allow_html=True)
            if st.button("Load Attack Sample", use_container_width=True):
                st.session_state.sample_loaded = "attack"
                st.session_state.sample_load_count += 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with sc2:
            if st.session_state.sample_loaded == "attack":
                st.info("Attack sample loaded. Click Analyse Traffic to predict.")

        st.markdown('<div class="section-title" style="margin-top:1rem;">Feature Input</div>', unsafe_allow_html=True)
        inputs = []
        if st.session_state.sample_loaded == "attack":
            sample_data = ATTACK_SAMPLE
        else:
            sample_data = None
        chunks = [feature_names[i:i+4] for i in range(0, len(feature_names), 4)]
        for chunk in chunks:
            cols = st.columns(len(chunk))
            for ci, feat in enumerate(chunk):
                with cols[ci]:
                    default_val = float(sample_data[feat]) if sample_data and feat in sample_data else 0.0
                    val = st.number_input(feat, value=default_val, key=f"f_{feat}_{st.session_state.sample_load_count}")
                    inputs.append(val)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analyse Traffic", use_container_width=True, key="manual_predict"):
            with st.spinner("Analysing..."):
                try:
                    arr = np.array(inputs, dtype=float).reshape(1, -1)
                    is_sample = st.session_state.sample_loaded == "attack"
                    result = make_prediction(arr, cnn_model, lstm_model, scaler, already_scaled=is_sample)
                    if not result["success"]:
                        st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
                    else:
                        save_prediction({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "cnn_result": result["cnn_result"],
                            "cnn_confidence": result["cnn_confidence"],
                            "lstm_result": result["lstm_result"],
                            "lstm_confidence": result["lstm_confidence"],
                            "source": "manual"
                        })
                        is_attack = result["cnn_result"] != "Normal" or result["lstm_result"] != "Normal"
                        if is_attack:
                            attack_type = result["cnn_result"] if result["cnn_result"] != "Normal" else result["lstm_result"]
                            st.markdown(f'<div class="attack-banner"><h3 style="color:#dc2626; margin:0;">Attack Detected — {attack_type}</h3><p style="color:#7f1d1d; margin:4px 0 0; font-size:0.88rem;">Immediate investigation recommended.</p></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="normal-banner"><h3 style="color:#16a34a; margin:0;">Traffic is Normal</h3><p style="color:#14532d; margin:4px 0 0; font-size:0.88rem;">No threats detected in this traffic pattern.</p></div>', unsafe_allow_html=True)
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("CNN Prediction", result["cnn_result"], f"{result['cnn_confidence']:.2%} confidence")
                        with m2:
                            st.metric("LSTM Prediction", result["lstm_result"], f"{result['lstm_confidence']:.2%} confidence")
                        st.markdown('<div class="section-title" style="margin-top:1.5rem;">LIME Explanation — CNN</div>', unsafe_allow_html=True)
                        with st.spinner("Generating explanation..."):
                            html, top_features, lime_ok = generate_lime(result["inputs_norm"], lime_training_data, cnn_model)
                        if lime_ok:
                            st.components.v1.html(html, height=800, scrolling=True)
                            st.markdown('<div class="section-title">Top Contributing Features</div>', unsafe_allow_html=True)
                            for feat, imp in top_features[:7]:
                                color = "#dc2626" if imp > 0 else "#16a34a"
                                direction = "increases" if imp > 0 else "decreases"
                                st.markdown(f'<p style="color:{color}; font-family:JetBrains Mono,monospace; font-size:0.82rem;">{direction} probability &nbsp;|&nbsp; <b>{feat}</b> &nbsp;|&nbsp; weight: {imp:.4f}</p>', unsafe_allow_html=True)
                        else:
                            st.warning("LIME explanation unavailable.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.code(traceback.format_exc())

    with tab2:
        st.markdown('<div class="section-title">Automatic Detection — Dataset Simulation</div>', unsafe_allow_html=True)
        if df is None:
            st.error(f"{CSV_FILE} not found. Place it in the app directory.")
            return
        st.markdown(f'<p style="color:#8a96a8; font-size:0.85rem; margin-bottom:1rem;">Dataset: {len(df)} rows &nbsp;|&nbsp; Current index: {st.session_state.current_row_index}</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            start_btn = st.button("Start", use_container_width=True, key="auto_start")
        with col2:
            stop_btn = st.button("Stop", use_container_width=True, key="auto_stop")
        with col3:
            reset_btn = st.button("Reset", use_container_width=True, key="auto_reset")
        with col4:
            delay = st.number_input("Delay (seconds)", min_value=1, max_value=10, value=2, key="auto_delay")
        if start_btn:
            st.session_state.is_running = True
        if stop_btn:
            st.session_state.is_running = False
        if reset_btn:
            st.session_state.current_row_index = 0
            st.session_state.is_running = False
            st.success("Reset to row 0.")
        result_placeholder = st.empty()
        data_placeholder = st.empty()
        lime_placeholder = st.empty()
        if st.session_state.is_running:
            if st.session_state.current_row_index >= len(df):
                st.session_state.current_row_index = 0
            row = df.iloc[st.session_state.current_row_index]
            feature_values = []
            for f in feature_names:
                if f in row:
                    try:
                        feature_values.append(float(row[f]))
                    except:
                        feature_values.append(0.0)
                else:
                    feature_values.append(0.0)
            arr = np.array(feature_values, dtype=float).reshape(1, -1)
            try:
                result = make_prediction(arr, cnn_model, lstm_model, scaler)
            except Exception as e:
                result = {"cnn_result": "Error", "cnn_confidence": 0.0, "lstm_result": "Error", "lstm_confidence": 0.0, "inputs_norm": arr, "success": False}
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if result.get("success", False):
                save_prediction({
                    "timestamp": timestamp,
                    "cnn_result": result["cnn_result"],
                    "cnn_confidence": result["cnn_confidence"],
                    "lstm_result": result["lstm_result"],
                    "lstm_confidence": result["lstm_confidence"],
                    "source": f"auto:row{st.session_state.current_row_index}"
                })
            is_attack = result["cnn_result"] != "Normal" or result["lstm_result"] != "Normal"
            with result_placeholder.container():
                if is_attack:
                    attack_type = result["cnn_result"] if result["cnn_result"] != "Normal" else result["lstm_result"]
                    st.markdown(f'<div class="attack-banner"><h3 style="color:#dc2626; margin:0;">Attack Detected — {attack_type}</h3><p style="color:#7f1d1d; margin:4px 0 0; font-size:0.85rem;">Row #{st.session_state.current_row_index} &nbsp;|&nbsp; {timestamp}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="normal-banner"><h3 style="color:#16a34a; margin:0;">Normal Traffic</h3><p style="color:#14532d; margin:4px 0 0; font-size:0.85rem;">Row #{st.session_state.current_row_index} &nbsp;|&nbsp; {timestamp}</p></div>', unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("CNN", result["cnn_result"], f"{result['cnn_confidence']:.2%}")
                with m2:
                    st.metric("LSTM", result["lstm_result"], f"{result['lstm_confidence']:.2%}")
            with data_placeholder.container():
                st.markdown('<div class="section-title">Row Data</div>', unsafe_allow_html=True)
                row_dict = {f: row[f] for f in feature_names if f in row}
                st.dataframe(pd.DataFrame([row_dict]), use_container_width=True)
            with lime_placeholder.container():
                st.markdown('<div class="section-title">LIME Explanation</div>', unsafe_allow_html=True)
                if result.get("success", False):
                    html, top_features, lime_ok = generate_lime(result["inputs_norm"], lime_training_data, cnn_model)
                    if lime_ok:
                        st.components.v1.html(html, height=700, scrolling=True)
                        for feat, imp in top_features[:5]:
                            color = "#dc2626" if imp > 0 else "#16a34a"
                            direction = "increases" if imp > 0 else "decreases"
                            st.markdown(f'<p style="color:{color}; font-family:JetBrains Mono,monospace; font-size:0.82rem;">{direction} &nbsp;|&nbsp; <b>{feat}</b>: {imp:.4f}</p>', unsafe_allow_html=True)
            st.session_state.current_row_index += 1
            time.sleep(delay)
            st.rerun()

def project_info_page():
    st.markdown("<h1>Project Information</h1>", unsafe_allow_html=True)
    sections = [
        ("Project Overview", "This system implements an Intrusion Detection System (IDS) using deep learning. It handles class imbalance in network traffic data using Generative Adversarial Networks (GANs) to synthesise minority attack class samples, then uses CNN and LSTM models for real-time detection."),
        ("Model Architecture", "**CNN Model** — Processes normalised input features for spatial feature extraction and classification.\n\n**LSTM Model** — Captures sequential patterns in network traffic by processing a 10-timestep window of normalised features.\n\n**Input** — 43 features from the UNSW-NB15 dataset, normalised via a pre-fitted MinMaxScaler.\n\n**Output** — One of 10 classes: Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms."),
        ("GAN-Based Data Augmentation", "The original UNSW-NB15 dataset is heavily imbalanced — the Normal class had ~37,000 samples while attack classes had far fewer. A separate GAN was trained per minority attack class. Each GAN generated synthetic samples to bring every class up to 37,000 samples, creating a fully balanced training dataset."),
        ("Explainability — LIME", "LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions.\n\n**Red bars** — features that increase the probability of the predicted class.\n\n**Blue/green bars** — features that decrease the probability.\n\nBar length represents feature importance magnitude. LIME uses the actual data distribution from balanced_sampled.csv for reliable explanations."),
        ("Detection Classes", "Normal — Benign traffic\n\nGeneric — Generic attack\n\nExploits — Exploitation attack\n\nFuzzers — Fuzzing attack\n\nDoS — Denial of Service\n\nReconnaissance — Scanning / probing\n\nAnalysis — Packet analysis attack\n\nBackdoor — Backdoor attack\n\nShellcode — Shellcode injection\n\nWorms — Network worms"),
    ]
    for title, content in sections:
        st.markdown(f'<div class="card"><div class="section-title">{title.upper()}</div>', unsafe_allow_html=True)
        st.markdown(content)
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    cnn_model, lstm_model, scaler, models_loaded = load_all_models()
    df = load_csv()
    if not st.session_state.authenticated:
        page = st.sidebar.radio("Menu", ["Sign In", "Register"], label_visibility="collapsed")
        if page == "Sign In":
            sign_in_page()
        else:
            register_page()
        return
    pages = ["Dashboard", "Intrusion Detection", "Project Info"]
    if st.session_state.nav_page in pages:
        default_idx = pages.index(st.session_state.nav_page)
    else:
        default_idx = 0
    page = st.sidebar.radio("Navigation", pages, index=default_idx, label_visibility="collapsed")
    st.session_state.nav_page = page
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<p style='font-size:0.8rem; color:#8a9ab8;'>Logged in as: <b style='color:#ffffff'>{st.session_state.username}</b></p>", unsafe_allow_html=True)
    if st.sidebar.button("Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.is_running = False
        st.rerun()
    if page == "Dashboard":
        dashboard_page(models_loaded)
    elif page == "Intrusion Detection":
        prediction_page(cnn_model, lstm_model, scaler, models_loaded, df)
    elif page == "Project Info":
        project_info_page()

if __name__ == "__main__":
    main()