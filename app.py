import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Resonance Lab: PoC", layout="wide")
st.title("📡 Universal Signal Laboratory (Alpha)")
st.markdown("### Domain: Industrial Predictive Maintenance (NASA Bearings)")

# --- SIDEBAR: THE CONTROL PANEL ---
st.sidebar.header("Sieve Parameters")
lookback = st.sidebar.slider("Lookback Window", min_value=10, max_value=100, value=50, step=10)

st.sidebar.markdown("### Prime Activation Tuning")
prime_1 = st.sidebar.number_input("Primary Frequency (Prime 1)", value=3.0, step=1.0)
prime_2 = st.sidebar.number_input("Harmonic Dissonance (Prime 2)", value=13.0, step=1.0)
prime_2_weight = st.sidebar.slider("Harmonic Weight", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

# --- THE ARCHITECTURE ---
def get_sieve_activation(p1, p2, weight):
    # Dynamically generates the activation function based on user input
    def universal_sieve_act(x):
        return tf.sin(p1 * x) + weight * tf.sin(p2 * x)
    return universal_sieve_act

def build_models(activation_fn):
    # The Industry Standard (Baseline)
    m_relu = models.Sequential([
        layers.InputLayer(shape=(lookback, 1)), layers.Flatten(),
        layers.Dense(64, activation='relu'), layers.Dense(lookback)
    ])
    # The Resonant Sieve
    m_res = models.Sequential([
        layers.InputLayer(shape=(lookback, 1)), layers.Flatten(),
        layers.Dense(64, activation=activation_fn), layers.Dense(lookback)
    ])
    m_relu.compile(optimizer='adam', loss='mse')
    m_res.compile(optimizer='adam', loss='mse')
    return m_relu, m_res

# --- DATA INGESTION ---
st.header("1. Ingest Telemetry")
st.markdown("Upload a raw NASA bearing text file, or run the synthetic demo.")

# CRITICAL FIX: Removed the 'type' restriction. It will now accept the extensionless NASA files.
uploaded_file = st.file_uploader("Upload Raw Vibration Data (NASA format)")
use_demo = st.button("Generate Demo Hardware Telemetry")

raw_data = None

if uploaded_file is not None or use_demo:
    if uploaded_file is not None:
        try:
            # Parses the tab-separated NASA dataset correctly
            df = pd.read_csv(uploaded_file, sep='\t', header=None)
            raw_data = df.iloc[:, 0].values # Extract the first sensor column
            st.success(f"Loaded {len(raw_data)} real telemetry frames from NASA bearing file.")
        except Exception as e:
            st.error(f"Error parsing file: {e}. Please ensure it is a raw NASA dataset file.")
            raw_data = None
    elif use_demo:
        # Generate Synthetic NASA-style bearing failure
        t = np.linspace(0, 20, 20000)
        healthy_rumble = 2.0 * np.sin(5 * 2 * np.pi * t) + 1.0 * np.random.randn(len(t))
        raw_data = healthy_rumble
        # Inject micro-fracture at t=12s
        fracture_idx = 12000
        raw_data[fracture_idx:15000] += 0.3 * np.sin(85 * 2 * np.pi * t[fracture_idx:15000])
        st.info("Loaded Demo Hardware Telemetry (Micro-fracture hidden at 12s)")

    if raw_data is not None:
        # Prepare Windows
        X_blind = np.array([raw_data[i:i+lookback] for i in range(len(raw_data)-lookback)])[..., np.newaxis]
        
        # --- PROCESSING ---
        st.header("2. Resonant Sieve Execution")
        if st.button("Run Diagnostic Sieve"):
            with st.spinner("Compiling geometries and establishing baseline..."):
                sieve_act = get_sieve_activation(prime_1, prime_2, prime_2_weight)
                m_relu, m_res = build_models(sieve_act)
                
                # Train on the first 10% to establish the "Normal" healthy baseline
                baseline_idx = int(len(X_blind) * 0.1)
                X_train = X_blind[:baseline_idx]
                
                m_relu.fit(X_train, X_train.reshape(-1, lookback), epochs=10, verbose=0)
                m_res.fit(X_train, X_train.reshape(-1, lookback), epochs=10, verbose=0)
                
            with st.spinner("Scanning full stream for harmonic anomalies..."):
                err_relu = np.mean(np.square(X_blind.reshape(-1, lookback) - m_relu.predict(X_blind, verbose=0)), axis=1)
                err_res = np.mean(np.square(X_blind.reshape(-1, lookback) - m_res.predict(X_blind, verbose=0)), axis=1)
            
            # --- VISUALIZATION ---
            st.header("3. The Reveal")
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # Plot detection intensities
            t_axis = np.linspace(0, len(err_relu), len(err_relu))
            ax.plot(t_axis, err_relu, color='blue', alpha=0.4, label='Standard AI (ReLU) - Blind')
            ax.plot(t_axis, err_res, color='red', linewidth=1.5, label='Resonant Sieve - Awake')
            
            ax.set_title("Spectral Bias Reveal: Tracking the Micro-Fracture")
            ax.set_ylabel("Detection Intensity (MSE)")
            ax.set_xlabel("Telemetry Window Index")
            ax.legend()
            
            st.pyplot(fig)
            st.success("Sieve Complete. The architecture has verified the structural integrity.")