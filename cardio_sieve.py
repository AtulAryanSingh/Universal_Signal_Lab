import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from scipy.io import wavfile

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cardio Sieve: Biotelemetry", layout="wide")
st.title("🫀 Resonant Sieve: Acoustic Cardiology Lab")
st.markdown("**Target:** Detecting high-frequency micro-fibrillations (Murmurs) bypassing low-frequency heartbeats.")

# --- SIDEBAR: THE BIOLOGICAL TUNER ---
st.sidebar.header("Biological Sieve Tuning")
lookback = st.sidebar.slider("Audio Window (Tensor Size)", min_value=20, max_value=500, value=80, step=10)

st.sidebar.markdown("### Prime Resonance Nodes")
prime_1 = st.sidebar.number_input("Primary Prime (p1)", value=3.0, step=1.0)
prime_2 = st.sidebar.number_input("Dissonance Prime (p2)", value=17.0, step=1.0)
prime_2_weight = st.sidebar.slider("Harmonic Weight", min_value=0.05, max_value=0.5, value=0.15, step=0.01)

# Dynamic Phase scaling for biology
omega = st.sidebar.slider("Biological Base Scale (ω)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
threshold = st.sidebar.slider("Anomaly Threshold (Absolute MSE)", min_value=0.02, max_value=1.0, value=0.25, step=0.05)

# --- THE ARCHITECTURE ---
def get_sieve_activation(p1, p2, weight, w):
    # The mathematical haloscope, upgraded with dynamic biological scaling (w)
    def universal_sieve_act(x):
        return tf.sin(p1 * w * x) + weight * tf.sin(p2 * w * x)
    return universal_sieve_act

def build_models(activation_fn):
    # Standard AI (The spatial baseline)
    m_relu = models.Sequential([
        layers.InputLayer(shape=(lookback, 1)), layers.Flatten(),
        layers.Dense(32, activation='relu'), layers.Dense(lookback)
    ])
    # The Resonant Sieve
    m_res = models.Sequential([
        layers.InputLayer(shape=(lookback, 1)), layers.Flatten(),
        layers.Dense(32, activation=activation_fn), layers.Dense(lookback)
    ])
    m_relu.compile(optimizer='adam', loss='mse')
    m_res.compile(optimizer='adam', loss='mse')
    return m_relu, m_res

# --- DATA INGESTION ---
st.header("1. Ingest PCG (Heart Audio)")
uploaded_file = st.file_uploader("Upload .wav file from Pascal 'set_b' folder", type=['wav'])

if uploaded_file is not None:
    with st.spinner("Processing biological acoustics..."):
        # Read the audio file
        samplerate, data = wavfile.read(uploaded_file)
        
        # If stereo, take just one channel
        if len(data.shape) > 1:
            data = data[:, 0]
            
        # Limit frames so the UI doesn't freeze on massive audio files
        max_frames = 8000
        if len(data) > max_frames:
            data = data[:max_frames]
            
        # Normalize the audio between -1 and 1
        raw_data = data / np.max(np.abs(data))
        
        st.success(f"Audio Loaded: {len(raw_data)} frames at {samplerate} Hz.")
        
        # Plot the raw audio so you can see the "Thump-Thump"
        st.subheader("Raw Acoustic Waveform (The Baseline)")
        st.line_chart(raw_data)

    # --- PROCESSING ---
    st.header("2. Run Biological Diagnostic")
    if st.button("Initialize Resonant Sieve"):
        X_blind = np.array([raw_data[i:i+lookback] for i in range(len(raw_data)-lookback)])[..., np.newaxis]
        
        with st.spinner("Training on initial rhythm to establish biological baseline..."):
            sieve_act = get_sieve_activation(prime_1, prime_2, prime_2_weight, omega)
            m_relu, m_res = build_models(sieve_act)
            
            # Train on the first 15% of the heartbeat
            baseline_idx = int(len(X_blind) * 0.15)
            X_train = X_blind[:baseline_idx]
            
            m_relu.fit(X_train, X_train.reshape(-1, lookback), epochs=8, verbose=0)
            m_res.fit(X_train, X_train.reshape(-1, lookback), epochs=8, verbose=0)
            
        with st.spinner("Scanning for high-frequency structural dissonance..."):
            err_relu = np.mean(np.square(X_blind.reshape(-1, lookback) - m_relu.predict(X_blind, verbose=0)), axis=1)
            err_res = np.mean(np.square(X_blind.reshape(-1, lookback) - m_res.predict(X_blind, verbose=0)), axis=1)
        
        # --- VISUALIZATION ---
        st.header("3. The Diagnostic Reveal")
        fig, ax = plt.subplots(figsize=(15, 6))
        
        t_axis = np.linspace(0, len(err_relu), len(err_relu))
        ax.plot(t_axis, err_relu, color='blue', alpha=0.4, label='Standard AI (ReLU) - Blind to Murmur')
        ax.plot(t_axis, err_res, color='red', linewidth=1.5, label='Resonant Sieve - Acoustic Detection')
        
        ax.set_title("Biotelemetry Analysis: Heart Valve Integrity")
        ax.set_ylabel("Detection Intensity (MSE)")
        ax.set_xlabel("Time (Frames)")
        ax.legend()
        
        st.pyplot(fig)
        
        # --- DYNAMIC INTELLIGENCE LOGIC ---
        peak_intensity = np.max(err_res)
        
        # We now use the absolute threshold set by the Architect in the sidebar
        if peak_intensity > threshold:
            st.error(f"🚨 CARDIAC ANOMALY DETECTED: High-frequency dissonance breached safety limits. High probability of valve murmur. (Peak Intensity: {peak_intensity:.4f} | Threshold: {threshold})")
        else:
            st.success(f"✅ Biological integrity verified. Normal periodic rhythm confirmed. (Peak Intensity: {peak_intensity:.4f} | Threshold: {threshold})")
        
        
        