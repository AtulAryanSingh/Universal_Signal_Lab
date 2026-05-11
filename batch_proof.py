import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.io import wavfile
import os
import glob
import random

st.set_page_config(page_title="Spectral Sieve: Frequency Domain", layout="wide")
st.title("🎛️ Universal Signal Laboratory: Spectral Sieve")
st.markdown("Upgrading from Time-Domain MSE to Frequency-Domain FFT Bandpass isolating.")

# --- THE TUNED PARAMETERS ---
lookback = 400
p1 = 5.0
p2 = 13.0
weight = 0.15
omega = 0.8
sample_rate = 4000 # Hz for Pascal dataset

st.sidebar.header("Global AI Architecture")
train_size = st.sidebar.slider("Healthy Calibration Patients", min_value=10, max_value=80, value=30, step=5)
z_score = st.sidebar.slider("Spectral Sensitivity (Z-Score)", min_value=1.0, max_value=15.0, value=4.0, step=0.5)

st.sidebar.header("Fourier Bandpass Filter")
st.sidebar.markdown("Targeting the high-frequency friction of a torn valve.")
low_cut_hz = st.sidebar.slider("Low Cutoff (Hz) - Ignore Bumps", 0, 1000, 150, 10)
high_cut_hz = st.sidebar.slider("High Cutoff (Hz) - Ignore Static", 100, 2000, 600, 10)

dataset_path = st.sidebar.text_input("Enter the path to the 'set_b' folder:", value="./set_b")

def get_sieve_activation(x):
    return tf.sin(p1 * omega * x) + weight * tf.sin(p2 * omega * x)

def build_sieve():
    m_res = models.Sequential([
        layers.InputLayer(shape=(lookback, 1)), layers.Flatten(),
        layers.Dense(32, activation=get_sieve_activation), layers.Dense(lookback)
    ])
    m_res.compile(optimizer='adam', loss='mse')
    return m_res

def process_wav(file_path):
    _, data = wavfile.read(file_path)
    if len(data.shape) > 1: data = data[:, 0]
    max_frames = 6000 
    if len(data) > max_frames: data = data[:max_frames]
    return data / np.max(np.abs(data))

# --- SPECTRAL ERROR FUNCTION ---
def calculate_spectral_error(X_true, X_pred):
    """ Converts time-domain error into frequency-domain bandpass energy """
    residuals = X_true.reshape(-1, lookback) - X_pred
    
    # Fast Fourier Transform on the residuals
    fft_res = np.fft.rfft(residuals, axis=1)
    frequencies = np.fft.rfftfreq(lookback, d=1/sample_rate)
    
    # Calculate Power Spectral Density (PSD)
    psd = np.abs(fft_res)**2
    
    # Create the Bandpass Mask
    band_mask = (frequencies >= low_cut_hz) & (frequencies <= high_cut_hz)
    
    # Sum the energy ONLY within the Murmur frequency band
    spectral_energy = np.sum(psd[:, band_mask], axis=1)
    return spectral_energy


if st.button("Initialize Spectral Clinical Trial"):
    if not os.path.exists(dataset_path):
        st.error("Cannot find dataset folder.")
    else:
        all_files = glob.glob(os.path.join(dataset_path, "*.wav"))
        pure_normals = [f for f in all_files if os.path.basename(f).startswith("normal__")]
        
        if len(pure_normals) < train_size:
            st.error("Not enough pure normal files.")
            st.stop()
            
        random.shuffle(pure_normals)
        calibration_files = pure_normals[:train_size]
        test_files = [f for f in all_files if f not in calibration_files and not os.path.basename(f).lower().startswith("b")]
        
        # --- STAGE 1: SPECTRAL CALIBRATION ---
        st.header("Stage 1: Spectral Calibration")
        calib_bar = st.progress(0)
        master_X_train = []
        
        for i, f in enumerate(calibration_files):
            raw_data = process_wav(f)
            X = np.array([raw_data[j:j+lookback] for j in range(len(raw_data)-lookback)])[..., np.newaxis]
            master_X_train.append(X)
            calib_bar.progress((i+1)/train_size)
            
        X_train_tensor = np.concatenate(master_X_train, axis=0)
        
        m_res = build_sieve()
        m_res.fit(X_train_tensor, X_train_tensor.reshape(-1, lookback), epochs=5, batch_size=256, verbose=0)
        
        # Calculate GLOBAL SPECTRAL THRESHOLD
        baseline_pred = m_res.predict(X_train_tensor, batch_size=512, verbose=0)
        spectral_baseline_err = calculate_spectral_error(X_train_tensor, baseline_pred)
        
        global_mean = np.mean(spectral_baseline_err)
        global_std = np.std(spectral_baseline_err)
        spectral_threshold = global_mean + (z_score * global_std)
        
        st.success(f"Spectral Baseline Locked. Targeting {low_cut_hz}Hz - {high_cut_hz}Hz. Threshold: {spectral_threshold:.4f}")
        
        # --- STAGE 2: SPECTRAL INFERENCE ---
        st.header("Stage 2: Frequency-Blind Inference")
        trial_bar = st.progress(0)
        
        TP = 0; TN = 0; FP = 0; FN = 0
        results = {"Normal (Clean)": {"Passed": 0, "Failed": 0}, 
                   "Noisy Normal": {"Passed": 0, "Failed": 0},
                   "Murmur": {"Caught": 0, "Missed": 0}}
                   
        st.warning("Note: Extrastoles (skipped beats) are removed from the matrix as they are time-domain arrhythmias, not spectral friction anomalies.")
        
        for i, f in enumerate(test_files):
            filename = os.path.basename(f).lower()
            if filename.startswith("extrastole"): 
                trial_bar.progress((i+1)/len(test_files))
                continue
                
            is_murmur = filename.startswith("murmur")
            
            try:
                raw_data = process_wav(f)
                X_test = np.array([raw_data[j:j+lookback] for j in range(len(raw_data)-lookback)])[..., np.newaxis]
                
                test_pred = m_res.predict(X_test, verbose=0)
                
                # USE FFT INSTEAD OF MSE
                spectral_test_err = calculate_spectral_error(X_test, test_pred)
                
                peak_intensity = np.max(spectral_test_err)
                sieve_triggered = peak_intensity > spectral_threshold
                
                if is_murmur:
                    if sieve_triggered:
                        TP += 1
                        results["Murmur"]["Caught"] += 1
                    else:
                        FN += 1
                        results["Murmur"]["Missed"] += 1
                else:
                    if not sieve_triggered:
                        TN += 1
                        if filename.startswith("normal_noisy"): results["Noisy Normal"]["Passed"] += 1
                        elif filename.startswith("normal"): results["Normal (Clean)"]["Passed"] += 1
                    else:
                        FP += 1
                        if filename.startswith("normal_noisy"): results["Noisy Normal"]["Failed"] += 1
                        elif filename.startswith("normal"): results["Normal (Clean)"]["Failed"] += 1
            except:
                pass
                
            trial_bar.progress((i+1)/len(test_files))
            
        # --- REVEAL ---
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**True Positives (Murmurs Caught):** {TP}")
            st.success(f"**True Negatives (Healthy Verified):** {TN}")
        with col2:
            st.error(f"**False Positives (False Alarms):** {FP}")
            st.error(f"**False Negatives (Missed Murmurs):** {FN}")
            
        st.write(results)
        
        total = TP + TN + FP + FN
        accuracy = ((TP + TN) / total) * 100 if total > 0 else 0
        st.markdown(f"## **Total Spectral Accuracy:** {accuracy:.2f}%")