"""
LOFAR Spectrogram Playground - Streamlit Web Application

This application provides an interactive interface for analyzing audio signals
using LOFAR (Low Frequency Analysis and Recording) spectrograms. Users can
upload audio files, adjust various signal processing parameters, and visualize
the results in real-time.

LOFAR is commonly used in underwater acoustics and sonar applications to
detect and analyze low-frequency signals with high sensitivity.
"""

import io
import json
from datetime import datetime
from math import floor
from pathlib import Path
from typing import Dict, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from matplotlib.colors import ListedColormap

from lofar import lofar
from demon import demongram

# ============================================================================
# Configuration Constants
# ============================================================================

# PROJECT_ROOT is the parent directory (project root) to access data/ and assets/
PROJECT_ROOT = Path(__file__).parent.parent
# Default directory containing sample audio files
DEFAULT_AUDIO_ROOT = PROJECT_ROOT / "data"
# LOGO_PATH is in the assets directory (parent of src)
LOGO_PATH = PROJECT_ROOT / "assets" / "Logo.png"

# Available window functions for STFT (Short-Time Fourier Transform)
# Each window has different spectral leakage characteristics:
# - hann: Good frequency resolution, moderate sidelobe suppression
# - hamming: Similar to hann but slightly better sidelobe suppression
# - blackman: Better sidelobe suppression, wider main lobe
# - bartlett: Triangular window, simple but less optimal
WINDOW_CHOICES = ["hann", "hamming", "blackman", "bartlett"]

# FFT window sizes (nperseg parameter)
# Larger values = better frequency resolution but worse time resolution
# Smaller values = better time resolution but worse frequency resolution
FFT_CHOICES = [256, 512, 1024, 2048, 4096]

# Standard matplotlib colormaps for visualization
BASE_CMAP_CHOICES = ["viridis", "magma", "plasma", "inferno", "cividis", "turbo", "gray", "jet"]

# Custom colormap designed to mimic traditional sonar displays (green on black)
CUSTOM_CMAPS = {
    "sonar_green": ListedColormap(
        [
            (0.0, 0.0, 0.0),      # Black (lowest values)
            (0.0, 0.03, 0.005),   # Very dark green
            (0.0, 0.08, 0.015),   # Dark green
            (0.0, 0.15, 0.03),    # Medium-dark green
            (0.0, 0.25, 0.05),    # Medium green
            (0.0, 0.4, 0.08),     # Medium-bright green
            (0.0, 0.55, 0.12),    # Bright green (highest values)
        ],
        name="sonar_green",
    )
}
CMAP_CHOICES = BASE_CMAP_CHOICES + list(CUSTOM_CMAPS.keys())

# Default parameter preset for LOFAR analysis
# These values provide a good starting point for most audio analysis tasks
DEFAULT_PRESET = {
    "window": WINDOW_CHOICES[0],      # Hann window (good balance)
    "nperseg": 2048,                  # FFT window size (1024 samples)
    "noverlap": 1905,                  # 50% overlap between windows
    "decimation_factor": 10,           # Downsample by factor of 3
    "floor_threshold": -0.8,          # Clamp values below -0.2 log10 power
    "floor_value": -0.7,               # Replace clamped values with 0.0
    "eps": 1e-12,                     # Small epsilon to prevent log(0)
    "tpsw_n": None,                   # TPSW half-window size (None = adaptive default)
    "tpsw_p": None,                   # TPSW guard band size (None = adaptive default)
    "tpsw_a": None,                   # TPSW threshold multiplier (None = default 2.0)
    "hough_enhance": False,           # Enable Hough transform line enhancement
    "hough_threshold_percentile": 75, # Percentile threshold for line detection
    "hough_min_line_length": None,    # Minimum line length (None = adaptive)
    "hough_max_line_gap": 5,          # Maximum gap between line segments
    "hough_theta_range": (85.0, 95.0), # Angle range in degrees for vertical lines
    "hough_enhancement_factor": 1.5,  # Enhancement factor for detected lines
    "hough_use_morphology": False,    # Enable morphological operations
    "hough_morph_kernel_size": 3,     # Morphological kernel size (must be odd)
    "hough_morph_iterations": 1,      # Number of morphological iterations
    "hough_morph_operation": "closing", # Morphological operation type
    "rotate_90": True,               # Standard orientation (time on X-axis)
    "cmap": CMAP_CHOICES[8],          # Default colormap (viridis)
    "gamma": 1.0,                     # Gamma correction factor (1.0 = no correction)
}


@st.cache_data(show_spinner=False)
def list_sample_files(root: Path) -> Tuple[str, ...]:
    """
    Discover and return all WAV files in the data directory.
    
    Uses Streamlit's caching to avoid re-scanning the directory on every
    page refresh. Returns relative paths from PROJECT_ROOT for portability.
    
    Args:
        root: Directory path to search for WAV files
        
    Returns:
        Tuple of relative file paths (empty if directory doesn't exist)
    """
    if not root.exists():
        return tuple()
    # Recursively find all .wav files and sort them
    wavs = sorted(p for p in root.rglob("*.wav"))
    # Return paths relative to project root for display and loading
    return tuple(str(p.relative_to(PROJECT_ROOT)) for p in wavs)


@st.cache_data(show_spinner=False)
def load_audio_from_path(path_str: str) -> Tuple[np.ndarray, int]:
    """
    Load an audio file from disk and convert to mono waveform.
    
    Uses librosa for robust audio loading (handles various formats).
    Cached by Streamlit to avoid reloading the same file repeatedly.
    
    Args:
        path_str: Relative path to audio file from PROJECT_ROOT
        
    Returns:
        Tuple of (waveform array, sample_rate)
    """
    full_path = PROJECT_ROOT / path_str
    # sr=None preserves original sample rate, mono=True converts to single channel
    waveform, sr = librosa.load(full_path, sr=None, mono=True)
    return waveform, sr


def load_audio_from_upload(upload) -> Tuple[np.ndarray, int]:
    """
    Load audio from a user-uploaded file through Streamlit's file uploader.
    
    Converts uploaded file to bytes, then reads with soundfile.
    If stereo/multi-channel, extracts only the first channel.
    
    Args:
        upload: Streamlit UploadedFile object
        
    Returns:
        Tuple of (mono waveform array, sample_rate)
    """
    # Read from uploaded file bytes
    data, sr = sf.read(io.BytesIO(upload.getvalue()))
    # If multi-channel, take only the first channel (left channel)
    if data.ndim > 1:
        data = data[:, 0]
    return data.astype(np.float32), sr


def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert a numpy waveform array to WAV file bytes for audio playback.
    
    This allows the Streamlit audio widget to play the loaded/processed audio.
    The audio is written to an in-memory buffer and returned as bytes.
    
    Args:
        waveform: Audio signal as numpy array
        sample_rate: Sample rate in Hz
        
    Returns:
        WAV file as bytes
    """
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV")
    buf.seek(0)  # Reset buffer position to beginning
    return buf.read()


def render_lofargram(waveform: np.ndarray, sample_rate: int, *, decimation_factor: int,
                     window: str, nperseg: int, noverlap: int,
                     floor_threshold: float, floor_value: float, eps: float,
                     tpsw_n=None, tpsw_p=None, tpsw_a=None,
                     hough_enhance=False, hough_threshold_percentile=75,
                     hough_min_line_length=None, hough_max_line_gap=5,
                     hough_theta_range=None, hough_enhancement_factor=1.5,
                     hough_use_morphology=False, hough_morph_kernel_size=3,
                     hough_morph_iterations=1, hough_morph_operation='closing'):
    """
    Wrapper function to compute LOFAR spectrogram with user-selected parameters.
    
    This function acts as a bridge between the Streamlit UI and the core
    LOFAR processing function in lofar.py. All parameters are passed
    through to the lofar() function.
    
    Args:
        waveform: Input audio signal
        sample_rate: Original sample rate of the audio
        decimation_factor: Downsampling factor (reduces computational load)
        window: Window function name for STFT
        nperseg: FFT window size in samples
        noverlap: Number of overlapping samples between windows
        floor_threshold: Minimum log10 power value before clamping
        floor_value: Value to replace clamped samples with
        eps: Small epsilon to prevent log(0) errors
        tpsw_n: Half-window size for TPSW normalization (None for adaptive default)
        tpsw_p: Guard band size for TPSW normalization (None for adaptive default)
        tpsw_a: Threshold multiplier for TPSW normalization (None for default 2.0)
        hough_enhance: Enable Hough transform line enhancement
        hough_threshold_percentile: Percentile threshold for line detection (0-100)
        hough_min_line_length: Minimum line length in pixels (None for adaptive)
        hough_max_line_gap: Maximum gap between line segments
        hough_theta_range: Angle range tuple in degrees (None for all angles)
        hough_enhancement_factor: Enhancement factor for detected lines
        hough_use_morphology: Enable morphological operations to filter and merge lines
        hough_morph_kernel_size: Size of morphological kernel (must be odd)
        hough_morph_iterations: Number of iterations for morphological operations
        hough_morph_operation: Morphological operation type ('opening', 'closing', 'both', 'erosion', 'dilation')
        
    Returns:
        Tuple of (time_axis, frequency_axis, spectrogram_matrix, hough_lines)
        where hough_lines is None if hough_enhance is False, otherwise a list of line coordinates
    """
    return lofar(
        waveform,
        sample_rate,
        decimation_factor=decimation_factor,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        floor_threshold=floor_threshold,
        floor_value=floor_value,
        eps=eps,
        tpsw_n=tpsw_n,
        tpsw_p=tpsw_p,
        tpsw_a=tpsw_a,
        hough_enhance=hough_enhance,
        hough_threshold_percentile=hough_threshold_percentile,
        hough_min_line_length=hough_min_line_length,
        hough_max_line_gap=hough_max_line_gap,
                    hough_theta_range=hough_theta_range,
                    hough_enhancement_factor=hough_enhancement_factor,
                    hough_use_morphology=hough_use_morphology,
                    hough_morph_kernel_size=hough_morph_kernel_size,
                    hough_morph_iterations=hough_morph_iterations,
                    hough_morph_operation=hough_morph_operation,
                )


def render_demongram(waveform: np.ndarray, sample_rate: int, *, method: str,
                     cutoff: float, high: float, low: float,
                     window_size: int, overlap: float):
    """
    Wrapper function to compute DEMONgram with user-selected parameters.
    
    This function acts as a bridge between the Streamlit UI and the core
    DEMON processing function in demon.py. All parameters are passed
    through to the demongram() function.
    
    Args:
        waveform: Input audio signal
        sample_rate: Original sample rate of the audio
        method: DEMON method ('square_law' or 'hilbert_detector')
        cutoff: DEMON cutoff frequency
        high: DEMON high frequency
        low: DEMON low frequency
        window_size: Window size in samples
        overlap: Overlap fraction (0-1)
        
    Returns:
        Tuple of (time_axis, frequency_axis, demongram_matrix)
    """
    demongram_output = demongram(
        waveform,
        window_size=window_size,
        overlap=overlap,
        method=method,
        cutoff=cutoff,
        high=high,
        low=low,
        fs=sample_rate,
    )
    
    # Calculate decimation rate to determine output sample rate
    n = int(floor(sample_rate / (cutoff * 2)))
    output_fs = sample_rate / n
    
    # demongram_output shape is (num_freq_bins, num_windows) - freq bins as rows, time windows as columns
    num_freq_bins, num_windows = demongram_output.shape
    
    # Create time axis (center of each window)
    actual_window_size = window_size if window_size is not None else sample_rate
    hop_size = int(actual_window_size * (1 - overlap))
    time_axis = np.arange(num_windows) * (hop_size / sample_rate) + (actual_window_size / sample_rate / 2)
    
    # Create frequency axis for FFT output (one-sided spectrum)
    # Frequency bins go from 0 to Nyquist (cutoff) of decimated signal
    freq_axis = np.linspace(0, cutoff, num_freq_bins)
    
    # Transpose to match LOFAR format (freq x time) - (num_freq_bins, num_windows) -> (num_freq_bins, num_windows)
    # Actually, demongram already has (freq, time) format, so no transpose needed
    return time_axis, freq_axis, demongram_output  # Already in (freq x time) format


def init_session_state():
    """
    Initialize Streamlit session state with default values.
    
    Session state persists across reruns, allowing the UI to remember
    user selections. This function sets default values only if they
    don't already exist (using setdefault to avoid overwriting).
    """
    defaults = {
        # Analysis method - can be "LOFAR", "DEMON", or ["LOFAR", "DEMON"]
        "analysis_method": "LOFAR",
        "show_lofar": True,
        "show_demon": False,
        # LOFAR processing parameters
        "window_select": DEFAULT_PRESET["window"],
        "nperseg_select": DEFAULT_PRESET["nperseg"],
        "noverlap_value": DEFAULT_PRESET["noverlap"],
        "decimation_factor": DEFAULT_PRESET["decimation_factor"],
        "floor_threshold": DEFAULT_PRESET["floor_threshold"],
        "floor_value": DEFAULT_PRESET["floor_value"],
        "eps_value": DEFAULT_PRESET["eps"],
        # TPSW normalization parameters
        "tpsw_n": DEFAULT_PRESET["tpsw_n"],
        "tpsw_p": DEFAULT_PRESET["tpsw_p"],
        "tpsw_a": DEFAULT_PRESET["tpsw_a"],
        # Hough transform line enhancement parameters
        "hough_enhance": DEFAULT_PRESET["hough_enhance"],
        "hough_threshold_percentile": DEFAULT_PRESET["hough_threshold_percentile"],
        "hough_min_line_length": DEFAULT_PRESET["hough_min_line_length"],
        "hough_max_line_gap": DEFAULT_PRESET["hough_max_line_gap"],
        "hough_theta_range": DEFAULT_PRESET["hough_theta_range"],
        "hough_enhancement_factor": DEFAULT_PRESET["hough_enhancement_factor"],
        "hough_use_morphology": DEFAULT_PRESET["hough_use_morphology"],
        "hough_morph_kernel_size": DEFAULT_PRESET["hough_morph_kernel_size"],
        "hough_morph_iterations": DEFAULT_PRESET["hough_morph_iterations"],
        "hough_morph_operation": DEFAULT_PRESET["hough_morph_operation"],
        # DEMON processing parameters
        "demon_method": "square_law",
        "demon_cutoff": 1000.0,
        "demon_high": 10000.0,  # More reasonable default for typical audio
        "demon_low": 5000.0,    # More reasonable default for typical audio
        "demon_window_size": None,  # Will default to fs (1 second)
        "demon_overlap": 0.5,
        # Display parameters
        "rotate_flag": DEFAULT_PRESET["rotate_90"],
        "cmap_choice": DEFAULT_PRESET["cmap"],
        "gamma": DEFAULT_PRESET["gamma"],
        "gamma_lofar": DEFAULT_PRESET["gamma"],
        "gamma_demon": DEFAULT_PRESET["gamma"],
        # Preset management
        "preset_name": "",
        "preset_history": [],
        "history_select": 0,
    }
    # Only set values that don't already exist (preserves user changes)
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_preset(preset: Dict):
    """
    Load a saved preset configuration into the session state.
    
    Validates and applies all parameters from a preset dictionary.
    Ensures values are within valid ranges (e.g., noverlap < nperseg).
    Triggers a page rerun to update the UI with new values.
    
    Args:
        preset: Dictionary containing parameter values to apply
    """
    # Validate and set FFT window size
    nperseg_value = int(preset.get("nperseg", DEFAULT_PRESET["nperseg"]))
    st.session_state["window_select"] = preset.get("window", DEFAULT_PRESET["window"])
    # Only allow valid FFT sizes from the choices list
    st.session_state["nperseg_select"] = nperseg_value if nperseg_value in FFT_CHOICES else DEFAULT_PRESET["nperseg"]
    
    # Validate overlap (must be less than window size)
    max_noverlap = max(0, st.session_state["nperseg_select"] - 1)
    desired_noverlap = int(preset.get("noverlap", DEFAULT_PRESET["noverlap"]))
    st.session_state["noverlap_value"] = min(max_noverlap, max(0, desired_noverlap))
    
    # Apply remaining processing parameters
    st.session_state["decimation_factor"] = int(preset.get("decimation_factor", DEFAULT_PRESET["decimation_factor"]))
    st.session_state["floor_threshold"] = float(preset.get("floor_threshold", DEFAULT_PRESET["floor_threshold"]))
    st.session_state["floor_value"] = float(preset.get("floor_value", DEFAULT_PRESET["floor_value"]))
    st.session_state["eps_value"] = float(preset.get("eps", DEFAULT_PRESET["eps"]))
    
    # Apply TPSW parameters (handle None values)
    # Note: We set auto checkboxes instead of directly setting widget values
    # to avoid conflicts with Streamlit's widget state management
    tpsw_n = preset.get("tpsw_n", DEFAULT_PRESET["tpsw_n"])
    tpsw_p = preset.get("tpsw_p", DEFAULT_PRESET["tpsw_p"])
    tpsw_a = preset.get("tpsw_a", DEFAULT_PRESET["tpsw_a"])
    
    # Set auto checkboxes based on whether values are None
    st.session_state["tpsw_n_auto"] = (tpsw_n is None)
    st.session_state["tpsw_p_auto"] = (tpsw_p is None)
    st.session_state["tpsw_a_auto"] = (tpsw_a is None)
    
    # Only set widget values if not in auto mode (widgets will be created later)
    if tpsw_n is not None:
        st.session_state["tpsw_n"] = int(tpsw_n)
    elif "tpsw_n" in st.session_state:
        del st.session_state["tpsw_n"]
    
    if tpsw_p is not None:
        st.session_state["tpsw_p"] = int(tpsw_p)
    elif "tpsw_p" in st.session_state:
        del st.session_state["tpsw_p"]
    
    if tpsw_a is not None:
        st.session_state["tpsw_a"] = float(tpsw_a)
    elif "tpsw_a" in st.session_state:
        del st.session_state["tpsw_a"]
    
    # Apply Hough transform parameters
    hough_enhance = preset.get("hough_enhance", DEFAULT_PRESET["hough_enhance"])
    st.session_state["hough_enhance"] = bool(hough_enhance)
    st.session_state["hough_threshold_percentile"] = float(preset.get("hough_threshold_percentile", DEFAULT_PRESET["hough_threshold_percentile"]))
    st.session_state["hough_max_line_gap"] = int(preset.get("hough_max_line_gap", DEFAULT_PRESET["hough_max_line_gap"]))
    st.session_state["hough_enhancement_factor"] = float(preset.get("hough_enhancement_factor", DEFAULT_PRESET["hough_enhancement_factor"]))
    
    hough_min_line_length = preset.get("hough_min_line_length", DEFAULT_PRESET["hough_min_line_length"])
    st.session_state["hough_min_line_length_auto"] = (hough_min_line_length is None)
    if hough_min_line_length is not None:
        st.session_state["hough_min_line_length"] = int(hough_min_line_length)
    elif "hough_min_line_length" in st.session_state:
        del st.session_state["hough_min_line_length"]
    
    hough_theta_range = preset.get("hough_theta_range", DEFAULT_PRESET["hough_theta_range"])
    st.session_state["hough_theta_range_auto"] = (hough_theta_range is None)
    if hough_theta_range is not None:
        # Ensure it's a tuple
        if isinstance(hough_theta_range, list):
            hough_theta_range = tuple(hough_theta_range)
        st.session_state["hough_theta_range"] = hough_theta_range
    elif "hough_theta_range" in st.session_state:
        del st.session_state["hough_theta_range"]
    
    # Apply morphological parameters
    st.session_state["hough_use_morphology"] = bool(preset.get("hough_use_morphology", DEFAULT_PRESET["hough_use_morphology"]))
    st.session_state["hough_morph_kernel_size"] = int(preset.get("hough_morph_kernel_size", DEFAULT_PRESET["hough_morph_kernel_size"]))
    st.session_state["hough_morph_iterations"] = int(preset.get("hough_morph_iterations", DEFAULT_PRESET["hough_morph_iterations"]))
    st.session_state["hough_morph_operation"] = preset.get("hough_morph_operation", DEFAULT_PRESET["hough_morph_operation"])
    
    # Apply display parameters
    st.session_state["rotate_flag"] = bool(preset.get("rotate_90", DEFAULT_PRESET["rotate_90"]))
    cmap_value = preset.get("cmap", DEFAULT_PRESET["cmap"])
    # Validate colormap choice
    st.session_state["cmap_choice"] = cmap_value if cmap_value in CMAP_CHOICES else DEFAULT_PRESET["cmap"]
    st.session_state["gamma"] = float(preset.get("gamma", DEFAULT_PRESET["gamma"]))
    
    # Preserve preset name if provided
    st.session_state["preset_name"] = preset.get("name", st.session_state.get("preset_name", ""))
    
    # Trigger page rerun to update UI with new values
    st.experimental_rerun()


def main():
    st.set_page_config(page_title="LOFAR/DEMON Playground", layout="wide")
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=200)
    st.title("VIBE NASA - LOFAR/DEMON Spectrogram Playground")
    st.caption("Interactively explore how LOFAR and DEMON parameters affect the spectrogram.")
    init_session_state()

    sample_paths = list_sample_files(DEFAULT_AUDIO_ROOT)
    # The sidebar radio drives whether we pull from bundled WAVs or user uploads
    source_choice = st.sidebar.radio("Audio source", ("Sample file", "Upload WAV"), key="audio_source")

    waveform = None
    sample_rate = None
    selected_label = ""

    if source_choice == "Sample file":
        if not sample_paths:
            st.sidebar.warning("No WAV files found under the data directory.")
        else:
            labels = [Path(p).name for p in sample_paths]
            default_index = 0
            idx = st.sidebar.selectbox("Choose a file", range(len(labels)),
                                       format_func=lambda i: labels[i], index=default_index)
            selected_path = sample_paths[idx]
            waveform, sample_rate = load_audio_from_path(selected_path)
            selected_label = labels[idx]
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a WAV file", type=["wav"])
        if uploaded_file is not None:
            waveform, sample_rate = load_audio_from_upload(uploaded_file)
            selected_label = uploaded_file.name

    # ------------------------------------------------------------------
    # Parameter sidebar: everything below manipulates LOFAR/DEMON computation.
    # Streamlit keys map directly into session_state so presets can reload.
    # ------------------------------------------------------------------
    st.sidebar.header("Analysis method")
    show_lofar = st.sidebar.checkbox("LOFAR", value=st.session_state.get("show_lofar", True), key="show_lofar")
    show_demon = st.sidebar.checkbox("DEMON", value=st.session_state.get("show_demon", False), key="show_demon")
    
    # Ensure at least one method is selected
    if not show_lofar and not show_demon:
        st.sidebar.warning("Please select at least one analysis method.")
        show_lofar = True
        st.session_state["show_lofar"] = True
    
    # Determine which methods to show
    show_both = show_lofar and show_demon
    
    # Initialize variables to avoid undefined errors
    window = None
    nperseg = None
    noverlap = None
    decimation_factor = None
    floor_threshold = None
    floor_value = None
    eps = None
    tpsw_n = None
    tpsw_p = None
    tpsw_a = None
    hough_enhance = False
    hough_threshold_percentile = DEFAULT_PRESET["hough_threshold_percentile"]
    hough_min_line_length = None
    hough_max_line_gap = DEFAULT_PRESET["hough_max_line_gap"]
    hough_theta_range = None
    hough_enhancement_factor = DEFAULT_PRESET["hough_enhancement_factor"]
    hough_use_morphology = DEFAULT_PRESET["hough_use_morphology"]
    hough_morph_kernel_size = DEFAULT_PRESET["hough_morph_kernel_size"]
    hough_morph_iterations = DEFAULT_PRESET["hough_morph_iterations"]
    hough_morph_operation = DEFAULT_PRESET["hough_morph_operation"]
    demon_method = None
    demon_cutoff = None
    demon_low = None
    demon_high = None
    demon_window_size = None
    demon_overlap = None
    
    if show_lofar:
        st.sidebar.header("LOFAR parameters")
        window_index = WINDOW_CHOICES.index(st.session_state["window_select"]) if st.session_state["window_select"] in WINDOW_CHOICES else 0
        window = st.sidebar.selectbox("Window function", WINDOW_CHOICES, index=window_index, key="window_select")
        nperseg = st.sidebar.select_slider("FFT window size (nperseg)", options=FFT_CHOICES,
                                           value=st.session_state["nperseg_select"], key="nperseg_select")

        max_noverlap = nperseg - 1
        if st.session_state["noverlap_value"] > max_noverlap:
            st.session_state["noverlap_value"] = max_noverlap
        default_overlap = min(st.session_state["noverlap_value"], max_noverlap)
        noverlap = st.sidebar.slider(
            "Overlap (samples)",
            min_value=0,
            max_value=max_noverlap,
            value=default_overlap,
            key="noverlap_value",
        )

        decimation_factor = st.sidebar.slider("Decimation factor", min_value=1, max_value=12,
                                              value=st.session_state["decimation_factor"], step=1, key="decimation_factor")
        floor_threshold = st.sidebar.slider("Floor threshold (log10 power)", min_value=-3.0, max_value=0.0,
                                            value=st.session_state["floor_threshold"], step=0.1, key="floor_threshold")
        floor_value = st.sidebar.slider("Floor replacement value", min_value=-3.0, max_value=3.0,
                                        value=st.session_state["floor_value"], step=0.1, key="floor_value")
        eps = st.sidebar.number_input("Stability epsilon", min_value=1e-15, max_value=1e-3,
                                      value=st.session_state["eps_value"], format="%.1e", key="eps_value")
        
        # Advanced TPSW Settings (expandable)
        with st.sidebar.expander("Advanced TPSW Settings", expanded=False):
            st.caption("Two-Pass Split Window normalization parameters. Leave as 'Auto' to use adaptive defaults.")
            
            # TPSW n (half-window size)
            tpsw_n_auto = st.checkbox("Auto (adaptive)", 
                                      value=(st.session_state.get("tpsw_n") is None),
                                      key="tpsw_n_auto",
                                      help="Auto calculates n as ~2% of frequency bins. Uncheck to set manually.")
            if not tpsw_n_auto:
                # Estimate reasonable max based on typical frequency bins (nperseg/2 + 1)
                max_freq_bins = nperseg // 2 + 1
                tpsw_n = st.number_input(
                    "TPSW half-window size (n)",
                    min_value=1,
                    max_value=max(10, max_freq_bins // 2),
                    value=int(st.session_state.get("tpsw_n", max(1, max_freq_bins // 50))) if st.session_state.get("tpsw_n") is not None else max(1, max_freq_bins // 50),
                    step=1,
                    key="tpsw_n",
                    help="Half-width of averaging window. Larger = smoother background, may blur narrow tones."
                )
                # Widget automatically updates session state, so we just use the widget value
            else:
                tpsw_n = None
                # Clear the session state value when switching to auto (use pop to avoid error)
                if "tpsw_n" in st.session_state:
                    del st.session_state["tpsw_n"]
            
            # TPSW p (guard band size)
            tpsw_p_auto = st.checkbox("Auto guard band (p)", 
                                      value=(st.session_state.get("tpsw_p") is None),
                                      key="tpsw_p_auto",
                                      help="Auto calculates p as n/8. Uncheck to set manually.")
            if not tpsw_p_auto:
                # Use current tpsw_n value or estimate from nperseg if tpsw_n is None
                # Get tpsw_n from session state if widget was created, otherwise use current variable
                current_n_value = st.session_state.get("tpsw_n", tpsw_n) if tpsw_n is not None else (nperseg // 2 + 1) // 50
                current_n_value = current_n_value if current_n_value is not None else (nperseg // 2 + 1) // 50
                max_p = max(1, current_n_value // 2)
                tpsw_p = st.number_input(
                    "TPSW guard band size (p)",
                    min_value=0,
                    max_value=max_p,
                    value=int(st.session_state.get("tpsw_p", max(1, max_p // 8))) if st.session_state.get("tpsw_p") is not None else max(1, max_p // 8),
                    step=1,
                    key="tpsw_p",
                    help="Bins around tones excluded from averaging. Larger = better protection for wider tones."
                )
                # Widget automatically updates session state, so we just use the widget value
            else:
                tpsw_p = None
                # Clear the session state value when switching to auto
                if "tpsw_p" in st.session_state:
                    del st.session_state["tpsw_p"]
            
            # TPSW a (threshold multiplier)
            tpsw_a_auto = st.checkbox("Auto threshold (a = 2.0)", 
                                      value=(st.session_state.get("tpsw_a") is None),
                                      key="tpsw_a_auto",
                                      help="Use default threshold multiplier of 2.0. Uncheck to set manually.")
            if not tpsw_a_auto:
                # Ensure we have a valid float value, never None
                tpsw_a_default = 2.0
                tpsw_a_value = st.session_state.get("tpsw_a")
                if tpsw_a_value is None or not isinstance(tpsw_a_value, (int, float)):
                    tpsw_a_value = tpsw_a_default
                else:
                    tpsw_a_value = float(tpsw_a_value)
                
                tpsw_a = st.slider(
                    "TPSW threshold multiplier (a)",
                    min_value=1.0,
                    max_value=4.0,
                    value=tpsw_a_value,
                    step=0.1,
                    key="tpsw_a",
                    help="Peak clipping threshold. Larger = more conservative (preserves detail), smaller = more aggressive (suppresses noise)."
                )
                # Widget automatically updates session state, so we just use the widget value
            else:
                tpsw_a = None
                # Clear the session state value when switching to auto
                if "tpsw_a" in st.session_state:
                    del st.session_state["tpsw_a"]
        
        # Hough Transform Line Enhancement Settings (expandable)
        with st.sidebar.expander("Hough Transform Line Enhancement", expanded=False):
            st.caption("Enhance linear features (tonal lines) in the spectrogram using Hough transform.")
            
            hough_enhance = st.checkbox(
                "Enable Hough line enhancement",
                value=st.session_state.get("hough_enhance", False),
                key="hough_enhance",
                help="Detect and enhance linear features (tonal lines) in the spectrogram."
            )
            
            if hough_enhance:
                hough_threshold_percentile = st.slider(
                    "Threshold percentile",
                    min_value=50.0,
                    max_value=95.0,
                    value=float(st.session_state.get("hough_threshold_percentile", 75.0)),
                    step=1.0,
                    key="hough_threshold_percentile",
                    help="Percentile threshold for edge detection. Higher values detect only stronger lines."
                )
                
                hough_min_line_length_auto = st.checkbox(
                    "Auto min line length",
                    value=(st.session_state.get("hough_min_line_length") is None),
                    key="hough_min_line_length_auto",
                    help="Auto calculates minimum line length as 5% of time dimension."
                )
                
                if not hough_min_line_length_auto:
                    # Estimate reasonable max based on typical time bins
                    # We'll use a default based on nperseg if available
                    max_time_bins = 10000  # Reasonable default
                    hough_min_line_length = st.number_input(
                        "Minimum line length (pixels)",
                        min_value=5,
                        max_value=max_time_bins,
                        value=int(st.session_state.get("hough_min_line_length", 50)) if st.session_state.get("hough_min_line_length") is not None else 50,
                        step=5,
                        key="hough_min_line_length",
                        help="Minimum length of detected lines in pixels."
                    )
                else:
                    hough_min_line_length = None
                    if "hough_min_line_length" in st.session_state:
                        del st.session_state["hough_min_line_length"]
                
                hough_max_line_gap = st.slider(
                    "Max line gap",
                    min_value=1,
                    max_value=20,
                    value=int(st.session_state.get("hough_max_line_gap", 5)),
                    step=1,
                    key="hough_max_line_gap",
                    help="Maximum gap between line segments to be connected."
                )
                
                hough_theta_range_auto = st.checkbox(
                    "Detect all angles",
                    value=(st.session_state.get("hough_theta_range") is None),
                    key="hough_theta_range_auto",
                    help="If unchecked, limit detection to vertical lines (85-95 degrees)."
                )
                
                if not hough_theta_range_auto:
                    default_range = st.session_state.get("hough_theta_range", (85.0, 95.0))
                    if default_range is None:
                        default_range = (85.0, 95.0)
                    theta_min = st.number_input(
                        "Min angle (degrees)",
                        min_value=0.0,
                        max_value=180.0,
                        value=float(default_range[0]) if isinstance(default_range, (tuple, list)) and len(default_range) > 0 else 85.0,
                        step=1.0,
                        key="hough_theta_min",
                        help="Minimum angle in degrees for line detection (85-95 for vertical lines)."
                    )
                    theta_max = st.number_input(
                        "Max angle (degrees)",
                        min_value=0.0,
                        max_value=180.0,
                        value=float(default_range[1]) if isinstance(default_range, (tuple, list)) and len(default_range) > 1 else 95.0,
                        step=1.0,
                        key="hough_theta_max",
                        help="Maximum angle in degrees for line detection (85-95 for vertical lines)."
                    )
                    if theta_min >= theta_max:
                        st.warning("Min angle must be less than max angle. Using default range (-5, 5).")
                        hough_theta_range = (-5.0, 5.0)
                    else:
                        hough_theta_range = (theta_min, theta_max)
                    st.session_state["hough_theta_range"] = hough_theta_range
                else:
                    hough_theta_range = None
                    if "hough_theta_range" in st.session_state:
                        del st.session_state["hough_theta_range"]
                
                hough_enhancement_factor = st.slider(
                    "Enhancement factor",
                    min_value=0.5,
                    max_value=3.0,
                    value=float(st.session_state.get("hough_enhancement_factor", 1.5)),
                    step=0.1,
                    key="hough_enhancement_factor",
                    help="Factor by which detected lines are enhanced. >1.0 brightens, <1.0 darkens."
                )
                
                # Morphological operations
                st.markdown("---")
                st.caption("**Morphological Operations**: Remove small lines and merge nearby ones")
                hough_use_morphology = st.checkbox(
                    "Enable morphological operations",
                    value=st.session_state.get("hough_use_morphology", False),
                    key="hough_use_morphology",
                    help="Use erosion/dilation to filter small lines and merge nearby ones."
                )
                
                if hough_use_morphology:
                    hough_morph_operation = st.selectbox(
                        "Operation type",
                        ["closing", "opening", "both", "erosion", "dilation"],
                        index=["closing", "opening", "both", "erosion", "dilation"].index(
                            st.session_state.get("hough_morph_operation", "closing")
                        ),
                        key="hough_morph_operation",
                        help="closing: merges nearby lines | opening: removes small lines | both: does both | erosion: shrinks lines | dilation: expands lines"
                    )
                    
                    hough_morph_kernel_size = st.slider(
                        "Kernel size",
                        min_value=3,
                        max_value=15,
                        value=int(st.session_state.get("hough_morph_kernel_size", 3)),
                        step=2,
                        key="hough_morph_kernel_size",
                        help="Size of morphological kernel (must be odd). Larger = more aggressive filtering/merging."
                    )
                    
                    hough_morph_iterations = st.slider(
                        "Iterations",
                        min_value=1,
                        max_value=5,
                        value=int(st.session_state.get("hough_morph_iterations", 1)),
                        step=1,
                        key="hough_morph_iterations",
                        help="Number of times to apply the operation. More iterations = stronger effect."
                    )
                else:
                    hough_use_morphology = False
                    hough_morph_kernel_size = DEFAULT_PRESET["hough_morph_kernel_size"]
                    hough_morph_iterations = DEFAULT_PRESET["hough_morph_iterations"]
                    hough_morph_operation = DEFAULT_PRESET["hough_morph_operation"]
            else:
                # Set defaults when disabled
                hough_threshold_percentile = DEFAULT_PRESET["hough_threshold_percentile"]
                hough_min_line_length = None
                hough_max_line_gap = DEFAULT_PRESET["hough_max_line_gap"]
                hough_theta_range = None
                hough_enhancement_factor = DEFAULT_PRESET["hough_enhancement_factor"]
                hough_use_morphology = False
                hough_morph_kernel_size = DEFAULT_PRESET["hough_morph_kernel_size"]
                hough_morph_iterations = DEFAULT_PRESET["hough_morph_iterations"]
                hough_morph_operation = DEFAULT_PRESET["hough_morph_operation"]
    else:
        # When LOFAR is not shown, set defaults
        hough_enhance = False
        hough_threshold_percentile = DEFAULT_PRESET["hough_threshold_percentile"]
        hough_min_line_length = None
        hough_max_line_gap = DEFAULT_PRESET["hough_max_line_gap"]
        hough_theta_range = None
        hough_enhancement_factor = DEFAULT_PRESET["hough_enhancement_factor"]
    
    if show_demon:
        st.sidebar.header("DEMON parameters")
        demon_method = st.sidebar.selectbox(
            "DEMON method",
            ("square_law", "hilbert_detector"),
            index=0 if st.session_state.get("demon_method", "square_law") == "square_law" else 1,
            key="demon_method"
        )
        
        # Get sample rate for validation (use a reasonable default if not loaded yet)
        if waveform is not None and sample_rate is not None:
            max_freq = sample_rate / 2.1  # Nyquist with small safety margin
            if max_freq < 1000:
                max_freq = 1000.0
        else:
            max_freq = 100000.0  # Default high value when audio not loaded yet
        
        demon_cutoff = st.sidebar.number_input(
            "Cutoff frequency (Hz)",
            min_value=100.0,
            max_value=min(10000.0, max_freq),
            value=float(st.session_state.get("demon_cutoff", 1000.0)),
            step=100.0,
            key="demon_cutoff",
            help=f"Must be less than Nyquist frequency ({max_freq*2:.0f} Hz / 2 = {max_freq:.0f} Hz)"
        )
        demon_low = st.sidebar.number_input(
            "Low frequency (Hz)",
            min_value=1000.0,
            max_value=max_freq,
            value=float(st.session_state.get("demon_low", min(20000.0, max_freq * 0.4))),
            step=1000.0,
            key="demon_low",
            help=f"Must be less than Nyquist frequency ({max_freq:.0f} Hz)"
        )
        demon_high = st.sidebar.number_input(
            "High frequency (Hz)",
            min_value=1000.0,
            max_value=max_freq,
            value=float(st.session_state.get("demon_high", min(30000.0, max_freq * 0.6))),
            step=1000.0,
            key="demon_high",
            help=f"Must be less than Nyquist frequency ({max_freq:.0f} Hz)"
        )
        demon_window_size_input = st.sidebar.number_input(
            "Window size (samples, 0 for auto)",
            min_value=0,
            max_value=1000000,
            value=int(st.session_state.get("demon_window_size", 0)) if st.session_state.get("demon_window_size") is not None else 0,
            step=1000,
            key="demon_window_size_input"
        )
        demon_window_size = None if demon_window_size_input == 0 else demon_window_size_input
        demon_overlap = st.sidebar.slider(
            "Overlap fraction",
            min_value=0.0,
            max_value=0.95,
            value=float(st.session_state.get("demon_overlap", 0.5)),
            step=0.05,
            key="demon_overlap"
        )
    
    # Display parameters (shared for both methods)
    st.sidebar.header("Display parameters")
    rotate_90 = st.sidebar.checkbox("Rotate 90° (time on Y axis)", value=st.session_state["rotate_flag"], key="rotate_flag")
    cmap_choice = st.sidebar.selectbox(
        "Colormap",
        CMAP_CHOICES,
        index=CMAP_CHOICES.index(st.session_state["cmap_choice"]) if st.session_state["cmap_choice"] in CMAP_CHOICES else 0,
        key="cmap_choice",
    )
    if cmap_choice in CUSTOM_CMAPS:
        cmap = CUSTOM_CMAPS[cmap_choice]
    else:
        cmap = plt.get_cmap(cmap_choice)
    
    # Gamma correction - separate for each method if both are selected
    if show_both:
        gamma_lofar = st.sidebar.slider("LOFAR Gamma correction", min_value=0.1, max_value=3.0,
                                       value=st.session_state.get("gamma_lofar", DEFAULT_PRESET["gamma"]), step=0.1, key="gamma_lofar",
                                       help="Gamma correction for LOFAR. Values < 1.0 brighten, values > 1.0 darken.")
        gamma_demon = st.sidebar.slider("DEMON Gamma correction", min_value=0.1, max_value=3.0,
                                       value=st.session_state.get("gamma_demon", DEFAULT_PRESET["gamma"]), step=0.1, key="gamma_demon",
                                       help="Gamma correction for DEMON. Values < 1.0 brighten, values > 1.0 darken.")
        gamma = gamma_lofar  # Default for single method case
    else:
        gamma = st.sidebar.slider("Gamma correction", min_value=0.1, max_value=3.0,
                                  value=st.session_state["gamma"], step=0.1, key="gamma",
                                  help="Gamma correction factor. Values < 1.0 brighten the image, values > 1.0 darken it.")
        gamma_lofar = gamma
        gamma_demon = gamma

    # Build current_params based on what's selected
    current_params = {}
    if show_lofar:
        # Get TPSW values from session state (None if auto is enabled)
        tpsw_n_param = None if st.session_state.get("tpsw_n_auto", True) else st.session_state.get("tpsw_n")
        tpsw_p_param = None if st.session_state.get("tpsw_p_auto", True) else st.session_state.get("tpsw_p")
        tpsw_a_param = None if st.session_state.get("tpsw_a_auto", True) else st.session_state.get("tpsw_a")
        
        # Get Hough values from session state
        hough_theta_range_param = st.session_state.get("hough_theta_range") if st.session_state.get("hough_theta_range") is not None else None
        
        current_params.update({
            "window": window,
            "nperseg": nperseg,
            "noverlap": noverlap,
            "decimation_factor": decimation_factor,
            "floor_threshold": floor_threshold,
            "floor_value": floor_value,
            "eps": eps,
            "tpsw_n": tpsw_n_param,
            "tpsw_p": tpsw_p_param,
            "tpsw_a": tpsw_a_param,
            "hough_enhance": hough_enhance,
            "hough_threshold_percentile": hough_threshold_percentile,
            "hough_min_line_length": hough_min_line_length,
            "hough_max_line_gap": hough_max_line_gap,
            "hough_theta_range": hough_theta_range_param,
            "hough_enhancement_factor": hough_enhancement_factor,
            "hough_use_morphology": hough_use_morphology,
            "hough_morph_kernel_size": hough_morph_kernel_size,
            "hough_morph_iterations": hough_morph_iterations,
            "hough_morph_operation": hough_morph_operation,
        })
    if show_demon:
        current_params.update({
            "method": demon_method,
            "cutoff": demon_cutoff,
            "high": demon_high,
            "low": demon_low,
            "window_size": demon_window_size,
            "overlap": demon_overlap,
        })
    current_params.update({
        "rotate_90": rotate_90,
        "cmap": cmap_choice,
        "gamma": gamma,
    })
    if show_both:
        current_params.update({
            "gamma_lofar": gamma_lofar,
            "gamma_demon": gamma_demon,
        })

    if waveform is None or sample_rate is None:
        st.info("Select or upload an audio file to begin.")
        return

    duration = len(waveform) / sample_rate
    st.subheader("Audio preview")
    st.write(f"**Selection:** {selected_label} | **Sample rate:** {sample_rate:,} Hz | "
             f"**Duration:** {duration:.2f} s")
    st.audio(waveform_to_wav_bytes(waveform, sample_rate))

    # Compute LOFAR if selected
    lofar_result = None
    if show_lofar:
        if show_both:
            st.markdown("### LOFAR Parameters")
        hop_length = nperseg - noverlap
        st.markdown(
            f"- **Window:** `{window}` | **nperseg:** {nperseg} samples\n"
            f"- **Overlap:** {noverlap} samples | **Hop length:** {hop_length} samples\n"
            f"- **Decimation factor:** {decimation_factor}"
        )

        with st.spinner("Computing LOFAR spectrogram..."):
            try:
                # Get TPSW values: use session state if auto is disabled, otherwise None
                tpsw_n_val = None if st.session_state.get("tpsw_n_auto", True) else st.session_state.get("tpsw_n")
                tpsw_p_val = None if st.session_state.get("tpsw_p_auto", True) else st.session_state.get("tpsw_p")
                tpsw_a_val = None if st.session_state.get("tpsw_a_auto", True) else st.session_state.get("tpsw_a")
                
                # Get Hough values
                hough_theta_range_val = st.session_state.get("hough_theta_range") if st.session_state.get("hough_theta_range") is not None else None
                hough_min_line_length_val = st.session_state.get("hough_min_line_length") if st.session_state.get("hough_min_line_length") is not None else None
                
                t_lofar, f_lofar, spectrogram_lofar, hough_lines_lofar = render_lofargram(
                    waveform,
                    sample_rate,
                    decimation_factor=decimation_factor,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    floor_threshold=floor_threshold,
                    floor_value=floor_value,
                    eps=eps,
                    tpsw_n=tpsw_n_val,
                    tpsw_p=tpsw_p_val,
                    tpsw_a=tpsw_a_val,
                    hough_enhance=hough_enhance,
                    hough_threshold_percentile=hough_threshold_percentile,
                    hough_min_line_length=hough_min_line_length_val,
                    hough_max_line_gap=hough_max_line_gap,
                    hough_theta_range=hough_theta_range_val,
                    hough_enhancement_factor=hough_enhancement_factor,
                    hough_use_morphology=hough_use_morphology,
                    hough_morph_kernel_size=hough_morph_kernel_size,
                    hough_morph_iterations=hough_morph_iterations,
                    hough_morph_operation=hough_morph_operation,
                )
                if t_lofar.size == 0 or f_lofar.size == 0 or spectrogram_lofar.size == 0:
                    st.error("LOFAR parameters produced an empty spectrogram. Try adjusting the parameters or using a longer audio clip.")
                else:
                    lofar_result = (t_lofar, f_lofar, spectrogram_lofar, hough_lines_lofar)
            except Exception as e:
                st.error(f"LOFAR processing error: {str(e)}")
                lofar_result = None

    # Compute DEMON if selected
    demon_result = None
    if show_demon:
        if show_both:
            st.markdown("### DEMON Parameters")
        window_size_display = demon_window_size if demon_window_size is not None else sample_rate
        hop_size = int(window_size_display * (1 - demon_overlap))
        st.markdown(
            f"- **Method:** `{demon_method}` | **Window size:** {window_size_display} samples\n"
            f"- **Overlap:** {demon_overlap:.1%} | **Hop size:** {hop_size} samples\n"
            f"- **Frequency range:** {demon_low:.0f} - {demon_high:.0f} Hz | **Cutoff:** {demon_cutoff:.0f} Hz"
        )

        with st.spinner("Computing DEMONgram..."):
            try:
                t_demon, f_demon, spectrogram_demon = render_demongram(
                    waveform,
                    sample_rate,
                    method=demon_method,
                    cutoff=demon_cutoff,
                    high=demon_high,
                    low=demon_low,
                    window_size=demon_window_size,
                    overlap=demon_overlap,
                )
                if t_demon.size == 0 or f_demon.size == 0 or spectrogram_demon.size == 0:
                    st.error("DEMON parameters produced an empty spectrogram. Try adjusting the parameters or using a longer audio clip.")
                else:
                    demon_result = (t_demon, f_demon, spectrogram_demon)
            except ValueError as e:
                st.error(f"DEMON processing error: {str(e)}\n\n"
                        f"**Tip:** Make sure your frequency settings are valid:\n"
                        f"- High frequency must be < {sample_rate/2:.0f} Hz (Nyquist frequency)\n"
                        f"- Low frequency must be < {sample_rate/2:.0f} Hz\n"
                        f"- Low frequency must be < High frequency")
                demon_result = None
            except Exception as e:
                st.error(f"DEMON processing error: {str(e)}")
                demon_result = None

    # Helper function to render a spectrogram
    def render_spectrogram(t, f, spectrogram_matrix, title, method_name, col=None, gamma_val=None, hough_lines=None):
        """Render a single spectrogram plot with optional red lines from Hough transform"""
        if gamma_val is None:
            gamma_val = gamma
        
        fig, ax = plt.subplots(figsize=(12, 4))
        time_extent = (t[0], t[-1] if len(t) > 1 else t[0] + 1e-6)
        freq_extent = (f[0], f[-1] if len(f) > 1 else f[0] + 1e-6)
        matrix_to_show = spectrogram_matrix.T if rotate_90 else spectrogram_matrix
        x_extent = freq_extent if rotate_90 else time_extent
        y_extent = time_extent if rotate_90 else freq_extent
        
        # Apply gamma correction
        vmax = float(np.max(spectrogram_matrix))
        vmin = float(np.min(spectrogram_matrix))
        # Normalize to [0, 1] range
        matrix_normalized = (matrix_to_show - vmin) / (vmax - vmin + 1e-10)
        # Apply gamma correction: output = input^(1/gamma)
        matrix_gamma_corrected = np.power(np.clip(matrix_normalized, 0, 1), 1.0 / gamma_val)
        
        im = ax.imshow(
            matrix_gamma_corrected,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
            vmin=0.0,
            vmax=1.0,
        )
        
        # Draw red lines if Hough lines are provided
        if hough_lines is not None and len(hough_lines) > 0:
            # Convert pixel coordinates to data coordinates
            # hough_lines are in (x, y) format where x is time index, y is frequency index
            # Original spectrogram shape: (freq_bins, time_bins)
            # Vertical lines: x1 ≈ x2 (constant time), y1 and y2 vary (varying frequency)
            freq_bins, time_bins = spectrogram_matrix.shape
            
            for line in hough_lines:
                x1, y1, x2, y2 = line
                # x is time index, y is frequency index in the original spectrogram
                # Convert to data coordinates
                if rotate_90:
                    # When rotated/transposed for waterfall:
                    # - x_extent is frequency (x-axis shows frequency)
                    # - y_extent is time (y-axis shows time)
                    # - Vertical lines (constant time in original) should appear vertical (constant y/time in display)
                    # - So swap: use y (freq) for x_data, use x (time) for y_data
                    x1_data = np.interp(y1, [0, freq_bins-1], [x_extent[0], x_extent[1]])  # freq -> x
                    y1_data = np.interp(x1, [0, time_bins-1], [y_extent[0], y_extent[1]])  # time -> y
                    x2_data = np.interp(y2, [0, freq_bins-1], [x_extent[0], x_extent[1]])  # freq -> x
                    y2_data = np.interp(x2, [0, time_bins-1], [y_extent[0], y_extent[1]])  # time -> y
                else:
                    # When not rotated, x is time, y is frequency
                    x1_data = np.interp(x1, [0, time_bins-1], [x_extent[0], x_extent[1]])
                    y1_data = np.interp(y1, [0, freq_bins-1], [y_extent[0], y_extent[1]])
                    x2_data = np.interp(x2, [0, time_bins-1], [x_extent[0], x_extent[1]])
                    y2_data = np.interp(y2, [0, freq_bins-1], [y_extent[0], y_extent[1]])
                
                # Draw red line
                ax.plot([x1_data, x2_data], [y1_data, y2_data], 'r-', linewidth=2, alpha=0.8)
        
        ax.set_xlabel("Frequency (Hz)" if rotate_90 else "Time (s)")
        ax.set_ylabel("Time (s)" if rotate_90 else "Frequency (Hz)")
        ax.set_title(f"{title}" + (" (rotated)" if rotate_90 else ""))
        colorbar_label = "Relative power (log10)" if method_name == "LOFAR" else "Envelope amplitude"
        plt.colorbar(im, ax=ax, label=colorbar_label)
        
        if col is not None:
            try:
                col.pyplot(fig, clear_figure=True)
            except TypeError:
                # Older Streamlit versions may not support clear_figure
                col.pyplot(fig)
        else:
            st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        
        # Create download buffer
        img_buffer = io.BytesIO()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(
            matrix_gamma_corrected,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
            vmin=0.0,
            vmax=1.0,
        )
        # Draw red lines in download buffer too
        if hough_lines is not None and len(hough_lines) > 0:
            freq_bins, time_bins = spectrogram_matrix.shape
            for line in hough_lines:
                x1, y1, x2, y2 = line
                if rotate_90:
                    # Swap coordinates for waterfall display
                    x1_data = np.interp(y1, [0, freq_bins-1], [x_extent[0], x_extent[1]])  # freq -> x
                    y1_data = np.interp(x1, [0, time_bins-1], [y_extent[0], y_extent[1]])  # time -> y
                    x2_data = np.interp(y2, [0, freq_bins-1], [x_extent[0], x_extent[1]])  # freq -> x
                    y2_data = np.interp(x2, [0, time_bins-1], [y_extent[0], y_extent[1]])  # time -> y
                else:
                    x1_data = np.interp(x1, [0, time_bins-1], [x_extent[0], x_extent[1]])
                    y1_data = np.interp(y1, [0, freq_bins-1], [y_extent[0], y_extent[1]])
                    x2_data = np.interp(x2, [0, time_bins-1], [x_extent[0], x_extent[1]])
                    y2_data = np.interp(y2, [0, freq_bins-1], [y_extent[0], y_extent[1]])
                ax.plot([x1_data, x2_data], [y1_data, y2_data], 'r-', linewidth=2, alpha=0.8)
        ax.set_axis_off()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        img_buffer.seek(0)
        return img_buffer
    
    # Helper function to render side-by-side spectrograms with shared y-axis
    def render_side_by_side(lofar_data, demon_data):
        """Render LOFAR and DEMON spectrograms side by side with shared y-axis"""
        t_lofar, f_lofar, spectrogram_lofar = lofar_data[:3]
        hough_lines_lofar = lofar_data[3] if len(lofar_data) > 3 else None
        t_demon, f_demon, spectrogram_demon = demon_data
        
        # Calculate shared y-axis extent (frequency axis when not rotated, time axis when rotated)
        if rotate_90:
            # When rotated, y-axis is time
            y_min = min(t_lofar[0], t_demon[0])
            y_max = max(t_lofar[-1] if len(t_lofar) > 1 else t_lofar[0] + 1e-6,
                       t_demon[-1] if len(t_demon) > 1 else t_demon[0] + 1e-6)
        else:
            # When not rotated, y-axis is frequency
            y_min = min(f_lofar[0], f_demon[0])
            y_max = max(f_lofar[-1] if len(f_lofar) > 1 else f_lofar[0] + 1e-6,
                       f_demon[-1] if len(f_demon) > 1 else f_demon[0] + 1e-6)
        
        # Create figure with two subplots sharing y-axis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Prepare LOFAR data
        time_extent_lofar = (t_lofar[0], t_lofar[-1] if len(t_lofar) > 1 else t_lofar[0] + 1e-6)
        freq_extent_lofar = (f_lofar[0], f_lofar[-1] if len(f_lofar) > 1 else f_lofar[0] + 1e-6)
        matrix_lofar = spectrogram_lofar.T if rotate_90 else spectrogram_lofar
        x_extent_lofar = freq_extent_lofar if rotate_90 else time_extent_lofar
        
        # Prepare DEMON data
        time_extent_demon = (t_demon[0], t_demon[-1] if len(t_demon) > 1 else t_demon[0] + 1e-6)
        freq_extent_demon = (f_demon[0], f_demon[-1] if len(f_demon) > 1 else f_demon[0] + 1e-6)
        matrix_demon = spectrogram_demon.T if rotate_90 else spectrogram_demon
        x_extent_demon = freq_extent_demon if rotate_90 else time_extent_demon
        
        # Apply gamma correction to LOFAR
        vmax_lofar = float(np.max(spectrogram_lofar))
        vmin_lofar = float(np.min(spectrogram_lofar))
        matrix_normalized_lofar = (matrix_lofar - vmin_lofar) / (vmax_lofar - vmin_lofar + 1e-10)
        matrix_gamma_lofar = np.power(np.clip(matrix_normalized_lofar, 0, 1), 1.0 / gamma_lofar)
        
        # Apply gamma correction to DEMON
        vmax_demon = float(np.max(spectrogram_demon))
        vmin_demon = float(np.min(spectrogram_demon))
        matrix_normalized_demon = (matrix_demon - vmin_demon) / (vmax_demon - vmin_demon + 1e-10)
        matrix_gamma_demon = np.power(np.clip(matrix_normalized_demon, 0, 1), 1.0 / gamma_demon)
        
        # Plot LOFAR
        im1 = ax1.imshow(
            matrix_gamma_lofar,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[x_extent_lofar[0], x_extent_lofar[1], y_min, y_max],
            vmin=0.0,
            vmax=1.0,
        )
        # Draw red lines on LOFAR if available
        if hough_lines_lofar is not None and len(hough_lines_lofar) > 0:
            freq_bins, time_bins = spectrogram_lofar.shape
            for line in hough_lines_lofar:
                x1, y1, x2, y2 = line
                if rotate_90:
                    # Swap coordinates for waterfall display
                    # x_extent_lofar is frequency, y_min/y_max is time
                    x1_data = np.interp(y1, [0, freq_bins-1], [x_extent_lofar[0], x_extent_lofar[1]])  # freq -> x
                    y1_data = np.interp(x1, [0, time_bins-1], [y_min, y_max])  # time -> y
                    x2_data = np.interp(y2, [0, freq_bins-1], [x_extent_lofar[0], x_extent_lofar[1]])  # freq -> x
                    y2_data = np.interp(x2, [0, time_bins-1], [y_min, y_max])  # time -> y
                else:
                    x1_data = np.interp(x1, [0, time_bins-1], [x_extent_lofar[0], x_extent_lofar[1]])
                    y1_data = np.interp(y1, [0, freq_bins-1], [y_min, y_max])
                    x2_data = np.interp(x2, [0, time_bins-1], [x_extent_lofar[0], x_extent_lofar[1]])
                    y2_data = np.interp(y2, [0, freq_bins-1], [y_min, y_max])
                ax1.plot([x1_data, x2_data], [y1_data, y2_data], 'r-', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel("Frequency (Hz)" if rotate_90 else "Time (s)")
        ax1.set_ylabel("Time (s)" if rotate_90 else "Frequency (Hz)")
        ax1.set_title("LOFAR Spectrogram" + (" (rotated)" if rotate_90 else ""))
        plt.colorbar(im1, ax=ax1, label="Relative power (log10)")
        
        # Plot DEMON
        im2 = ax2.imshow(
            matrix_gamma_demon,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[x_extent_demon[0], x_extent_demon[1], y_min, y_max],
            vmin=0.0,
            vmax=1.0,
        )
        ax2.set_xlabel("Frequency (Hz)" if rotate_90 else "Time (s)")
        ax2.set_title("DEMON Spectrogram" + (" (rotated)" if rotate_90 else ""))
        plt.colorbar(im2, ax=ax2, label="Envelope amplitude")
        
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        
        # Create download buffers for individual spectrograms
        img_buffer_lofar = io.BytesIO()
        fig_dl, ax_dl = plt.subplots(figsize=(12, 4))
        ax_dl.imshow(
            matrix_gamma_lofar,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[x_extent_lofar[0], x_extent_lofar[1], y_min, y_max],
            vmin=0.0,
            vmax=1.0,
        )
        # Draw red lines in download buffer too
        if hough_lines_lofar is not None and len(hough_lines_lofar) > 0:
            freq_bins, time_bins = spectrogram_lofar.shape
            for line in hough_lines_lofar:
                x1, y1, x2, y2 = line
                if rotate_90:
                    # Swap coordinates for waterfall display
                    x1_data = np.interp(y1, [0, freq_bins-1], [x_extent_lofar[0], x_extent_lofar[1]])  # freq -> x
                    y1_data = np.interp(x1, [0, time_bins-1], [y_min, y_max])  # time -> y
                    x2_data = np.interp(y2, [0, freq_bins-1], [x_extent_lofar[0], x_extent_lofar[1]])  # freq -> x
                    y2_data = np.interp(x2, [0, time_bins-1], [y_min, y_max])  # time -> y
                else:
                    x1_data = np.interp(x1, [0, time_bins-1], [x_extent_lofar[0], x_extent_lofar[1]])
                    y1_data = np.interp(y1, [0, freq_bins-1], [y_min, y_max])
                    x2_data = np.interp(x2, [0, time_bins-1], [x_extent_lofar[0], x_extent_lofar[1]])
                    y2_data = np.interp(y2, [0, freq_bins-1], [y_min, y_max])
                ax_dl.plot([x1_data, x2_data], [y1_data, y2_data], 'r-', linewidth=2, alpha=0.8)
        ax_dl.set_axis_off()
        plt.savefig(img_buffer_lofar, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig_dl)
        img_buffer_lofar.seek(0)
        
        img_buffer_demon = io.BytesIO()
        fig_dl, ax_dl = plt.subplots(figsize=(12, 4))
        ax_dl.imshow(
            matrix_gamma_demon,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[x_extent_demon[0], x_extent_demon[1], y_min, y_max],
            vmin=0.0,
            vmax=1.0,
        )
        ax_dl.set_axis_off()
        plt.savefig(img_buffer_demon, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig_dl)
        img_buffer_demon.seek(0)
        
        return img_buffer_lofar, img_buffer_demon

    # Display spectrograms
    if show_both:
        # Side by side display with shared y-axis
        if lofar_result is not None and demon_result is not None:
            st.subheader("LOFAR & DEMON Spectrograms (Side by Side)")
            img_buffer_lofar, img_buffer_demon = render_side_by_side(lofar_result, demon_result)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download LOFAR (PNG)",
                    data=img_buffer_lofar,
                    file_name=f"{Path(selected_label).stem}_lofar.png",
                    mime="image/png",
                    key="download_lofar"
                )
            with col2:
                st.download_button(
                    label="Download DEMON (PNG)",
                    data=img_buffer_demon,
                    file_name=f"{Path(selected_label).stem}_demon.png",
                    mime="image/png",
                    key="download_demon"
                )
        elif lofar_result is not None:
            st.warning("LOFAR computed successfully, but DEMON computation failed or produced empty result.")
            st.subheader("LOFARgram")
            hough_lines_lofar = lofar_result[3] if len(lofar_result) > 3 else None
            img_buffer_lofar = render_spectrogram(
                lofar_result[0], lofar_result[1], lofar_result[2],
                "LOFAR Spectrogram", "LOFAR", gamma_val=gamma_lofar, hough_lines=hough_lines_lofar
            )
            st.download_button(
                label="Download LOFAR (PNG)",
                data=img_buffer_lofar,
                file_name=f"{Path(selected_label).stem}_lofar.png",
                mime="image/png",
                key="download_lofar"
            )
        elif demon_result is not None:
            st.warning("DEMON computed successfully, but LOFAR computation failed or produced empty result.")
            st.subheader("DEMONgram")
            img_buffer_demon = render_spectrogram(
                demon_result[0], demon_result[1], demon_result[2],
                "DEMON Spectrogram", "DEMON", gamma_val=gamma_demon, hough_lines=None
            )
            st.download_button(
                label="Download DEMON (PNG)",
                data=img_buffer_demon,
                file_name=f"{Path(selected_label).stem}_demon.png",
                mime="image/png",
                key="download_demon"
            )
        else:
            st.error("Both LOFAR and DEMON computations failed or produced empty results.")
    else:
        # Single display
        if show_lofar and lofar_result is not None:
            st.subheader("LOFARgram")
            hough_lines_lofar = lofar_result[3] if len(lofar_result) > 3 else None
            img_buffer = render_spectrogram(
                lofar_result[0], lofar_result[1], lofar_result[2],
                "LOFAR Spectrogram", "LOFAR", gamma_val=gamma_lofar, hough_lines=hough_lines_lofar
            )
            st.download_button(
                label="Download spectrogram (PNG)",
                data=img_buffer,
                file_name=f"{Path(selected_label).stem}_lofar.png",
                mime="image/png",
            )
        elif show_demon and demon_result is not None:
            st.subheader("DEMONgram")
            img_buffer = render_spectrogram(
                demon_result[0], demon_result[1], demon_result[2],
                "DEMON Spectrogram", "DEMON", gamma_val=gamma_demon, hough_lines=None
            )
            st.download_button(
                label="Download spectrogram (PNG)",
                data=img_buffer,
                file_name=f"{Path(selected_label).stem}_demon.png",
                mime="image/png",
            )

    with st.sidebar.expander("Presets & History", expanded=False):
        preset_name = st.text_input("Preset name", key="preset_name", placeholder="e.g. Narrow band focus")
        history = st.session_state["preset_history"]

        if st.button("Save preset", use_container_width=True):
            preset_label = preset_name or f"Preset {len(history) + 1}"
            preset_payload = {
                **current_params,
                "name": preset_label,
                "saved_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
            history.append(preset_payload)
            st.session_state["preset_history"] = history
            st.success(f"Saved preset '{preset_label}'")

        if history:
            preset_options = [f"{idx + 1}. {item.get('name', f'Preset {idx + 1}')}" for idx, item in enumerate(history)]
            selected_history_idx = st.selectbox("Preset history", range(len(history)),
                                                format_func=lambda i: preset_options[i], key="history_select")
            if st.button("Load selected preset", use_container_width=True):
                apply_preset(history[selected_history_idx])

            history_json = json.dumps(history, indent=2).encode("utf-8")
            st.download_button("Download preset history (JSON)", data=history_json,
                               file_name="lofar_presets_history.json", mime="application/json",
                               use_container_width=True)
        else:
            st.info("No presets saved yet.")

        current_preset_json = json.dumps({**current_params, "name": preset_name or "current"}, indent=2).encode("utf-8")
        st.download_button("Download current preset (JSON)", data=current_preset_json,
                           file_name="lofar_preset.json", mime="application/json",
                           use_container_width=True)

        uploaded_presets = st.file_uploader("Import presets JSON", type="json", key="preset_import")
        if uploaded_presets is not None:
            try:
                payload = json.load(uploaded_presets)
                if isinstance(payload, dict):
                    payload = [payload]
                if isinstance(payload, list):
                    valid_presets = [p for p in payload if isinstance(p, dict)]
                    history.extend(valid_presets)
                    st.session_state["preset_history"] = history
                    st.success(f"Imported {len(valid_presets)} preset(s).")
                else:
                    st.error("Uploaded file must be a JSON object or list of objects.")
            except Exception as exc:
                st.error(f"Failed to import presets: {exc}")


if __name__ == "__main__":
    main()

