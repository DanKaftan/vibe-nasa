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
from pathlib import Path
from typing import Dict, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from matplotlib.colors import ListedColormap

from transforms import lofar

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
            (0.0, 0.05, 0.01),    # Very dark green
            (0.0, 0.15, 0.03),    # Dark green
            (0.0, 0.3, 0.07),     # Medium-dark green
            (0.0, 0.5, 0.13),     # Medium green
            (0.1, 0.75, 0.2),     # Bright green
            (0.3, 0.95, 0.35),    # Very bright green (highest values)
        ],
        name="sonar_green",
    )
}
CMAP_CHOICES = BASE_CMAP_CHOICES + list(CUSTOM_CMAPS.keys())

# Default parameter preset for LOFAR analysis
# These values provide a good starting point for most audio analysis tasks
DEFAULT_PRESET = {
    "window": WINDOW_CHOICES[0],      # Hann window (good balance)
    "nperseg": 1024,                  # FFT window size (1024 samples)
    "noverlap": 512,                  # 50% overlap between windows
    "decimation_factor": 3,           # Downsample by factor of 3
    "floor_threshold": -0.2,          # Clamp values below -0.2 log10 power
    "floor_value": 0.0,               # Replace clamped values with 0.0
    "eps": 1e-12,                     # Small epsilon to prevent log(0)
    "rotate_90": False,               # Standard orientation (time on X-axis)
    "cmap": CMAP_CHOICES[0],          # Default colormap (viridis)
    "contrast_db": 40.0,              # 40 dB dynamic range for display
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
                     floor_threshold: float, floor_value: float, eps: float):
    """
    Wrapper function to compute LOFAR spectrogram with user-selected parameters.
    
    This function acts as a bridge between the Streamlit UI and the core
    LOFAR processing function in transforms.py. All parameters are passed
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
        
    Returns:
        Tuple of (time_axis, frequency_axis, spectrogram_matrix)
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
    )


def init_session_state():
    """
    Initialize Streamlit session state with default values.
    
    Session state persists across reruns, allowing the UI to remember
    user selections. This function sets default values only if they
    don't already exist (using setdefault to avoid overwriting).
    """
    defaults = {
        # LOFAR processing parameters
        "window_select": DEFAULT_PRESET["window"],
        "nperseg_select": DEFAULT_PRESET["nperseg"],
        "noverlap_value": DEFAULT_PRESET["noverlap"],
        "decimation_factor": DEFAULT_PRESET["decimation_factor"],
        "floor_threshold": DEFAULT_PRESET["floor_threshold"],
        "floor_value": DEFAULT_PRESET["floor_value"],
        "eps_value": DEFAULT_PRESET["eps"],
        # Display parameters
        "rotate_flag": DEFAULT_PRESET["rotate_90"],
        "cmap_choice": DEFAULT_PRESET["cmap"],
        "contrast_db": DEFAULT_PRESET["contrast_db"],
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
    
    # Apply display parameters
    st.session_state["rotate_flag"] = bool(preset.get("rotate_90", DEFAULT_PRESET["rotate_90"]))
    cmap_value = preset.get("cmap", DEFAULT_PRESET["cmap"])
    # Validate colormap choice
    st.session_state["cmap_choice"] = cmap_value if cmap_value in CMAP_CHOICES else DEFAULT_PRESET["cmap"]
    st.session_state["contrast_db"] = float(preset.get("contrast_db", DEFAULT_PRESET["contrast_db"]))
    
    # Preserve preset name if provided
    st.session_state["preset_name"] = preset.get("name", st.session_state.get("preset_name", ""))
    
    # Trigger page rerun to update UI with new values
    st.experimental_rerun()


def main():
    st.set_page_config(page_title="LOFAR Playground", layout="wide")
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=200)
    st.title("VIBE NASA - LOFAR Spectrogram Playground")
    st.caption("Interactively explore how LOFAR parameters affect the spectrogram.")
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
    # Parameter sidebar: everything below manipulates LOFAR computation.
    # Streamlit keys map directly into session_state so presets can reload.
    # ------------------------------------------------------------------
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
    contrast_db = st.sidebar.slider("Color dynamic range (dB)", min_value=5.0, max_value=80.0,
                                    value=st.session_state["contrast_db"], step=1.0, key="contrast_db")

    current_params = {
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "decimation_factor": decimation_factor,
        "floor_threshold": floor_threshold,
        "floor_value": floor_value,
        "eps": eps,
        "rotate_90": rotate_90,
        "cmap": cmap_choice,
        "contrast_db": contrast_db,
    }

    if waveform is None or sample_rate is None:
        st.info("Select or upload an audio file to begin.")
        return

    duration = len(waveform) / sample_rate
    st.subheader("Audio preview")
    st.write(f"**Selection:** {selected_label} | **Sample rate:** {sample_rate:,} Hz | "
             f"**Duration:** {duration:.2f} s")
    st.audio(waveform_to_wav_bytes(waveform, sample_rate))

    hop_length = nperseg - noverlap
    st.markdown(
        f"- **Window:** `{window}` | **nperseg:** {nperseg} samples\n"
        f"- **Overlap:** {noverlap} samples | **Hop length:** {hop_length} samples\n"
        f"- **Decimation factor:** {decimation_factor}"
    )

    with st.spinner("Computing LOFAR spectrogram..."):
        t, f, lofar_matrix = render_lofargram(
            waveform,
            sample_rate,
            decimation_factor=decimation_factor,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            floor_threshold=floor_threshold,
            floor_value=floor_value,
            eps=eps,
        )

    if t.size == 0 or f.size == 0 or lofar_matrix.size == 0:
        st.error("Parameters produced an empty LOFAR spectrogram. Try reducing the overlap or using a longer audio clip.")
        return

    st.subheader("LOFARgram")
    fig, ax = plt.subplots(figsize=(12, 4))
    time_extent = (t[0], t[-1] if len(t) > 1 else t[0] + 1e-6)
    freq_extent = (f[0], f[-1] if len(f) > 1 else f[0] + 1e-6)
    matrix_to_show = lofar_matrix.T if rotate_90 else lofar_matrix
    x_extent = freq_extent if rotate_90 else time_extent
    y_extent = time_extent if rotate_90 else freq_extent
    vmax = float(np.max(lofar_matrix))
    vmin = vmax - float(contrast_db)
    im = ax.imshow(
        matrix_to_show,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Frequency (Hz)" if rotate_90 else "Time (s)")
    ax.set_ylabel("Time (s)" if rotate_90 else "Frequency (Hz)")
    ax.set_title("LOFAR Spectrogram" + (" (rotated)" if rotate_90 else ""))
    plt.colorbar(im, ax=ax, label="Relative power (log10)")
    st.pyplot(fig)
    plt.close(fig)

    img_buffer = io.BytesIO()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(
        matrix_to_show,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_axis_off()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    img_buffer.seek(0)

    st.download_button(
        label="Download spectrogram (PNG)",
        data=img_buffer,
        file_name=f"{Path(selected_label).stem}_lofar.png",
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

