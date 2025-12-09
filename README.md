# VIBE NASA - LOFAR Spectrogram Playground

<div align="center">
  <img src="assets/Logo.png" alt="VIBE NASA Logo" width="200">
</div>

VIBE NASA is a Reaserch Tool designed for exploring LOFAR and DEMON transformations for feature extraction exploration and Optimisation.



## Features

- Built-in sample recordings under `data/` plus WAV upload support.
- Tunable DSP chain (windowing, FFT length, overlap, decimation, noise floor).
- Custom colormaps including a sonar-style green palette.
- Preset manager with save/load/import/export (JSON).
- One-click downloads for spectrogram images and preset collections.

## DEMON Feature

**DEMON (Detection of Envelope Modulation on Noise)** is an algorithm designed to estimate the envelope of amplitude-modulated noise signals. This feature is particularly useful for detecting modulation patterns in underwater acoustic signals, such as those produced by small boats and marine vessels.

The implementation provides two methods:

- **Square Law DEMON**: Applies a bandpass filter, squares the signal, decimates it, and computes the frequency-domain spectrum. This method returns a magnitude spectrum that reveals modulation frequencies in the signal.

- **Hilbert Transform DEMON**: Uses the Hilbert transform to extract the envelope, followed by decimation and normalization. This method returns the time-domain envelope signal.

Both methods support configurable parameters:
- **Bandpass filtering**: Define high and low frequency limits to focus on specific frequency bands
- **Cutoff frequency**: Controls the decimation rate for envelope extraction
- **Sampling rate**: Adapts to your input signal's sample rate

The tool also includes a **demongram** function that applies DEMON processing to overlapping windows of the signal, creating a time-frequency representation similar to a spectrogram but optimized for detecting envelope modulations.

### DEMON Parameters

- **Method** (`method`): Selects the DEMON algorithm variant:
  - `square_law`: Applies bandpass filtering, squares the signal, then decimates and computes FFT. Best for detecting modulation frequencies in the frequency domain.
  - `hilbert_detector`: Uses Hilbert transform to extract the envelope, then decimates. Best for time-domain envelope analysis.

- **Cutoff Frequency** (`cutoff`): The maximum frequency (in Hz) for the decimated envelope signal. This parameter determines the decimation rate using the formula `n = floor(sample_rate / (cutoff * 2))`. 

  The formula is derived from the Nyquist sampling theorem: after decimation by factor `n`, the new sample rate becomes `sample_rate / n`, and the maximum representable frequency (Nyquist frequency) is `(sample_rate / n) / 2 = sample_rate / (2 * n)`. To preserve frequencies up to `cutoff`, we require `sample_rate / (2 * n) >= cutoff`, which rearranges to `n <= sample_rate / (cutoff * 2)`. Taking the floor gives the largest integer decimation factor that still preserves frequencies up to `cutoff`.
  
  Lower cutoff values result in more aggressive decimation (larger `n`) and focus on slower modulations. Must be less than the Nyquist frequency (sample_rate / 2). Typical range: 100-10000 Hz.

- **High Frequency** (`high`): The upper limit (in Hz) of the bandpass filter applied before envelope extraction. This defines the top of the frequency band of interest. Must be less than the Nyquist frequency. Typical range: 1000+ Hz.

- **Low Frequency** (`low`): The lower limit (in Hz) of the bandpass filter applied before envelope extraction. This defines the bottom of the frequency band of interest. Must be less than the high frequency and less than the Nyquist frequency. Typical range: 1000+ Hz.

- **Window Size** (`window_size`): The number of samples in each analysis window. When set to 0 (auto), defaults to the sample rate (1 second of audio). Larger windows provide better frequency resolution but worse time resolution. Smaller windows provide better time resolution but worse frequency resolution. Typical range: 0 (auto) or 1000-1000000 samples.

- **Overlap Fraction** (`overlap`): The fraction of overlap between consecutive windows (0.0 to 0.95). Higher overlap provides smoother time resolution but increases computational cost. A value of 0.5 (50% overlap) is a common default. Typical range: 0.0-0.95.

*Reference:\
Theory: Pollara, A., Sutin, A., & Salloum, H. (2016). Improvement of the Detection of Envelope Modulation on Noise (DEMON) and its application to small boats. In OCEANS 2016 MTS-IEEE Monterey IEEE.\
Code based on the following repo: https://github.com/lxpollara/pyDEMON*

## LOFAR Feature

**LOFAR (Low Frequency Analysis and Recording)** is a signal processing technique designed for analyzing underwater acoustic signals and extracting narrow-band tonal features from noisy environments.

The LOFAR processing chain includes:

1. **Decimation**: Optional downsampling of the input signal to reduce computational load while preserving relevant frequency content.

2. **Short-Time Fourier Transform (STFT)**: Converts the time-domain signal into a time-frequency representation using configurable window functions (e.g., Hann window), window size (`nperseg`), and overlap parameters.

3. **TPSW Normalization**: Applies Two-Pass Split Window (TPSW) normalization to remove slow-varying background energy. This critical step enhances the visibility of narrow-band tonal lines by suppressing broadband noise and emphasizing periodic components.

4. **Noise Floor Thresholding**: Optional logarithmic scaling and floor thresholding to further enhance contrast and suppress low-level noise.

The resulting LOFARgram (LOFAR spectrogram) is optimized for detecting and visualizing:
- Narrow-band tonal signals (e.g., propeller harmonics, machinery tones)
- Periodic modulations in underwater acoustic data
- Features that may be obscured in standard spectrograms

Key tunable parameters include decimation factor, window type and size, overlap percentage, and noise floor thresholds, allowing you to optimize the analysis for different signal characteristics and detection requirements.

### LOFAR Parameters

- **Window Function** (`window`): The window function applied to each segment before FFT computation. Different windows have different spectral leakage characteristics:
  - `hann`: Good frequency resolution with moderate sidelobe suppression. Good general-purpose choice.
  - `hamming`: Similar to Hann but with slightly better sidelobe suppression at the cost of slightly wider main lobe.
  - `blackman`: Better sidelobe suppression but wider main lobe, reducing frequency resolution.
  - `bartlett`: Triangular window, simple but less optimal than other options.

- **FFT Window Size** (`nperseg`): The number of samples in each FFT window. This is a trade-off between time and frequency resolution:
  - Larger values (e.g., 4096): Better frequency resolution but worse time resolution. Good for detecting stable tonal signals.
  - Smaller values (e.g., 256): Better time resolution but worse frequency resolution. Good for detecting transient or rapidly changing signals.
  - Typical choices: 256, 512, 1024, 2048, 4096 samples.

- **Overlap** (`noverlap`): The number of overlapping samples between consecutive windows. Must be less than `nperseg`. Higher overlap provides smoother time resolution and better detection of transient features, but increases computational cost. Typical values range from 50% to 90% of `nperseg` (e.g., for `nperseg=2048`, overlap might be 1024-1905 samples).

- **Decimation Factor** (`decimation_factor`): The downsampling factor applied to the input signal before STFT processing. Reduces computational load and focuses analysis on lower frequencies. A factor of 1 means no decimation. Higher factors (e.g., 10) downsample more aggressively. Typical range: 1-12. Note: Decimation reduces the effective sample rate, which lowers the maximum analyzable frequency.

- **Floor Threshold** (`floor_threshold`): The minimum log10 power value (in log10 units) before clamping. Values below this threshold are replaced with `floor_value`. This helps suppress low-level noise and enhance contrast. Typical range: -3.0 to 0.0. More negative values preserve more detail; less negative values suppress more noise.

- **Floor Replacement Value** (`floor_value`): The value (in log10 units) used to replace samples below the `floor_threshold`. This sets the visual floor level in the spectrogram. Typical range: -3.0 to 3.0. More negative values create a darker floor; less negative or positive values create a brighter floor.

- **Stability Epsilon** (`eps`): A small constant added to prevent log(0) errors when computing logarithmic power. Very small values (e.g., 1e-12) are typically sufficient. Typical range: 1e-15 to 1e-3. This parameter rarely needs adjustment unless you encounter numerical stability issues.

### TPSW Normalization Parameters

The TPSW (Two-Pass Split Window) normalization uses internal parameters that are currently set to default values but can be adjusted in the code if needed:

- **Half-Window Size** (`n`): The half-width of the averaging window used to estimate background energy. Default: `int(round(npts * 0.04 / 2.0 + 1))`, where `npts` is the number of frequency bins. This creates a window that spans approximately 4% of the frequency range. 
  - **Choosing values**: Larger `n` values provide smoother background estimates but may blur narrow tonal lines. Smaller `n` values preserve more detail but may be more sensitive to noise. Typical range: 1-50% of the number of frequency bins. The default (2% of frequency bins) is a good starting point for most signals.

- **Guard Band Size** (`p`): The number of frequency bins around a potential tonal line that are excluded from the averaging window. This protects narrow tonal signals from being averaged into the background estimate. Default: `int(round(n / 8.0 + 1))`, which is approximately 1/8 of the half-window size.
  - **Choosing values**: Larger `p` values provide better protection for wider tonal lines but reduce the effective averaging window size. Smaller `p` values allow more averaging but may suppress narrow tones. Typical range: 1 to `n/2`. The default (n/8) works well for most narrow-band signals.

- **Threshold Multiplier** (`a`): The multiplier used to determine which frequency bins exceed the background estimate and should be clipped in the first pass. Bins where `signal > a * background` are clipped to the background level before the second pass. Default: `2.0`.
  - **Choosing values**: Larger `a` values (e.g., 2.5-3.0) are more conservative and only clip very strong peaks, preserving more signal detail. Smaller `a` values (e.g., 1.5-2.0) are more aggressive and clip weaker peaks, providing stronger noise suppression. Typical range: 1.5-3.0. The default of 2.0 provides a good balance for most applications.

**Note**: Currently, these TPSW parameters are not exposed in the UI and use their default values. To adjust them, you would need to modify the `tpsw()` function call in `src/lofar.py` line 101 to pass custom values: `tpsw(Sxx_magnitude, n=n_value, p=p_value, a=a_value)`.

## Requirements

- Python 3.10+ (tested on macOS 15 / Apple Silicon).
- Required packages (see `requirements.txt`):
  - streamlit
  - numpy
  - scipy
  - librosa
  - soundfile
  - matplotlib

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

```shell
streamlit run src/app.py
```

The app looks for bundled WAV files in `data/` and an optional logo at `assets/Logo.png`.

## Usage Tips

- Use the sidebar to select a built-in sample or upload your own mono/stereo WAV.
- Adjust the LOFAR parameters and watch the spectrogram update instantly.
- Click **Save preset** to capture current settings; download/export for later reuse.
- The **Download spectrogram** button saves a PNG with the current orientation and color scale.

## Project Layout

```
assets/          Branding/logo resources
data/            Sample WAV files for demo use
src/app.py       Streamlit UI and orchestration layer
src/transforms.py DSP helpers (TPSW normalization, LOFAR computation)
```

## Troubleshooting

- **No audio found**: add `.wav` files to `data/` or use the upload option.
- **Empty spectrogram**: reduce overlap or pick a longer clip; extremely short or silent signals can collapse the STFT.
- **Performance issues**: lower `nperseg`, decimation factor, or disable rotation to reduce rendering load.


# vibe-nasa
