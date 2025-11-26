import numpy as np
import scipy.signal as signal

def tpsw(x, npts=None, n=None, p=None, a=None):
    """
    Two-Pass Split Window (TPSW) normalization as described in classic LOFAR literature.
    The routine removes slow-varying background energy so that narrow-band
    tonal lines stand out in the spectrogram.

    Args:
        x: Spectrogram magnitude matrix (freq x time or vice versa). Will be coerced to 2D.
        npts: Number of time samples to process. Defaults to x.shape[0].
        n: Half-window size for the averaging window.
        p: Guard band size that protects the tone being measured.
        a: Threshold multiplier that controls peak clipping strength.

    Returns:
        np.ndarray: Background estimate that can be used to normalize the spectrogram.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if npts is None:
        npts = x.shape[0]
    if n is None:
        n=int(round(npts*.04/2.0+1))
    if p is None:
        p =int(round(n / 8.0 + 1))
    if a is None:
        a = 2.0
    # Build the split-window kernel (unity except for a zeroed guard band)
    if p>0:
        h = np.concatenate((np.ones((n-p+1)), np.zeros(2 * p-1), np.ones((n-p+1))), axis=None)
    else:
        h = np.ones((1, 2*n+1))
        p = 1
    h /= np.linalg.norm(h, 1)

    def apply_on_spectre(xs):
        return signal.convolve(h, xs, mode='full')

    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    ix = int(np.floor((h.shape[0] + 1)/2.0)) # filter delay
    mx = mx[ix-1:npts+ix-1] # shift to compensate delay
    # Edge-correction factors to avoid under-estimating energy at the borders
    ixp = ix - p
    mult=2*ixp/np.concatenate([np.ones(p-1)*ixp, range(ixp,2*ixp + 1)], axis=0)[:, np.newaxis] # Correcao dos pontos extremos
    mx[:ix,:] = mx[:ix,:]*(np.matmul(mult, np.ones((1, x.shape[1])))) # Pontos iniciais
    mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*np.matmul(np.flipud(mult),np.ones((1, x.shape[1]))) # Pontos finais
    #return mx
    # Identify bins that exceed the threshold; those will be clipped to the background
    #indl= np.where((x-a*mx) > 0) # Pontos maiores que a*mx
    indl = (x-a*mx) > 0
    #x[indl] = mx[indl]
    x = np.where(indl, mx, x)
    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    mx=mx[ix-1:npts+ix-1,:]
    #Corrige pontos extremos do espectro
    mx[:ix,:]=mx[:ix,:]*(np.matmul(mult,np.ones((1, x.shape[1])))) # Pontos iniciais
    mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*(np.matmul(np.flipud(mult),np.ones((1,x.shape[1])))) # Pontos finais
    return mx


def lofar(audio, fs, decimation_factor=3, window='hann', nperseg=1024, noverlap=0,
          floor_threshold=-0.2, floor_value=0.0, eps=1e-12):
    """
    Perform LOFAR analysis on the input audio.

    Parameters:
        audio (numpy array): Input audio signal.
        fs (int): Sampling rate of the audio signal.
        decimation_factor (int): Downsampling factor applied before STFT.
        window (str): Window function passed to `scipy.signal.spectrogram`.
        nperseg (int): STFT window size.
        noverlap (int): Number of samples to overlap between segments.
        floor_threshold (float or None): If set, values below this log10 power
            will be clamped to `floor_value`.
        floor_value (float or None): Replacement value for entries below
            `floor_threshold`. If None, values are clamped to the threshold.
        eps (float): Small constant added before logarithm to avoid log(0).

    Returns:
        time (numpy array): Time axis for the spectrogram.
        freq (numpy array): Frequency axis for the spectrogram.
        spectrogram (numpy array): Computed spectrogram.
    """
    # Step 1: Decimation (low-pass filter and resample)
    if decimation_factor > 1:
        audio_decimated = signal.decimate(audio, decimation_factor, 10, 'fir', zero_phase=True)
        fs_decimated = fs // decimation_factor
    else:
        audio_decimated = audio
        fs_decimated = fs

    # Step 2: Apply Short-Time Fourier Transform (STFT)
    f, t, Sxx = signal.spectrogram(audio_decimated, fs=fs_decimated, window=window, 
                                   nperseg=nperseg, noverlap=noverlap, 
                                   detrend=False, scaling='spectrum', mode='magnitude')

    # Step 3: Normalize using TPSW (Two-Pass Split Window)
    Sxx_magnitude = np.abs(Sxx)
    Sxx_magnitude = Sxx_magnitude / tpsw(Sxx_magnitude)
    Sxx_magnitude = np.log10(np.maximum(Sxx_magnitude, eps))

    if floor_threshold is not None:
        mask = Sxx_magnitude < floor_threshold
        replacement = floor_threshold if floor_value is None else floor_value
        Sxx_magnitude[mask] = replacement

    return t, f, Sxx_magnitude


def lofar_shape(audio_len, fs, decimation_factor=3, window='hann', nperseg=1024, noverlap=0):
    """
    Utility helper that mirrors `lofar` sizing logic without performing the STFT.
    Lets the UI pre-compute how large the spectrogram matrix will be for layout.

    Args:
        audio_len: Number of audio samples.
        fs: Sampling rate before decimation.
        decimation_factor: Same definition as in `lofar`.
        window: Kept for API parity; unused.
        nperseg: FFT window length.
        noverlap: Number of overlapping samples between windows.

    Returns:
        tuple[int, int]: (frequency_bins, time_bins)
    """
    if decimation_factor > 1:
        audio_len = int(np.ceil(audio_len / decimation_factor))
        fs = int(np.ceil(fs / decimation_factor))

    if noverlap is None:
        noverlap = nperseg // 8
    nfft = nperseg

    n_freqs = nfft // 2 + 1
    step = nperseg - noverlap

    if audio_len < nperseg:
        n_segments = 0
    else:
        n_segments = 1 + (audio_len - nperseg) // step

    return (n_freqs, n_segments)


