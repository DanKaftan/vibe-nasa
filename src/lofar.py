import numpy as np
import scipy.signal as signal
import cv2

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


def hough_line_enhancement(spectrogram, threshold_percentile=75, min_line_length=None, 
                           max_line_gap=5, line_theta_range=None, enhancement_factor=1.5,
                           use_morphology=False, morph_kernel_size=3, morph_iterations=1,
                           morph_operation='closing'):
    """
    Detect vertical lines in a spectrogram using Hough transform.
    
    This function detects vertical lines in the spectrogram and returns the original
    spectrogram along with line coordinates for visualization.
    
    Args:
        spectrogram (np.ndarray): Input spectrogram matrix (freq x time).
        threshold_percentile (float): Percentile threshold for edge detection (0-100).
            Higher values detect only stronger lines. Default: 75.
        min_line_length (int or None): Minimum line length in pixels. If None, uses
            5% of the frequency dimension (for vertical lines). Default: None.
        max_line_gap (int): Maximum gap between line segments to be connected. Default: 5.
        line_theta_range (tuple or None): Range of angles (in degrees) to detect lines.
            For vertical lines, use (85, 95) or similar. Default: None (all angles).
        enhancement_factor (float): Not used, kept for API compatibility. Default: 1.5.
        use_morphology (bool): Enable morphological operations to filter and merge lines. Default: False.
        morph_kernel_size (int): Size of the morphological kernel (must be odd). Default: 3.
        morph_iterations (int): Number of iterations for morphological operations. Default: 1.
        morph_operation (str): Type of morphological operation: 'opening', 'closing', 'both', 'erosion', 'dilation'. Default: 'closing'.
    
    Returns:
        tuple: (spectrogram, lines) where:
            - spectrogram: Original spectrogram unchanged
            - lines: List of line coordinates [(x1, y1, x2, y2), ...] or None
    """
    if spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be 2D (freq x time)")
    
    # Normalize spectrogram to 0-255 range for image processing
    spectrogram_min = np.min(spectrogram)
    spectrogram_max = np.max(spectrogram)
    if spectrogram_max == spectrogram_min:
        return (spectrogram, None)  # No variation, return as-is
    
    spectrogram_norm = ((spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min) * 255).astype(np.uint8)
    
    # Set default min_line_length if not provided (for vertical lines, use frequency dimension)
    if min_line_length is None:
        min_line_length = max(10, int(spectrogram.shape[0] * 0.05))  # 5% of frequency dimension
    
    # Apply thresholding to create binary image for line detection
    threshold_value = np.percentile(spectrogram_norm, threshold_percentile)
    _, binary_image = cv2.threshold(spectrogram_norm, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to filter and merge lines
    if use_morphology:
        # Ensure kernel size is odd
        kernel_size = morph_kernel_size if morph_kernel_size % 2 == 1 else morph_kernel_size + 1
        # Create vertical kernel for vertical lines (better for merging vertical lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
        
        if morph_operation == 'opening':
            # Erosion followed by dilation - removes small lines
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
        elif morph_operation == 'closing':
            # Dilation followed by erosion - merges nearby lines and fills gaps
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        elif morph_operation == 'both':
            # Opening then closing - removes small lines and merges nearby ones
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        elif morph_operation == 'erosion':
            # Erosion only - shrinks lines, removes small ones
            binary_image = cv2.erode(binary_image, kernel, iterations=morph_iterations)
        elif morph_operation == 'dilation':
            # Dilation only - expands lines, merges nearby ones
            binary_image = cv2.dilate(binary_image, kernel, iterations=morph_iterations)
    
    # Detect lines using probabilistic Hough transform (HoughLinesP)
    # For vertical lines, we want angles around 90 degrees
    lines = cv2.HoughLinesP(
        binary_image,
        rho=1,              # Distance resolution in pixels
        theta=np.pi/180,    # Angular resolution in radians (1 degree)
        threshold=50,       # Minimum votes for a line
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    detected_lines = None
    if lines is not None:
        detected_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle of the line
            if x2 != x1:
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.rad2deg(angle_rad)
                # Normalize to 0-180 range
                if angle_deg < 0:
                    angle_deg += 180
            else:
                angle_deg = 90  # Vertical line
            
            # Filter by angle range if specified
            if line_theta_range is not None:
                min_angle, max_angle = line_theta_range
                # Check if line angle is within the specified range
                # Handle both normal range and wrap-around case
                if min_angle <= max_angle:
                    is_in_range = (min_angle <= angle_deg <= max_angle)
                else:
                    # Handle wrap-around case (e.g., 175-5 degrees)
                    is_in_range = (angle_deg >= min_angle or angle_deg <= max_angle)
                
                if not is_in_range:
                    continue
            
            detected_lines.append((x1, y1, x2, y2))
    
    # Return original spectrogram and line coordinates
    return (spectrogram, detected_lines)


def lofar(audio, fs, decimation_factor=3, window='hann', nperseg=1024, noverlap=0,
          floor_threshold=-0.2, floor_value=0.0, eps=1e-12, tpsw_n=None, tpsw_p=None, tpsw_a=None,
          hough_enhance=False, hough_threshold_percentile=75, hough_min_line_length=None,
          hough_max_line_gap=5, hough_theta_range=None, hough_enhancement_factor=1.5,
          hough_use_morphology=False, hough_morph_kernel_size=3, hough_morph_iterations=1,
          hough_morph_operation='closing'):
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
        tpsw_n (int or None): Half-window size for TPSW normalization. If None, uses adaptive default.
        tpsw_p (int or None): Guard band size for TPSW normalization. If None, uses adaptive default.
        tpsw_a (float or None): Threshold multiplier for TPSW normalization. If None, uses default (2.0).
        hough_enhance (bool): If True, apply Hough transform line enhancement. Default: False.
        hough_threshold_percentile (float): Percentile threshold for Hough line detection (0-100). Default: 75.
        hough_min_line_length (int or None): Minimum line length for Hough detection. If None, uses adaptive default.
        hough_max_line_gap (int): Maximum gap between line segments for Hough detection. Default: 5.
        hough_theta_range (tuple or None): Angle range in degrees for line detection. None = all angles. Default: None.
        hough_enhancement_factor (float): Enhancement factor for detected lines. Default: 1.5.
        hough_use_morphology (bool): Enable morphological operations to filter and merge lines. Default: False.
        hough_morph_kernel_size (int): Size of morphological kernel (must be odd). Default: 3.
        hough_morph_iterations (int): Number of iterations for morphological operations. Default: 1.
        hough_morph_operation (str): Morphological operation type: 'opening', 'closing', 'both', 'erosion', 'dilation'. Default: 'closing'.

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
    Sxx_magnitude = Sxx_magnitude / tpsw(Sxx_magnitude, n=tpsw_n, p=tpsw_p, a=tpsw_a)
    Sxx_magnitude = np.log10(np.maximum(Sxx_magnitude, eps))

    if floor_threshold is not None:
        mask = Sxx_magnitude < floor_threshold
        replacement = floor_threshold if floor_value is None else floor_value
        Sxx_magnitude[mask] = replacement

    # Step 4: Optional Hough transform line detection
    hough_lines = None
    if hough_enhance:
        Sxx_magnitude, hough_lines = hough_line_enhancement(
            Sxx_magnitude,
            threshold_percentile=hough_threshold_percentile,
            min_line_length=hough_min_line_length,
            max_line_gap=hough_max_line_gap,
            line_theta_range=hough_theta_range,
            enhancement_factor=hough_enhancement_factor,
            use_morphology=hough_use_morphology,
            morph_kernel_size=hough_morph_kernel_size,
            morph_iterations=hough_morph_iterations,
            morph_operation=hough_morph_operation
        )

    return t, f, Sxx_magnitude, hough_lines


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

