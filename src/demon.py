from scipy.signal import butter, lfilter, decimate, hilbert
from numpy import square, sqrt, mean, abs, array, zeros
from math import floor


"""
Python implementation of the DEMON algorithm for estimating the
envelope of an amplitude modulated noise. Two versions are given,
the square law and hilbert transform DEMON algorithms. Depending 
on the use case either algorithm may be more desirable. Both expect
an input array of the original data, and filter parameters. Both
return an output array containing the estimated envelope of the 
input data.

author: Alex Pollara

Algorithm is described in:
Pollara, A., Sutin, A., & Salloum, H. (2016). 
Improvement of the Detection of Envelope Modulation on Noise (DEMON) 
and its application to small boats. In OCEANS 2016 MTS-IEEE Monterey
IEEE.
    
"""


def square_law(x, cutoff=1000.0, high=30000, low=20000, fs=200000):
    """
    :param x: numpy.ndarray
    :param cutoff: float
    :param high: float
    :param low: float
    :param fs: float
    :return: numpy.ndarray
    """

    # Bandpass filter parameters
    nyq = .5 * fs  # band limit of signal Hz
    
    # Validate frequencies are within Nyquist limit
    if high >= nyq:
        raise ValueError(f"High frequency ({high} Hz) must be less than Nyquist frequency ({nyq} Hz) for sample rate {fs} Hz")
    if low >= nyq:
        raise ValueError(f"Low frequency ({low} Hz) must be less than Nyquist frequency ({nyq} Hz) for sample rate {fs} Hz")
    if low >= high:
        raise ValueError(f"Low frequency ({low} Hz) must be less than high frequency ({high} Hz)")
    
    # check that parameters meet bandwidth requirements
    if (high+low)/2 <= 2*(high-low):
        raise Exception("Error, band width exceeds pass band center frequency")

    # Passband limits as a fraction of signal band limit
    high_norm = high / nyq
    low_norm = low / nyq
    order = 3

    # Butterworth bandpass filter coefficients
    # noinspection PyTupleAssignmentBalance
    b, a = butter(order, [low_norm, high_norm], btype='band')

    # filter signal
    x = lfilter(b, a, x)

    # square signal
    x = square(x)

    # calculate decimation rate
    n = int(floor(fs / (cutoff * 2)))
    # decimate signal by n using a low pass filter
    x = decimate(x, n, ftype='fir')

    # square root of signal
    x = sqrt(x)

    # subtract mean
    x = x - mean(x)

    return x


def hilbert_detector(x, cutoff=1000.0, high=30000, low=20000, fs=200000):
    """
    :param x: numpy.ndarray
    :param cutoff: float
    :param high: float
    :param low: float
    :param fs: float
    :return: numpy.ndarray
    """
    # Bandpass filter parameters
    nyq = .5 * fs  # band limit of signal Hz
    
    # Validate frequencies are within Nyquist limit
    if high >= nyq:
        raise ValueError(f"High frequency ({high} Hz) must be less than Nyquist frequency ({nyq} Hz) for sample rate {fs} Hz")
    if low >= nyq:
        raise ValueError(f"Low frequency ({low} Hz) must be less than Nyquist frequency ({nyq} Hz) for sample rate {fs} Hz")
    if low >= high:
        raise ValueError(f"Low frequency ({low} Hz) must be less than high frequency ({high} Hz)")

    # Passband limits as a fraction of signal band limit
    high_norm = high / nyq
    low_norm = low / nyq
    order = 3

    # Butterworth bandpass filter coefficients
    # noinspection PyTupleAssignmentBalance
    b, a = butter(order, [low_norm, high_norm], btype='band')

    # filter signal
    x = lfilter(b, a, x)

    # hilbert transform of signal
    x = hilbert(x)

    # absolute value of signal
    x = abs(x)

    # calculate decimation rate
    n = int(floor(fs / (cutoff * 2)))

    # decimate signal by n using a low pass filter
    x = decimate(x, n, ftype='fir')

    # square root of signal
    x = sqrt(x)

    # subtract mean
    x = x - mean(x)

    return x


def demongram(x, window_size=None, hop_size=None, overlap=None, 
              method='square_law', cutoff=1000.0, high=30000, 
              low=20000, fs=200000):
    """
    DEMONgram: Apply DEMON algorithm to overlapping windows of the signal,
    similar to STFT but using DEMON processing.
    
    :param x: numpy.ndarray - Input signal
    :param window_size: int - Window size in samples. If None, uses fs (1 second)
    :param hop_size: int - Hop size in samples. If None, calculated from overlap
    :param overlap: float - Overlap fraction (0-1). Used if hop_size is None. Default 0.5
    :param method: str - DEMON method to use: 'square_law' or 'hilbert_detector'
    :param cutoff: float - DEMON cutoff frequency
    :param high: float - DEMON high frequency
    :param low: float - DEMON low frequency
    :param fs: float - Sampling frequency
    :return: numpy.ndarray - 2D array where rows are time windows and columns are DEMON output
    """
    x = array(x)
    signal_length = len(x)
    
    # Set default window size (1 second)
    if window_size is None:
        window_size = int(fs)
    
    # Calculate hop size from overlap if not provided
    if hop_size is None:
        if overlap is None:
            overlap = 0.5
        hop_size = int(window_size * (1 - overlap))
    
    # Ensure hop_size is at least 1
    hop_size = max(1, hop_size)
    
    # Select DEMON method
    if method == 'square_law':
        demon_func = square_law
    elif method == 'hilbert_detector':
        demon_func = hilbert_detector
    else:
        raise ValueError(f"Unknown method: {method}. Use 'square_law' or 'hilbert_detector'")
    
    # Calculate number of windows
    num_windows = max(1, (signal_length - window_size) // hop_size + 1)
    
    # Process first window to determine output size
    first_window = x[0:window_size]
    first_output = demon_func(first_window, cutoff=cutoff, high=high, 
                              low=low, fs=fs)
    output_length = len(first_output)
    
    # Initialize output array
    demongram_output = zeros((num_windows, output_length))
    demongram_output[0] = first_output
    
    # Process remaining windows
    for i in range(1, num_windows):
        start_idx = i * hop_size
        end_idx = start_idx + window_size
        
        if end_idx > signal_length:
            # Pad last window with zeros if needed
            window = zeros(window_size)
            window[0:signal_length - start_idx] = x[start_idx:signal_length]
        else:
            window = x[start_idx:end_idx]
        
        demongram_output[i] = demon_func(window, cutoff=cutoff, high=high, 
                                         low=low, fs=fs)
    
    return demongram_output
