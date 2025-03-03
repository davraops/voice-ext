import numpy as np

def smbPitchShift(pitch_shift, num_samples, fft_frame_size, osamp, sample_rate, indata):
    """
    Simplified Python version of the smbPitchShift algorithm using an FFT length
    of exactly `fft_frame_size` (e.g., 1024). No real/imag interleaving of 2 * fft_frame_size.
    """

    # Persistent buffers across calls (like static variables in C)
    static_info = smbPitchShift.__dict__.setdefault('static_info', {})
    gInFIFO  = static_info.setdefault('gInFIFO',  np.zeros(fft_frame_size, dtype=np.float32))
    gOutFIFO = static_info.setdefault('gOutFIFO', np.zeros(fft_frame_size, dtype=np.float32))
    gLastPhase = static_info.setdefault('gLastPhase', np.zeros(fft_frame_size//2+1, dtype=np.float32))
    gSumPhase  = static_info.setdefault('gSumPhase',  np.zeros(fft_frame_size//2+1, dtype=np.float32))
    gAnaFreq   = static_info.setdefault('gAnaFreq',   np.zeros(fft_frame_size//2+1, dtype=np.float32))
    gAnaMagn   = static_info.setdefault('gAnaMagn',   np.zeros(fft_frame_size//2+1, dtype=np.float32))
    gOutputAccum = static_info.setdefault('gOutputAccum', np.zeros(fft_frame_size, dtype=np.float32))

    outdata = np.zeros(num_samples, dtype=np.float32)

    step_size = fft_frame_size // osamp
    freq_per_bin = sample_rate / float(fft_frame_size)
    expct = 2.0 * np.pi * step_size / fft_frame_size

    idx = 0
    while idx < num_samples:
        # Shift in new samples
        gInFIFO[:-step_size] = gInFIFO[step_size:]
        remaining = num_samples - idx
        gInFIFO[-step_size:] = (indata[idx:idx+step_size] if remaining >= step_size
                                else np.concatenate([indata[idx:idx+remaining],
                                                     np.zeros(step_size - remaining, dtype=np.float32)]))
        idx += step_size

        # Windowing
        windowed = gInFIFO * np.hanning(fft_frame_size).astype(np.float32)

        # Compute FFT of size fft_frame_size
        spectrum = np.fft.rfft(windowed, n=fft_frame_size)
        magn = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Phase difference
        phase_diff = phase - gLastPhase[:len(phase)]
        gLastPhase[:len(phase)] = phase

        # Map delta phase into +/- pi range
        phase_diff = phase_diff - 2.0*np.pi * np.floor((phase_diff / (2.0*np.pi)) + 0.5)

        # True frequency
        freq = freq_per_bin * (np.arange(len(phase_diff)) + phase_diff / expct)

        # Scale by pitch_shift
        freq *= pitch_shift

        # Store analysis
        gAnaMagn[:len(magn)] = magn
        gAnaFreq[:len(freq)] = freq

        # Synthesis
        gSumPhase[:len(phase_diff)] += expct * (gAnaFreq[:len(phase_diff)] / freq_per_bin)
        out_phase = gSumPhase[:len(phase_diff)]
        res = gAnaMagn[:len(phase_diff)] * np.exp(1j * out_phase)

        # Inverse FFT
        inv_spectrum = np.fft.irfft(res, n=fft_frame_size)
        windowed_inv = inv_spectrum * np.hanning(fft_frame_size).astype(np.float32)

        gOutputAccum += windowed_inv

        # Output
        outdata_idx = idx - step_size
        if outdata_idx + step_size <= num_samples:
            outdata[outdata_idx:outdata_idx+step_size] = gOutputAccum[:step_size]

        # Shift accum
        gOutputAccum[:-step_size] = gOutputAccum[step_size:]
        gOutputAccum[-step_size:] = 0.0

    return outdata
