import numpy as np
import soundfile as sf
from scipy import signal
from scripts.wav2lip.hparams import hparams as hp


def load_wav(path, sr):
    audio, _ = sf.read(path, dtype='float32')
    return audio


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    sf.write(path, wav, sr, subtype='PCM_16')


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def _lws_processor():
    # Using lws library here as there's no direct alternative in scipy or numpy
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")


def _stft(y):
    if hp.use_lws:
        return _lws_processor().stft(y).T
    else:
        # Using scipy for STFT
        return signal.stft(y, fs=hp.sample_rate, nperseg=hp.win_size, noverlap=hp.n_fft - get_hop_size())[2].T


# Compute the number of time frames of spectrogram
def num_frames(length, fsize, fshift):
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


# Compute left and right padding
def pad_lr(x, fsize, fshift):
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


# Alternative to librosa's padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
_mel_basis = None


# Convert linear spectrogram to mel spectrogram
def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


# Build the mel basis matrix
def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return create_mel_filter_bank(hp.sample_rate, hp.n_fft, hp.num_mels, hp.fmin, hp.fmax)


# Convert amplitude to decibel
def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


# Convert decibel to amplitude
def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


# Normalize the spectrogram
def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)

    assert S.max() <= 0 <= S.min() - hp.min_level_db
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))


# Denormalize the spectrogram
def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value, hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (
                        2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return (np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db

    if hp.symmetric_mels:
        return ((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db
    else:
        return (D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db


def create_mel_filter_bank(sr, n_fft, n_mels, fmin, fmax):
    # Compute Mel frequencies
    def mel(f):
        return 2595 * np.log10(1 + f / 700)

    def inv_mel(m):
        return 700 * (10 ** (m / 2595) - 1)

    # Equally spaced in Mel scale
    mel_points = np.linspace(mel(fmin), mel(fmax), n_mels + 2)
    # Convert back to frequency
    freq_points = inv_mel(mel_points)

    # FFT bin frequencies
    fft_bins = np.linspace(0, sr / 2, n_fft // 2 + 1)

    # Initialize filter bank
    mel_filter_bank = np.zeros((n_mels, len(fft_bins)))

    # Create triangular filters
    for i in range(1, n_mels + 1):
        left, center, right = freq_points[i - 1], freq_points[i], freq_points[i + 1]
        # Slopes for the triangular filters
        left_slope = (fft_bins >= left) & (fft_bins <= center)
        right_slope = (fft_bins > center) & (fft_bins <= right)
        # Assign triangular filter
        mel_filter_bank[i - 1, left_slope] = (fft_bins[left_slope] - left) / (center - left)
        mel_filter_bank[i - 1, right_slope] = (right - fft_bins[right_slope]) / (right - center)

    return mel_filter_bank
