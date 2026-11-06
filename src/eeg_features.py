"""Feature extraction, analytic sample and Layer 1 statistics.

Shared by 02_features_group_comparison.ipynb and 04_stability_sensitivity.ipynb.
Inputs are the outputs of 01_preprocessing.ipynb: for each condition, a folder
classical_<COND>/ with <ID>_windows.npy (n_windows, 512, 19, volts),
<ID>_window_indices.npy and <ID>_group.npy.
"""
import hashlib
import json
import os
from collections import Counter
from math import factorial

import numpy as np
import pandas as pd
import pywt
from scipy import signal, stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

CONDITIONS = ['EC', 'EO', 'TASK']
SFREQ = 256
UV_PER_V = 1e6            # windows are stored in volts; features are computed in µV

# --- Analytic sample and equalization -----------------------------------------
MIN_WINDOWS = 89          # minimum valid 2-s windows in every condition
BLOCK_WINDOWS = 5         # 5 consecutive 2-s windows = one 10-s block
MIN_BLOCKS = 3            # minimum 10-s blocks in every condition
N_TARGET_PERCENTILE = 10  # cap on 2-s windows = this percentile of the available windows
SEED = 42                 # base seed for window and block selection

# --- Layer 1 statistics -------------------------------------------------------
ALPHA_FDR = 0.05
N_BOOT = 2000             # bootstrap resamples for the CI of Cliff's delta

# --- Frequency bands ----------------------------------------------------------
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
ALPHA_SUB = {'low_alpha': (8, 10), 'high_alpha': (10, 13)}

# --- Feature parameters -------------------------------------------------------
WELCH_NPERSEG = 256       # 1-s segments within each 2-s window (1-Hz resolution)
SAMPEN_M = 2
SAMPEN_R_FAC = 0.20
PERMEN_ORDER = 3
PERMEN_DELAY = 1
HIGUCHI_KMAX = 10
CWT_WAVELET = 'cmor1.5-1.0'
CWT_N_FREQS_PER_BAND = 5
CWT_EDGE_CYCLES = 1.0     # trim one period of the lowest band frequency at each edge
COHERENCE_NPERSEG = 512

# --- Channels (order saved by 01_preprocessing.ipynb in channel_order.json) ---
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
CH_IDX = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
N_CHANNELS = len(CHANNEL_NAMES)
REGIONS = {
    'frontal':   ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'central':   ['C3', 'Cz', 'C4'],
    'temporal':  ['T3', 'T4', 'T5', 'T6'],
    'parietal':  ['P3', 'Pz', 'P4'],
    'occipital': ['O1', 'O2'],
}
ASYMMETRY_PAIRS = {          # (right, left)
    'frontal_F4_F3':   ('F4', 'F3'),
    'frontal_F8_F7':   ('F8', 'F7'),
    'frontal_Fp2_Fp1': ('Fp2', 'Fp1'),
    'central_C4_C3':   ('C4', 'C3'),
    'parietal_P4_P3':  ('P4', 'P3'),
    'temporal_T4_T3':  ('T4', 'T3'),
    'temporal_T6_T5':  ('T6', 'T5'),
    'occipital_O2_O1': ('O2', 'O1'),
}
LEFT_CH = ['Fp1', 'F7', 'F3', 'T3', 'C3', 'T5', 'P3', 'O1']
RIGHT_CH = ['Fp2', 'F8', 'F4', 'T4', 'C4', 'T6', 'P4', 'O2']


def check_channel_order(preprocessed_dir):
    # The arrays must be stored in CHANNEL_NAMES order (written by 01_preprocessing.ipynb).
    with open(os.path.join(preprocessed_dir, 'channel_order.json')) as fh:
        saved = json.load(fh)['channel_names']
    if saved != CHANNEL_NAMES:
        raise ValueError(f'channel_order.json {saved} differs from CHANNEL_NAMES {CHANNEL_NAMES}')
    used = set(sum(REGIONS.values(), [])) | set(sum(ASYMMETRY_PAIRS.values(), ())) | set(LEFT_CH + RIGHT_CH)
    assert used <= set(CHANNEL_NAMES)
    assert all(r in RIGHT_CH and l in LEFT_CH for r, l in ASYMMETRY_PAIRS.values())


def parameters():
    # All analysis parameters of this module, for export and for run signatures.
    names = ['CONDITIONS', 'SFREQ', 'UV_PER_V', 'MIN_WINDOWS', 'BLOCK_WINDOWS', 'MIN_BLOCKS',
             'N_TARGET_PERCENTILE', 'SEED', 'ALPHA_FDR', 'N_BOOT', 'BANDS', 'ALPHA_SUB', 'WELCH_NPERSEG',
             'SAMPEN_M', 'SAMPEN_R_FAC', 'PERMEN_ORDER', 'PERMEN_DELAY', 'HIGUCHI_KMAX', 'CWT_WAVELET',
             'CWT_N_FREQS_PER_BAND', 'CWT_EDGE_CYCLES', 'COHERENCE_NPERSEG', 'CHANNEL_NAMES', 'REGIONS',
             'ASYMMETRY_PAIRS', 'LEFT_CH', 'RIGHT_CH']
    return {n: globals()[n] for n in names}


# =============================================================================
# Spectral domain (2-s windows)
# =============================================================================

def band_mask(freqs, lo, hi, include_hi=False):
    freqs = np.asarray(freqs)
    return (freqs >= lo) & (freqs <= hi) if include_hi else (freqs >= lo) & (freqs < hi)


def compute_psd_welch(window, sfreq=SFREQ, nperseg=WELCH_NPERSEG):
    # window: (samples, channels). Returns freqs (F,) and psd (channels, F).
    f, _ = signal.welch(window[:, 0], fs=sfreq, nperseg=nperseg)
    psd = np.zeros((window.shape[1], len(f)))
    for c in range(window.shape[1]):
        _, psd[c] = signal.welch(window[:, c], fs=sfreq, nperseg=nperseg)
    return f, psd


def band_power(freqs, psd, bands):
    df = freqs[1] - freqs[0]
    total = np.sum(psd[:, (freqs >= 1) & (freqs <= 45)], axis=1) * df
    absp, relp = {}, {}
    items = list(bands.items())
    for k, (name, (lo, hi)) in enumerate(items):
        m = band_mask(freqs, lo, hi, include_hi=(k == len(items) - 1))
        absp[name] = np.sum(psd[:, m], axis=1) * df
        relp[name] = absp[name] / (total + 1e-12)
    return absp, relp


def individual_alpha_frequency(freqs, psd, lo=8, hi=13):
    m = (freqs >= lo) & (freqs <= hi)
    return freqs[m][np.argmax(psd[:, m], axis=1)]


def spectral_entropy(freqs, psd, lo=1, hi=45):
    # Shannon entropy of the normalized PSD, divided by log2 of the number of bins.
    pb = psd[:, (freqs >= lo) & (freqs <= hi)]
    pn = pb / (pb.sum(axis=1, keepdims=True) + 1e-12)
    return -np.sum(pn * np.log2(pn + 1e-12), axis=1) / np.log2(pb.shape[1])


def spectral_edge_frequency(freqs, psd, pct=0.95, lo=1, hi=45):
    m = (freqs >= lo) & (freqs <= hi)
    fb, pb = freqs[m], psd[:, m]
    df = freqs[1] - freqs[0]
    sef = np.zeros(psd.shape[0])
    for c in range(psd.shape[0]):
        cs = np.cumsum(pb[c]) * df
        sef[c] = fb[min(np.searchsorted(cs, pct * cs[-1]), len(fb) - 1)]
    return sef


def compute_spectral_features(window):
    f, psd = compute_psd_welch(window)
    ab, rl = band_power(f, psd, BANDS)
    ab2, rl2 = band_power(f, psd, ALPHA_SUB)
    ab.update(ab2)
    rl.update(rl2)
    return {'abs': ab, 'rel': rl, 'iaf': individual_alpha_frequency(f, psd),
            'se': spectral_entropy(f, psd), 'sef': spectral_edge_frequency(f, psd)}


# =============================================================================
# Time-frequency domain (10-s blocks): complex Morlet CWT
# =============================================================================

def cwt_band_frequencies(lo, hi, n=CWT_N_FREQS_PER_BAND):
    return lo + (np.arange(n, dtype=float) + 0.5) * (hi - lo) / float(n)


def compute_tf_features(epoch, sfreq=SFREQ):
    feat = {}
    for bname, (lo, hi) in BANDS.items():
        scales = pywt.frequency2scale(CWT_WAVELET, cwt_band_frequencies(lo, hi) / float(sfreq))
        pwr = np.zeros((N_CHANNELS, epoch.shape[0]))
        for c in range(N_CHANNELS):
            coef, _ = pywt.cwt(epoch[:, c], scales, CWT_WAVELET, sampling_period=1.0 / sfreq)
            pwr[c] = np.mean(np.abs(coef) ** 2, axis=0)

        trim = min(int(np.ceil(CWT_EDGE_CYCLES * sfreq / float(lo))), max(0, (epoch.shape[0] - 4) // 2))
        valid = pwr[:, trim:epoch.shape[0] - trim] if trim > 0 else pwr
        if valid.shape[1] < 4:
            raise ValueError(f'CWT {bname}: not enough samples after edge trimming.')

        stats_ = {'mean': valid.mean(axis=1), 'std': valid.std(axis=1)}
        stats_['cv'] = stats_['std'] / (stats_['mean'] + 1e-12)
        stats_['iqr'] = np.percentile(valid, 75, axis=1) - np.percentile(valid, 25, axis=1)
        for ci, ch in enumerate(CHANNEL_NAMES):
            for s in ('mean', 'std', 'cv', 'iqr'):
                feat[f'wt_{s}_{bname}_{ch}'] = stats_[s][ci]
        for rn, rch in REGIONS.items():
            ri = [CH_IDX[c] for c in rch]
            for s in ('mean', 'std', 'cv', 'iqr'):
                feat[f'wt_{s}_{bname}_{rn}'] = stats_[s][ri].mean()
    return feat


# =============================================================================
# Complexity: SampEn, PermEn and Higuchi FD in 2-s windows; Hurst in 10-s blocks
# =============================================================================

def sample_entropy(x, m=SAMPEN_M, r_fac=SAMPEN_R_FAC):
    # Chebyshev distance, matches with d <= r, no self-matches, same N - m templates for m and m + 1.
    x = np.asarray(x, dtype=float)
    r = r_fac * np.std(x)
    n_templates = len(x) - m
    if not np.isfinite(r) or r <= 0 or n_templates < 2:
        return np.nan

    def pair_count(tpls):
        return sum(int(np.sum(np.max(np.abs(tpls[i + 1:] - tpls[i]), axis=1) <= r))
                   for i in range(len(tpls) - 1))

    B = pair_count(np.array([x[i:i + m] for i in range(n_templates)]))
    A = pair_count(np.array([x[i:i + m + 1] for i in range(n_templates)]))
    return np.nan if A == 0 or B == 0 else -np.log(A / B)


def permutation_entropy(x, order=PERMEN_ORDER, delay=PERMEN_DELAY):
    n = len(x) - (order - 1) * delay
    if n <= 0:
        return np.nan
    patterns = [tuple(np.argsort(x[[i + j * delay for j in range(order)]], kind='mergesort'))
                for i in range(n)]
    p = np.array(list(Counter(patterns).values()), dtype=float) / len(patterns)
    return -np.sum(p * np.log2(p)) / np.log2(factorial(order))


def higuchi_fd(x, kmax=HIGUCHI_KMAX):
    x = np.asarray(x, dtype=float)
    n_times = x.size
    if n_times < kmax + 2:
        return np.nan
    lk = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = []
        for m in range(k):
            n_max = int(np.floor((n_times - m - 1) / k))
            if n_max < 2:
                continue
            ll = sum(abs(x[m + j * k] - x[m + (j - 1) * k]) for j in range(1, n_max))
            lm.append(ll / k * ((n_times - 1) / (k * n_max)))
        lk[k - 1] = np.mean(lm) if lm else np.nan
    kvec = np.arange(1, kmax + 1, dtype=float)
    valid = np.isfinite(lk) & (lk > 0)
    if valid.sum() < 2:
        return np.nan
    return float(stats.linregress(np.log(1.0 / kvec[valid]), np.log(lk[valid])).slope)


def hurst_exponent(x):
    # Rescaled-range analysis: block divided into k = 2..19 segments of length N // k.
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 20:
        return np.nan
    ns, rss = [], []
    for k in range(2, min(20, N // 2)):
        n = N // k
        if n < 4:
            continue
        rs = []
        for i in range(k):
            seg = x[i * n:(i + 1) * n]
            Y = np.cumsum(seg - seg.mean())
            S = seg.std(ddof=1)
            if S > 0:
                rs.append((Y.max() - Y.min()) / S)
        if rs:
            ns.append(n)
            rss.append(np.mean(rs))
    if len(ns) < 3:
        return np.nan
    return float(stats.linregress(np.log(ns), np.log(rss)).slope)


def compute_complexity_features(window):
    feat = {}
    for ci, ch in enumerate(CHANNEL_NAMES):
        x = window[:, ci]
        feat[f'sampen_{ch}'] = sample_entropy(x[::2] if len(x) > 256 else x)   # 256 points
        feat[f'permen_{ch}'] = permutation_entropy(x)
        feat[f'higuchi_fd_{ch}'] = higuchi_fd(x)
    for rn, rch in REGIONS.items():
        for meas in ('sampen', 'permen', 'higuchi_fd'):
            feat[f'{meas}_{rn}'] = np.nanmean([feat[f'{meas}_{c}'] for c in rch])
    return feat


def compute_hurst_features(long_epochs):
    feat = {}
    for ci, ch in enumerate(CHANNEL_NAMES):
        vals = [h for h in (hurst_exponent(ep[:, ci]) for ep in long_epochs) if np.isfinite(h)]
        feat[f'hurst_{ch}'] = np.nanmean(vals) if vals else np.nan
    for rn, rch in REGIONS.items():
        vals = [feat[f'hurst_{c}'] for c in rch if np.isfinite(feat[f'hurst_{c}'])]
        feat[f'hurst_{rn}'] = np.nanmean(vals) if vals else np.nan
    return feat


# =============================================================================
# Functional connectivity (10-s blocks): coherence and PLV
# =============================================================================

def bandpass_fir(data, lo, hi, sfreq=SFREQ):
    # Linear-phase FIR (Hamming) with reflective padding; data: (samples, channels).
    order = int(3 * sfreq / lo)
    order += (order % 2 == 0)
    order = min(order, data.shape[0] // 3)
    order += (order % 2 == 0)
    coefs = signal.firwin(order, [lo, hi], pass_zero=False, fs=sfreq, window='hamming')
    pad = min(3 * order, data.shape[0] - 1)
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        x = data[:, ch]
        padded = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]])
        out[:, ch] = signal.fftconvolve(padded, coefs, mode='same')[pad:pad + data.shape[0]]
    return out


def compute_coherence(epoch, sfreq=SFREQ, nperseg=COHERENCE_NPERSEG):
    mats = {}
    items = list(BANDS.items())
    for bidx, (bname, (lo, hi)) in enumerate(items):
        C = np.eye(N_CHANNELS)
        for i in range(N_CHANNELS):
            for j in range(i + 1, N_CHANNELS):
                f, Cxy = signal.coherence(epoch[:, i], epoch[:, j], fs=sfreq,
                                          nperseg=nperseg, noverlap=nperseg // 2)
                C[i, j] = C[j, i] = np.mean(Cxy[band_mask(f, lo, hi, include_hi=(bidx == len(items) - 1))])
        mats[bname] = C
    return mats


def compute_plv(epoch, sfreq=SFREQ):
    mats = {}
    for bname, (lo, hi) in BANDS.items():
        filt = bandpass_fir(epoch, lo, hi, sfreq)
        n_trim = min(int(sfreq / lo), epoch.shape[0] // 6)
        if n_trim > 0:
            filt = filt[n_trim:-n_trim, :]
        phase = np.angle(signal.hilbert(filt, axis=0))
        P = np.eye(N_CHANNELS)
        for i in range(N_CHANNELS):
            for j in range(i + 1, N_CHANNELS):
                P[i, j] = P[j, i] = np.abs(np.mean(np.exp(1j * (phase[:, i] - phase[:, j]))))
        mats[bname] = P
    return mats


def connectivity_features(coh_mats, plv_mats):
    feat = {}
    upper = np.triu_indices(N_CHANNELS, k=1)
    ridx = {r: [CH_IDX[c] for c in cs] for r, cs in REGIONS.items()}
    left, right = [CH_IDX[c] for c in LEFT_CH], [CH_IDX[c] for c in RIGHT_CH]
    posterior = ridx['occipital'] + ridx['parietal']
    for bname in BANDS:
        mats = {'coh': coh_mats[bname], 'plv': plv_mats[bname]}
        for m, M in mats.items():
            feat[f'{m}_global_{bname}'] = np.mean(M[upper])
        for rn, idx in ridx.items():
            ui = np.triu_indices(len(idx), k=1)
            for m, M in mats.items():
                feat[f'{m}_{rn}_{bname}'] = np.mean(M[np.ix_(idx, idx)][ui])
        for m, M in mats.items():
            feat[f'{m}_interhemi_{bname}'] = np.mean(M[np.ix_(left, right)])
        for m, M in mats.items():
            feat[f'{m}_fp_{bname}'] = np.mean(M[np.ix_(ridx['frontal'], posterior)])
    return feat


# =============================================================================
# Hemispheric asymmetry and band ratios (subject-mean absolute power)
# =============================================================================

def asymmetry_features(abs_power):
    # ln(right) - ln(left) for 8 pairs and 7 bands (5 bands + low/high alpha).
    feat = {}
    for bname in list(BANDS) + list(ALPHA_SUB):
        pw = abs_power[bname]
        for pname, (rch, lch) in ASYMMETRY_PAIRS.items():
            feat[f'asym_{pname}_{bname}'] = np.log(pw[CH_IDX[rch]] + 1e-12) - np.log(pw[CH_IDX[lch]] + 1e-12)
    return feat


def ratio_features(abs_power):
    # theta/beta, theta/alpha, alpha/beta and delta/alpha per channel; regional ratio = ratio of regional means.
    feat = {}
    for rname, num, den in [('theta_beta', 'theta', 'beta'), ('theta_alpha', 'theta', 'alpha'),
                            ('alpha_beta', 'alpha', 'beta'), ('delta_alpha', 'delta', 'alpha')]:
        for ci, ch in enumerate(CHANNEL_NAMES):
            feat[f'ratio_{rname}_{ch}'] = abs_power[num][ci] / (abs_power[den][ci] + 1e-12)
        for rn, rch in REGIONS.items():
            ri = [CH_IDX[c] for c in rch]
            feat[f'ratio_{rname}_{rn}'] = np.mean(abs_power[num][ri]) / (np.mean(abs_power[den][ri]) + 1e-12)
    return feat


# =============================================================================
# All features of one subject and condition; feature domains and families
# =============================================================================

def extract_subject_features(windows, long_epochs=None, do_conn=True, do_wavelet=True, do_complexity=True):
    # windows: (n, 512, 19) in µV; long_epochs: (n_blocks, 2560, 19) in µV.
    # The do_* flags allow a fast spectral-only extraction (rejection diagnostic).
    if (do_conn or do_wavelet or do_complexity) and (long_epochs is None or len(long_epochs) == 0):
        raise ValueError('No 10-s blocks available.')

    spec = [compute_spectral_features(w) for w in windows]
    feat = {}
    all_bands = list(BANDS) + list(ALPHA_SUB)
    mean_abs = {b: np.mean([s['abs'][b] for s in spec], axis=0) for b in all_bands}
    for b in all_bands:
        mean_rel = np.mean([s['rel'][b] for s in spec], axis=0)
        for ci, ch in enumerate(CHANNEL_NAMES):
            feat[f'abs_{b}_{ch}'] = mean_abs[b][ci]
            feat[f'rel_{b}_{ch}'] = mean_rel[ci]
        for rn, rch in REGIONS.items():
            ri = [CH_IDX[c] for c in rch]
            feat[f'abs_{b}_{rn}'] = mean_abs[b][ri].mean()
            feat[f'rel_{b}_{rn}'] = mean_rel[ri].mean()
    for key, tag in [('iaf', 'iaf'), ('se', 'se'), ('sef', 'sef95')]:
        mv = np.mean([s[key] for s in spec], axis=0)
        for ci, ch in enumerate(CHANNEL_NAMES):
            feat[f'{tag}_{ch}'] = mv[ci]
        for rn, rch in REGIONS.items():
            feat[f'{tag}_{rn}'] = mv[[CH_IDX[c] for c in rch]].mean()
    feat.update(asymmetry_features(mean_abs))
    feat.update(ratio_features(mean_abs))

    if do_wavelet:
        tf = [compute_tf_features(ep) for ep in long_epochs]
        for k in tf[0]:
            feat[k] = np.nanmean([t[k] for t in tf])

    if do_complexity:
        cx = [compute_complexity_features(w) for w in windows]
        for k in cx[0]:
            vals = [c[k] for c in cx if np.isfinite(c[k])]
            feat[k] = np.nanmean(vals) if vals else np.nan
        feat.update(compute_hurst_features(long_epochs))

    if do_conn:
        coh = {b: [] for b in BANDS}
        plv = {b: [] for b in BANDS}
        for ep in long_epochs:
            cm, pm = compute_coherence(ep), compute_plv(ep)
            for b in BANDS:
                coh[b].append(cm[b])
                plv[b].append(pm[b])
        feat.update(connectivity_features({b: np.mean(coh[b], axis=0) for b in BANDS},
                                          {b: np.mean(plv[b], axis=0) for b in BANDS}))
    return feat


def domain_of(name):
    if name.startswith(('abs_', 'rel_', 'iaf', 'se_', 'sef')):
        return 'spectral'
    if name.startswith('wt_'):
        return 'time-frequency'
    if name.startswith(('sampen', 'permen', 'higuchi', 'hurst')):
        return 'complexity'
    if name.startswith(('coh_', 'plv_')):
        return 'connectivity'
    if name.startswith('asym_'):
        return 'asymmetry'
    if name.startswith('ratio_'):
        return 'ratio'
    return 'other'


def family_of(feature):
    # Feature family without location, e.g. 'spectral_rel_beta', 'complexity_hurst'.
    f = str(feature)
    bands = list(BANDS) + list(ALPHA_SUB)
    for kind in ('abs', 'rel'):
        if f.startswith(kind + '_'):
            return f"spectral_{kind}_{next((b for b in bands if f.startswith(f'{kind}_{b}_')), 'other')}"
    for prefix, fam in (('iaf_', 'spectral_iaf'), ('se_', 'spectral_entropy'), ('sef95_', 'spectral_sef95')):
        if f.startswith(prefix):
            return fam
    if f.startswith('wt_'):
        p = f.split('_')
        return f'tf_{p[1]}_{p[2]}'
    for k in ('sampen', 'permen', 'higuchi_fd', 'hurst'):
        if f.startswith(k + '_'):
            return f'complexity_{k}'
    if f.startswith('ratio_'):
        p = f.split('_')
        return f'ratio_{p[1]}_{p[2]}'
    if f.startswith('asym_'):
        return f"asymmetry_{next((b for b in bands if f'_{b}' in f), 'other')}"
    if f.startswith(('coh_', 'plv_')):
        metric = 'coherence' if f.startswith('coh_') else 'plv'
        return f"connectivity_{metric}_{next((b for b in BANDS if f'_{b}_' in f or f.endswith('_' + b)), 'other')}"
    return f'{domain_of(f)}_other'



# =============================================================================
# Analytic sample, 10-s blocks of consecutive windows, equalization
# =============================================================================

def condition_dir(preprocessed_dir, condition):
    return os.path.join(preprocessed_dir, f'classical_{condition}')


def list_subjects(preprocessed_dir, condition):
    return {f[:-len('_windows.npy')] for f in os.listdir(condition_dir(preprocessed_dir, condition))
            if f.endswith('_windows.npy')}


def load_windows(preprocessed_dir, condition, sid):
    # Returns windows in µV (n, 512, 19), original window indices and group label.
    d = condition_dir(preprocessed_dir, condition)
    windows = np.load(os.path.join(d, f'{sid}_windows.npy')).astype(np.float64) * UV_PER_V
    indices = np.load(os.path.join(d, f'{sid}_window_indices.npy')).astype(np.int64)
    group = str(np.load(os.path.join(d, f'{sid}_group.npy'))[0])
    if len(windows) != len(indices) or np.any(np.diff(indices) <= 0):
        raise ValueError(f'{condition}/{sid}: windows and indices do not match.')
    return windows, indices, group


def build_blocks(windows, indices, n_concat=BLOCK_WINDOWS):
    # Non-overlapping blocks of n_concat consecutive windows: (n_blocks, n_concat * 512, 19).
    blocks, block_idx = [], []
    i = 0
    while i <= len(indices) - n_concat:
        inds = indices[i:i + n_concat]
        if np.all(np.diff(inds) == 1):
            blocks.append(np.concatenate(windows[i:i + n_concat], axis=0))
            block_idx.append(inds.copy())
            i += n_concat
        else:
            i += 1
    if not blocks:
        return (np.empty((0, n_concat * windows.shape[1], windows.shape[2])),
                np.empty((0, n_concat), dtype=int))
    return np.stack(blocks), np.stack(block_idx)


def select_sample(preprocessed_dir, verbose=True):
    # Subjects present in all conditions, with MIN_WINDOWS windows and MIN_BLOCKS blocks in each.
    # Returns the sample, n_target, n_long_target and the window and block counts.
    by_cond = {c: list_subjects(preprocessed_dir, c) for c in CONDITIONS}
    union, core = set.union(*by_cond.values()), set.intersection(*by_cond.values())
    n_win = {c: {s: len(np.load(os.path.join(condition_dir(preprocessed_dir, c), f'{s}_window_indices.npy'))) for s in core}
             for c in CONDITIONS}
    n_blk = {c: {s: len(build_blocks(np.zeros((n_win[c][s], 1, 1)),
                                     np.load(os.path.join(condition_dir(preprocessed_dir, c),
                                                          f'{s}_window_indices.npy')))[1])
                 for s in core} for c in CONDITIONS}

    excluded = {}
    for c in CONDITIONS:
        for s in core:
            if n_win[c][s] < MIN_WINDOWS or n_blk[c][s] < MIN_BLOCKS:
                excluded.setdefault(s, []).append(f'{c}: {n_win[c][s]} windows, {n_blk[c][s]} blocks')
    sample = core - set(excluded)

    win_vals = [n_win[c][s] for c in CONDITIONS for s in sample]
    blk_vals = [n_blk[c][s] for c in CONDITIONS for s in sample]
    n_target = int(np.percentile(win_vals, N_TARGET_PERCENTILE))
    n_long_target = min(min(blk_vals), n_target // BLOCK_WINDOWS)
    if n_long_target < MIN_BLOCKS:
        raise RuntimeError(f'Only {n_long_target} common 10-s blocks.')

    if verbose:
        n_hc = lambda s: sum(x.startswith('HC') for x in s)
        print('Subjects per condition:')
        for c in CONDITIONS:
            print(f'  {c:5s}: {len(by_cond[c]):2d} | missing: {sorted(union - by_cond[c]) or "-"}')
        print(f'Present in all conditions: {len(core)} ({n_hc(core)} HC, {len(core) - n_hc(core)} MDD)')
        for s, why in sorted(excluded.items()):
            print(f'  excluded {s}: {"; ".join(why)}')
        print(f'Analytic sample: {len(sample)} ({n_hc(sample)} HC, {len(sample) - n_hc(sample)} MDD)')
        print(f'2-s windows: min {min(win_vals)} | p{N_TARGET_PERCENTILE} {n_target} | '
              f'median {int(np.median(win_vals))} | max {max(win_vals)} -> n_target = {n_target}')
        print(f'10-s blocks: min {min(blk_vals)} | median {int(np.median(blk_vals))} | max {max(blk_vals)} '
              f'| cap n_target // {BLOCK_WINDOWS} = {n_target // BLOCK_WINDOWS} -> n_long_target = {n_long_target}')
    return sample, n_target, n_long_target, n_win, n_blk


def subsample_seed(base_seed, condition, subject, stream='short'):
    # Deterministic seed per condition, subject and stream (2-s windows or 10-s blocks).
    token = f'{int(base_seed)}|{condition}|{subject}' + ('' if stream == 'short' else f'|{stream}')
    return int.from_bytes(hashlib.sha256(token.encode('utf-8')).digest()[:8], 'little') % (2**32 - 1)


def subsample(n, n_target, rng):
    # Positions of a random subset of size n_target without replacement, in temporal order.
    if n_target is None or n <= n_target:
        return np.arange(n, dtype=int)
    return np.sort(rng.choice(n, size=int(n_target), replace=False)).astype(int)


def normalize_rms_global(windows, eps=1e-12):
    # Sensitivity analysis: divide the record by one global RMS (all windows, channels, samples).
    rms = float(np.sqrt(np.nanmean(windows ** 2)))
    if not np.isfinite(rms) or rms <= eps:
        return windows.copy(), np.nan
    return windows / rms, rms


# =============================================================================
# Layer 1 statistics: Mann-Whitney + FDR, Cliff's delta, Hedges' g
# =============================================================================

def cliffs_delta(ctrl, mdd):
    # Positive = second group (MDD) larger.
    return float(np.sign(np.asarray(mdd, float)[:, None] - np.asarray(ctrl, float)[None, :]).mean())


def cliffs_delta_ci(ctrl, mdd, n_boot=N_BOOT, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    nc, nm = len(ctrl), len(mdd)
    b = np.array([cliffs_delta(ctrl[rng.integers(0, nc, nc)], mdd[rng.integers(0, nm, nm)])
                  for _ in range(n_boot)])
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def hedges_g(ctrl, mdd):
    nc, nm = len(ctrl), len(mdd)
    sp = np.sqrt(((nc - 1) * ctrl.var(ddof=1) + (nm - 1) * mdd.var(ddof=1)) / (nc + nm - 2))
    return float((mdd.mean() - ctrl.mean()) / (sp + 1e-12) * (1 - 3 / (4 * (nc + nm) - 9)))


def effect_magnitude(delta):
    a = abs(delta)
    return 'negligible' if a < 0.147 else 'small' if a < 0.330 else 'medium' if a < 0.474 else 'large'


def compare_groups(features, ci=True, seed=SEED):
    # features: DataFrame with 'group' ('control'/'mdd') and one column per feature.
    names = sorted(c for c in features.columns if c not in ('subject', 'group'))
    is_hc = features['group'].eq('control')
    results = []
    for fn in names:
        cv = features.loc[is_hc, fn].to_numpy(float)
        mv = features.loc[~is_hc, fn].to_numpy(float)
        cc, mc = cv[np.isfinite(cv)], mv[np.isfinite(mv)]
        if len(cc) < 3 or len(mc) < 3:
            continue
        st, pv = mannwhitneyu(cc, mc, alternative='two-sided')
        delta = cliffs_delta(cc, mc)
        lo, hi = cliffs_delta_ci(cc, mc, N_BOOT, seed=seed) if ci else (np.nan, np.nan)
        results.append({
            'feature': fn, 'domain': domain_of(fn), 'n_ctrl': len(cc), 'n_mdd': len(mc),
            'ctrl_median': float(np.median(cc)), 'mdd_median': float(np.median(mc)),
            'ctrl_mean': float(cc.mean()), 'ctrl_std': float(cc.std(ddof=1)),
            'mdd_mean': float(mc.mean()), 'mdd_std': float(mc.std(ddof=1)),
            'test': 'Mann-Whitney', 'statistic': float(st), 'p_value': float(pv),
            'cliffs_delta': delta, 'cliff_lo': lo, 'cliff_hi': hi, 'hedges_g': hedges_g(cc, mc),
            'effect': effect_magnitude(delta), 'direction': 'MDD > HC' if delta > 0 else 'HC > MDD',
        })
    out = pd.DataFrame(results)
    rej, pcorr, *_ = multipletests(out['p_value'], alpha=ALPHA_FDR, method='fdr_bh')
    out['p_corrected'] = pcorr
    out['significant'] = rej
    return out.sort_values('p_corrected', kind='stable').reset_index(drop=True)


# =============================================================================
# Layer 1 for one condition; detection pattern
# =============================================================================

def run_condition(preprocessed_dir, condition, subjects, n_target, n_long_target, seed=SEED, ci=True,
                  amplitude_normalization=None, save_dir=None, verbose=True):
    # 2-s windows and 10-s blocks are selected with independent deterministic seeds.
    rows, log = [], []
    subs = sorted(subjects)
    for i, sid in enumerate(subs, 1):
        windows, indices, group = load_windows(preprocessed_dir, condition, sid)
        rms = 1.0
        if amplitude_normalization == 'rms_global':
            windows, rms = normalize_rms_global(windows)
        elif amplitude_normalization is not None:
            raise ValueError("amplitude_normalization must be None or 'rms_global'.")

        short_seed = subsample_seed(seed, condition, sid, 'short')
        pos = subsample(len(windows), n_target, np.random.default_rng(short_seed))

        blocks, block_idx = build_blocks(windows, indices)
        long_seed = subsample_seed(seed, condition, sid, 'long')
        bpos = subsample(len(blocks), n_long_target, np.random.default_rng(long_seed))
        if len(bpos) != n_long_target:
            raise RuntimeError(f'{condition}/{sid}: {len(bpos)} blocks instead of {n_long_target}.')

        feat = extract_subject_features(windows[pos], blocks[bpos])
        rows.append({'subject': sid, 'group': group, **feat})
        log.append({
            'subject': sid, 'group': group, 'condition': condition,
            'global_seed': seed, 'short_seed': short_seed, 'long_seed': long_seed,
            'n_windows_available': len(windows), 'n_windows_used': len(pos),
            'selected_window_indices': ';'.join(map(str, indices[pos].tolist())),
            'n_blocks_available': len(blocks), 'n_blocks_used': len(bpos),
            'selected_block_indices': '|'.join(','.join(map(str, r)) for r in block_idx[bpos].tolist()),
            'amplitude_normalization': amplitude_normalization or 'none', 'global_rms': rms,
        })
        if verbose:
            print(f'  [{i:2d}/{len(subs)}] {sid:8s} ({group:7s}) {len(windows):4d} -> {len(pos):4d} windows | '
                  f'{len(blocks):2d} -> {len(bpos):2d} blocks | {len(feat)} features')

    features = pd.DataFrame(rows)
    results = compare_groups(features, ci=ci, seed=seed)
    if verbose:
        print(f'{condition}: {int(features.group.eq("control").sum())} HC, {int(features.group.eq("mdd").sum())} MDD | '
              f'significant (FDR < {ALPHA_FDR}): {int(results.significant.sum())}/{len(results)}')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        features.to_csv(os.path.join(save_dir, f'features_{condition}.csv'), index=False)
        results.to_csv(os.path.join(save_dir, f'stats_{condition}.csv'), index=False)
        pd.DataFrame(log).to_csv(os.path.join(save_dir, f'sampling_log_{condition}.csv'), index=False)
    return results, features


def layer1_summary(stats_by_cond):
    rows = []
    feats = sorted(set().union(*[set(s['feature']) for s in stats_by_cond.values()]))
    tabs = {c: s.set_index('feature') for c, s in stats_by_cond.items()}
    for f in feats:
        row = {'feature': f, 'domain': domain_of(f)}
        detected = []
        for c in CONDITIONS:
            r = tabs[c].loc[f] if f in tabs[c].index else None
            row[f'detected_{c}'] = bool(r['significant']) if r is not None else False
            row[f'cliffs_delta_{c}'] = float(r['cliffs_delta']) if r is not None else np.nan
            row[f'p_fdr_{c}'] = float(r['p_corrected']) if r is not None else np.nan
            if row[f'detected_{c}']:
                detected.append(c)
        row['n_conditions_detected'] = len(detected)
        row['detection_pattern'] = ('none' if not detected else 'all three' if len(detected) == 3
                                    else '+'.join(detected))
        rows.append(row)
    return pd.DataFrame(rows)


def domain_table(summary):
    t = summary.groupby('domain').agg(
        n=('feature', 'size'),
        **{c: (f'detected_{c}', 'sum') for c in CONDITIONS},
        any=('n_conditions_detected', lambda x: int((x > 0).sum())),
        all_three=('n_conditions_detected', lambda x: int((x == 3).sum())))
    t.loc['total'] = t.sum()
    return t.astype(int)


# =============================================================================
# Parameter tables for the supplementary material
# =============================================================================

def cwt_frequency_table():
    # Requested and effective CWT frequencies, scales and edge trimming per band (Table S2).
    rows = []
    n_block = BLOCK_WINDOWS * 2 * SFREQ
    for band, (lo, hi) in BANDS.items():
        req = cwt_band_frequencies(lo, hi)
        scales = pywt.frequency2scale(CWT_WAVELET, req / float(SFREQ))
        eff = pywt.scale2frequency(CWT_WAVELET, scales) * SFREQ
        trim = min(int(np.ceil(CWT_EDGE_CYCLES * SFREQ / float(lo))), max(0, (n_block - 4) // 2))
        for f_req, scale, f_eff in zip(req, scales, eff):
            rows.append({'band': band, 'band_low_hz': lo, 'band_high_hz': hi, 'wavelet': CWT_WAVELET,
                         'requested_frequency_hz': f_req, 'scale': scale, 'effective_frequency_hz': f_eff,
                         'block_samples': n_block, 'edge_trim_samples_each_side': trim,
                         'edge_trim_seconds_each_side': trim / SFREQ, 'valid_samples': n_block - 2 * trim})
    return pd.DataFrame(rows)


def connectivity_pair_table():
    # Channel pairs averaged in each connectivity summary.
    from itertools import combinations, product
    rows = [('global', a, b) for a, b in combinations(CHANNEL_NAMES, 2)]
    rows += [(region, a, b) for region, chs in REGIONS.items() for a, b in combinations(chs, 2)]
    rows += [('interhemi', a, b) for a, b in product(LEFT_CH, RIGHT_CH)]
    rows += [('fp', a, b) for a, b in product(REGIONS['frontal'], REGIONS['parietal'] + REGIONS['occipital'])]
    return pd.DataFrame(rows, columns=['summary', 'channel_1', 'channel_2'])
