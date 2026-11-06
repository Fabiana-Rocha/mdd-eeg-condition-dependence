"""Layer 2 (Group x Condition interaction) and Layer 3 (equivalence across conditions).

Shared by 03_condition_dependence.ipynb and 04_stability_sensitivity.ipynb.
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from eeg_features import CONDITIONS, domain_of

ALPHA = 0.05
# log10 before the z-score for these prefixes, when all values are > 0.
LOG_PREFIXES = ('abs_', 'ratio_', 'wt_mean_', 'wt_std_', 'wt_iqr_', 'wt_cv_')
PERM_N = 50000
PERM_SEED = 20261029
EQUIV_MARGIN = 0.30       # SD, pre-specified
CONTRASTS = ['EO_vs_EC', 'TASK_vs_EC', 'TASK_vs_EO']


def parameters():
    return {n: globals()[n] for n in ['ALPHA', 'LOG_PREFIXES', 'PERM_N', 'PERM_SEED', 'EQUIV_MARGIN', 'CONTRASTS']}


def long_from_wide(features_by_cond):
    # features_by_cond: {condition: DataFrame with subject, group ('control'/'mdd') and features}.
    # Returns one row per subject and condition; only subjects with all three conditions.
    frames = []
    for cond in CONDITIONS:
        d = features_by_cond[cond].copy()
        d.insert(2, 'condition', cond)
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data['group'] = data['group'].map({'control': 'HC', 'mdd': 'MDD'})
    n_cond = data.groupby('subject')['condition'].nunique()
    data = data[data.subject.isin(n_cond[n_cond == len(CONDITIONS)].index)].copy()
    assert (data.groupby('subject')['group'].nunique() == 1).all(), 'group differs across conditions'
    data['group'] = pd.Categorical(data['group'], categories=['HC', 'MDD'], ordered=True)
    data['condition'] = pd.Categorical(data['condition'], categories=CONDITIONS, ordered=True)
    return data


def load_long_features(results_dir):
    # Reads features_<COND>.csv written by eeg_features.run_condition (exact float round trip).
    return long_from_wide({c: pd.read_csv(os.path.join(results_dir, f'features_{c}.csv'),
                                          float_precision='round_trip') for c in CONDITIONS})


# =============================================================================
# Layer 2
# =============================================================================

def transform(feature, y):
    y = np.array(y, dtype=float, copy=True)
    if feature.startswith(LOG_PREFIXES):
        finite = y[np.isfinite(y)]
        if finite.size and np.all(finite > 0):
            return np.log10(y), 'log10'
        return y, 'none (values <= 0)'
    return y, 'none'


def fit_mixedlm(d):
    # REML fit, trying several optimizers; only converged fits are kept.
    last_error = 'MixedLM did not converge.'
    for method in ('lbfgs', 'bfgs', 'powell'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit = smf.mixedlm('_y ~ C(group) * C(condition)', data=d, groups=d['subject'],
                                  re_formula='1').fit(reml=True, method=method, maxiter=300, disp=False)
            if np.isfinite(fit.llf) and bool(getattr(fit, 'converged', False)):
                return fit, method, None
            last_error = f'{method}: not converged'
        except Exception as e:
            last_error = str(e)
    return None, None, last_error


def wald_test(fit, terms):
    b = fit.params.loc[terms].to_numpy(float)
    cov = fit.cov_params().loc[terms, terms].to_numpy(float)
    stat = float(b.T @ np.linalg.pinv(cov) @ b)
    return stat, len(terms), float(chi2.sf(stat, len(terms)))


def linear_contrast(fit, weights, df):
    names = list(fit.params.index)
    c = np.zeros(len(names))
    for term, w in weights.items():
        c[names.index(term)] = w
    est = float(c @ fit.params.to_numpy(float))
    var = float(c @ fit.cov_params().loc[names, names].to_numpy(float) @ c)
    if not np.isfinite(var) or var <= 0:
        return est, np.nan, np.nan
    se = float(np.sqrt(var))
    return est, se, float(2 * stats.t.sf(abs(est / se), df))


def mixed_model_interaction(data, feature):
    d = data[['subject', 'group', 'condition', feature]].copy()
    d[feature] = pd.to_numeric(d[feature], errors='coerce')
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    n_cond = d.groupby('subject')['condition'].nunique()
    d = d[d.subject.isin(n_cond[n_cond == len(CONDITIONS)].index)].copy()
    base = {'feature': feature, 'domain': domain_of(feature), 'n_subjects': d.subject.nunique(),
            'n_hc': d.loc[d.group == 'HC', 'subject'].nunique(),
            'n_mdd': d.loc[d.group == 'MDD', 'subject'].nunique()}
    if base['n_hc'] < 3 or base['n_mdd'] < 3:
        return {**base, 'model_ok': False, 'error': 'fewer than 3 subjects in a group'}

    y, base['transform'] = transform(feature, d[feature].to_numpy(float))
    y[~np.isfinite(y)] = np.nan
    sd = np.nanstd(y, ddof=1)
    if not np.isfinite(sd) or sd <= 1e-12:
        return {**base, 'model_ok': False, 'error': 'no variance'}
    d['_y'] = (y - np.nanmean(y)) / sd
    d = d[np.isfinite(d['_y'])].copy()
    df_resid = float((len(CONDITIONS) - 1) * (d.subject.nunique() - 2))
    d['group'] = pd.Categorical(d['group'].astype(str), categories=['HC', 'MDD'], ordered=True)
    d['condition'] = pd.Categorical(d['condition'].astype(str), categories=CONDITIONS, ordered=True)

    fit, optimizer, err = fit_mixedlm(d)
    if fit is None:
        return {**base, 'model_ok': False, 'error': err}

    terms = [t for t in fit.fe_params.index if ':' in t and 'C(group)' in t and 'C(condition)' in t]
    wald, df_int, p_wald = wald_test(fit, terms)
    t_eo = next(t for t in terms if '[T.EO]' in t)
    t_task = next(t for t in terms if '[T.TASK]' in t)
    contrasts = {'EO_vs_EC': {t_eo: 1.0}, 'TASK_vs_EC': {t_task: 1.0}, 'TASK_vs_EO': {t_task: 1.0, t_eo: -1.0}}
    out = {**base, 'model_ok': True, 'optimizer': optimizer, 'wald_interaction': wald,
           'df_interaction': df_int, 'p_interaction_wald_chi2': p_wald, 'df_resid': df_resid}
    p_post = []
    for name, w in contrasts.items():
        est, se, p = linear_contrast(fit, w, df_resid)
        out[f'delta_effect_{name}_z'], out[f'se_{name}_z'], out[f'p_{name}'] = est, se, p
        p_post.append(p)
    p_post = np.array(p_post)
    holm = np.full(3, np.nan)
    ok = np.isfinite(p_post)
    if ok.any():
        holm[ok] = multipletests(p_post[ok], method='holm')[1]
    for name, p in zip(contrasts, holm):
        out[f'p_{name}_holm'] = p
    out['random_intercept_var'] = float(fit.cov_re.iloc[0, 0])
    out['error'] = ''
    return out


def interaction_f(P, G):
    # P: (N, K) subject-centered profiles; G: (B, N) MDD indicators for B labelings.
    N, K = P.shape
    tot = P.sum(axis=0)
    n1 = G.sum(axis=1)[:, None]
    S1 = G @ P
    S0 = tot[None, :] - S1
    between = (S1 ** 2 / n1 + S0 ** 2 / (N - n1)).sum(axis=1)
    ss_int = between - (tot ** 2).sum() / N
    ss_err = (P ** 2).sum() - between
    return (ss_int / (K - 1)) / (ss_err / ((N - 2) * (K - 1)))


def permutation_interaction(data, features, n_perm=None, seed=None, chunk=2000):
    n_perm = PERM_N if n_perm is None else n_perm
    seed = PERM_SEED if seed is None else seed
    d = data.copy()
    d['condition'] = d['condition'].astype(str)
    group_of = d.drop_duplicates('subject').set_index('subject')['group'].astype(str)
    rng = np.random.default_rng(seed)
    cache, rows = {}, []
    for feature in features:
        y, tr = transform(feature, pd.to_numeric(d[feature], errors='coerce'))
        tmp = d[['subject', 'condition']].copy()
        tmp['_y'] = y
        tmp.loc[~np.isfinite(tmp['_y']), '_y'] = np.nan
        w = (tmp.pivot_table(index='subject', columns='condition', values='_y', aggfunc='first')
             .reindex(columns=CONDITIONS).dropna())
        g = group_of.loc[w.index].eq('MDD').to_numpy(float)
        N, K = w.shape
        base = {'feature': feature, 'transform_perm': tr, 'n_subjects_perm': N}
        if g.sum() < 3 or N - g.sum() < 3:
            rows.append(base)
            continue
        Y = w.to_numpy(float)
        P = Y - Y.mean(axis=1, keepdims=True)
        F_obs = float(interaction_f(P, g[None, :])[0])
        df1, df2 = K - 1, (N - 2) * (K - 1)
        key = tuple(w.index)
        if key not in cache:   # the same permutations for every feature with the same subjects
            cache[key] = np.array([rng.permutation(N) for _ in range(n_perm)])
        perms = cache[key]
        n_ge = sum(int(np.sum(interaction_f(P, g[perms[i:i + chunk]]) >= F_obs * (1 - 1e-10)))
                   for i in range(0, n_perm, chunk))
        rows.append({**base, 'F_interaction': F_obs, 'df1': df1, 'df2': df2,
                     'p_interaction_F': float(stats.f.sf(F_obs, df1, df2)),
                     'p_interaction_perm': (1 + n_ge) / (n_perm + 1)})
    return pd.DataFrame(rows)


def run_layer2(data, features, n_perm=None, save_dir=None, verbose=True):
    import time
    t0 = time.time()
    perm = permutation_interaction(data, features, n_perm=n_perm)
    if verbose:
        print(f'Permutations done ({(time.time() - t0) / 60:.1f} min); fitting mixed models...')
    models = pd.DataFrame([mixed_model_interaction(data, f) for f in features])
    out = models.merge(perm, on='feature', how='left')
    out['p_interaction'] = out['p_interaction_perm']
    valid = np.isfinite(pd.to_numeric(out['p_interaction'], errors='coerce'))
    out['p_interaction_fdr'] = np.nan
    out['interaction_significant'] = False
    reject, q = multipletests(out.loc[valid, 'p_interaction'], alpha=ALPHA, method='fdr_bh')[:2]
    out.loc[valid, 'p_interaction_fdr'] = q
    out.loc[valid, 'interaction_significant'] = reject
    out['interaction_significant'] = out['interaction_significant'].astype(bool)
    out['random_intercept_boundary'] = pd.to_numeric(out['random_intercept_var'], errors='coerce') < 1e-6
    out = out.sort_values(['p_interaction_fdr', 'p_interaction'], na_position='last').reset_index(drop=True)
    if save_dir:
        out.to_csv(os.path.join(save_dir, 'layer2_interaction.csv'), index=False)
    return out


# =============================================================================
# Layer 3
# =============================================================================

def tost_p(est, se, df, margin):
    # Largest of the two one-sided p-values (H0: theta <= -margin and H0: theta >= +margin).
    return np.maximum(stats.t.sf((est + margin) / se, df), stats.t.cdf((est - margin) / se, df))


def run_layer3(layer2, margin=None, save_dir=None):
    margin = EQUIV_MARGIN if margin is None else margin
    out = layer2.copy()
    df = out['df_resid'].to_numpy(float)
    t_crit = stats.t.ppf(1 - ALPHA, df)        # 90% CI for alpha = 0.05
    for c in CONTRASTS:
        est, se = out[f'delta_effect_{c}_z'].to_numpy(float), out[f'se_{c}_z'].to_numpy(float)
        ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
        out[f'p_tost_{c}'] = np.where(ok, tost_p(est, se, df, margin), np.nan)
        out[f'ci90_low_{c}'] = np.where(ok, est - t_crit * se, np.nan)
        out[f'ci90_high_{c}'] = np.where(ok, est + t_crit * se, np.nan)
        out[f'equiv_raw_{c}'] = (out[f'p_tost_{c}'] < ALPHA) & (out[f'ci90_low_{c}'] > -margin) \
                                & (out[f'ci90_high_{c}'] < margin)
    out['p_equivalence_global'] = out[[f'p_tost_{c}' for c in CONTRASTS]].max(axis=1, skipna=False)
    valid = out['model_ok'].fillna(False).astype(bool) & out['p_equivalence_global'].notna()
    out['p_equivalence_fdr'] = np.nan
    out['equivalence_significant'] = False
    reject, q = multipletests(out.loc[valid, 'p_equivalence_global'], alpha=ALPHA, method='fdr_bh')[:2]
    out.loc[valid, 'p_equivalence_fdr'] = q
    out.loc[valid, 'equivalence_significant'] = reject
    out['equivalence_significant'] = out['equivalence_significant'].astype(bool)
    out['equivalence_margin_sd'] = margin

    inter, equiv = out['interaction_significant'], out['equivalence_significant']
    out['formal_condition_status'] = np.select(
        [~valid, inter & ~equiv, ~inter & equiv, inter & equiv],
        ['not-tested', 'condition-dependent', 'condition-equivalent', 'different-but-within-margin'],
        default='inconclusive')
    if save_dir:
        out.to_csv(os.path.join(save_dir, 'layer3_equivalence.csv'), index=False)
    return out

