"""History-only probabilistic forecasts for the released House Edge task.

All local fits use observations strictly before a query's cutoff. Neither
condition labels nor validation targets are inputs to this module.
"""
from collections import defaultdict
from itertools import combinations
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, expit, logsumexp

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.schema import PAYOUTS, index_history


PAIRS = tuple(combinations(range(8), 2))


def prepare(queries, history):
    """Create exact-tick lag features; unknown ticks remain unknown."""
    indexed = index_history(queries, history)
    episodes = defaultdict(list)
    for query in queries:
        episodes[query.episode_id].append(query)
    result = []
    for episode in sorted(episodes):
        qs = sorted(episodes[episode], key=lambda q: q.machine_id)
        cutoff = max(q.cutoff for q in qs)
        sequence = np.full((len(qs), cutoff + 1), -1, dtype=np.int8)
        for i, query in enumerate(qs):
            for play in indexed.get((query.episode_id, query.machine_id), ()):
                sequence[i, play.tick] = PAYOUTS.index(play.payout)
        for i, query in enumerate(qs):
            ticks = np.arange(8, query.cutoff + 1)
            raw_lags = sequence[i, ticks[:, None] - np.arange(1, 9)]
            valid_lags = (raw_lags >= 0).all(axis=1)
            y = sequence[i, ticks]
            observed = (y >= 0) & (ticks < query.cutoff) & valid_lags
            win = np.where(raw_lags >= 0, 2.0 * (raw_lags > 0) - 1, 0)
            big = np.where(raw_lags >= 0, 2.0 * (raw_lags > 1) - 1, 0)
            xparts = [np.ones((len(ticks), 1)), win]
            xparts.append(np.stack([win[:, a] * win[:, b] for a, b in PAIRS], axis=1))
            xparts.append(big)
            xparts.append(np.stack([big[:, a] * big[:, b] for a, b in PAIRS], axis=1))
            # Pool earlier outcomes of the other machines within this episode.
            others = np.delete(sequence, i, axis=0)
            peer_lags = others[:, ticks[:, None] - np.arange(1, 9)]
            known = peer_lags >= 0
            denom = np.maximum(known.sum(axis=0), 1)
            peer_win = np.where(known, 2.0 * (peer_lags > 0) - 1, 0).sum(axis=0) / denom
            peer_big = np.where(known, 2.0 * (peer_lags > 1) - 1, 0).sum(axis=0) / denom
            xparts.extend([peer_win, peer_big])
            x = np.concatenate(xparts, axis=1)
            plays = indexed.get((query.episode_id, query.machine_id), ())
            all_y = np.array([PAYOUTS.index(p.payout) for p in plays], dtype=np.int8)
            all_age = np.array([query.cutoff - 1 - p.tick for p in plays], dtype=float)
            # Last row is the requested cutoff; it is never a training row.
            result.append(dict(query=query, x=x[observed], xnext=x[-1],
                               y=y[observed], age=query.cutoff - 1 - ticks[observed],
                               lag=raw_lags[observed], lagnext=raw_lags[-1],
                               all_y=all_y, all_age=all_age,
                               complete_next=bool(valid_lags[-1])))
    return result


def prior_for(item, priors):
    q = item['query']
    return np.array(priors['context_priors'].get(f'{q.machine_type}|{q.signal}', priors['global_prior']))


def frequency(item, prior, half_life, strength):
    weights = np.exp2(-item['all_age'] / half_life)
    counts = np.bincount(item['all_y'], weights=weights, minlength=5) + strength * prior
    return counts / counts.sum()


def positive_distribution(item, prior, half_life=64.0, strength=60.0):
    positive = prior[1:] / prior[1:].sum()
    counts = np.bincount(item['all_y'], weights=np.exp2(-item['all_age'] / half_life), minlength=5)[1:]
    return (counts + strength * positive) / (counts.sum() + strength)


def logistic_win(item, prior, half_life=96.0, regularization=8.0, features='win', intercept_strength=4.0,
                 feature_indices=None, regularization_weights=None):
    fallback = frequency(item, prior, half_life, 20.0)
    if len(item['y']) < 10 or not item['complete_next']:
        return 1.0 - fallback[0]
    dims = {'lags': 9, 'win': 37, 'amount': 73, 'peers': 89}[features]
    x, xn = item['x'][:, :dims], item['xnext'][:dims]
    if feature_indices is not None:
        x, xn = x[:, feature_indices], xn[feature_indices]
        dims = len(feature_indices)
    y = (item['y'] > 0).astype(float)
    weights = np.exp2(-item['age'] / half_life)
    pw = float(np.clip(1 - prior[0], 1e-5, 1 - 1e-5))
    penalty = np.full(dims, regularization)
    if regularization_weights is not None:
        penalty *= np.array(regularization_weights)
    penalty[0] = 0.0
    initial = np.zeros(dims)
    initial[0] = math.log((1 - fallback[0]) / fallback[0])

    def objective(beta):
        eta = x @ beta
        loss = np.sum(weights * (np.logaddexp(0, eta) - y * eta))
        loss += .5 * np.sum(penalty * beta * beta)
        loss += intercept_strength * (np.logaddexp(0, beta[0]) - pw * beta[0])
        gradient = x.T @ (weights * (expit(eta) - y)) + penalty * beta
        gradient[0] += intercept_strength * (expit(beta[0]) - pw)
        return loss, gradient

    fit = minimize(objective, initial, jac=True, method='L-BFGS-B',
                   options={'maxiter': 70, 'ftol': 1e-10, 'gtol': 1e-6})
    if not np.isfinite(fit.fun) or not np.all(np.isfinite(fit.x)):
        raise RuntimeError('Nonfinite local logistic fit')
    return float(expit(xn @ fit.x))


def mixture_win(item, prior, half_life=96., strength=8., temperature=1., coding='win'):
    """Bayesian average of independent, single-lag, and two-lag models."""
    fallback = frequency(item, prior, half_life, 20.)
    if len(item['y']) < 10 or not item['complete_next']:
        return 1 - fallback[0]
    y = (item['y'] > 0).astype(float)
    weights = np.exp2(-item['age'] / half_life)
    pw = float(np.clip(1 - prior[0], .001, .999))
    preds, evidences = [], []
    codings = ('win', 'big') if coding == 'both' else (coding,)
    for code in codings:
        threshold = 0 if code == 'win' else 1
        lag, nxt = item['lag'] > threshold, item['lagnext'] > threshold
        contexts = [(np.zeros(len(y), dtype=int), 0, 1)]
        contexts += [(lag[:, j].astype(int), int(nxt[j]), 2) for j in range(8)]
        contexts += [(2 * lag[:, a].astype(int) + lag[:, b], 2 * int(nxt[a]) + int(nxt[b]), 4) for a, b in PAIRS]
        for state, next_state, size in contexts:
            wins = np.bincount(state, weights=weights * y, minlength=size)
            losses = np.bincount(state, weights=weights * (1 - y), minlength=size)
            alpha, beta = strength * pw, strength * (1 - pw)
            evidence = np.sum(betaln(alpha + wins, beta + losses) - betaln(alpha, beta))
            prediction = (alpha + wins[next_state]) / (strength + wins[next_state] + losses[next_state])
            preds.append(prediction)
            evidences.append(evidence)
    posterior = np.exp(np.array(evidences) / temperature - logsumexp(np.array(evidences) / temperature))
    return float(posterior @ np.array(preds))


def forecast(items, priors, config):
    result = []
    for item in items:
        prior = prior_for(item, priors)
        if config['family'] == 'frequency':
            p = frequency(item, prior, config['half_life'], config['strength'])
        else:
            parameters = {k: v for k, v in config.items() if k not in ('family', 'amount_half_life', 'amount_strength')}
            fn = logistic_win if config['family'] == 'logistic' else mixture_win
            pw = fn(item, prior, **parameters)
            amount = positive_distribution(item, prior, config.get('amount_half_life', 64), config.get('amount_strength', 60))
            p = np.r_[1 - pw, pw * amount]
        result.append(p)
    return np.array(result)


def fit_priors(queries, history, labels):
    model = FrequencyForecaster('pooled').fit(queries, history, labels)
    return dict(global_prior=model.global_prior, context_priors=model.context_priors)


def learn_relevance(items):
    """Estimate recurring feature relevance using training histories only.

    Squaring each machine's association allows machines to have opposite
    responses. The subtraction approximately corrects sampling noise.
    """
    values = []
    for item in items:
        if len(item['y']) < 10:
            continue
        x = item['x'][:, 1:37]
        y = (item['y'] > 0).astype(float)
        weights = np.exp2(-item['age'] / 32)
        weights /= weights.sum()
        xc, yc = x - weights @ x, y - weights @ y
        covariance = weights @ (xc * yc[:, None])
        variance = weights @ (xc * xc)
        yvariance = weights @ (yc * yc)
        values.append(covariance ** 2 / (variance + .05) - yvariance * np.sum(weights ** 2))
    if not values:
        raise ValueError('Insufficient training history for feature relevance')
    return np.mean(values, axis=0)
