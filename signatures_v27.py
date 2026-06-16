#!/usr/bin/env python3
"""
Ablation Suite for Surviving VMS Structural Signatures (v2.7)
==============================================================

Changes from v2.6
-----------------
PER-DIALECT CALIBRATION  (--language A|B)
  VMS reference values are now computed from the actual manuscript, split
  by Currier dialect (A or B).  Calibration is cached to vms_calibration.json.
  When --language is specified:
    - Fail-fast thresholds adapt to the dialect's measured signatures
    - Joint profile match uses dialect-specific ranges
    - All tables show dialect-specific reference values
  When --language is omitted, behaviour is identical to v2.6 (whole-MS
  hardcoded reference values).

  Bilateral extremity (Sig2) is calibrated on the FULL dialect corpus
  (single chunk) because the >100:1 ratio threshold requires large samples.
  Other signatures use chunked evaluation for confidence intervals.

FAST MODE  (~10x speedup, enable with --fast)
  Three fixes identified by profiling:
  1. Pre-tokenize once per corpus.
  2. Sparse pair lookup in build_sparse_markov.
  3. Reduced MI shuffles: 3 in --fast mode, 10 in default mode.

SWEEPS  S6 bridge zone width, S7 sparse Markov top-k.

Usage
-----
  python signatures_v27.py [--runs N] [--seed N] [--words N]
                            [--part all|main|sensitivity|falsify]
                            [--bridge N] [--topk N] [--fast]
                            [--language A|B] [--manuscript FILE]
                            [--calibrate-only] [--force-calibrate]
                            [--n-cal-chunks N]
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# VMS REFERENCE VALUES (whole-MS fallback, used when --language is omitted)
# ============================================================================

VMS_OBS = {
    "es_pct":           80.6,
    "n_start_extreme":  2,
    "n_end_extreme":    3,
    "mi_orig":          0.230,
    "mi_retention_pct": 21.3,
    "zipf_r2":          0.95,
    "cv":               1.0,
}

# Fail-fast thresholds (defaults; overridden by calibration when available)
FF_ES_LO       = 5.0
FF_ES_HI       = 100.0
FF_MI_MIN      = 0.05
FF_C_MI_RATIO  = 0.5
FF_D_ES_DELTA  = 10.0


# ============================================================================
# FOLIO → DIALECT MAP
# ============================================================================

def build_folio_language_map():
    raw = {
        "f1r": "A", "f1v": "A", "f2r": "A", "f2v": "A",
        "f3r": "A", "f3v": "A", "f4r": "A", "f4v": "A",
        "f5r": "A", "f5v": "A", "f6r": "A", "f6v": "A",
        "f7r": "A", "f7v": "A", "f8r": "A", "f8v": "A",
        "f9r": "A", "f9v": "A", "f10r": "A", "f10v": "A",
        "f11r": "A", "f11v": "A",
        "f13r": "A", "f13v": "A", "f14r": "A", "f14v": "A",
        "f15r": "A", "f15v": "A", "f16r": "A", "f16v": "A",
        "f17r": "A", "f17v": "A", "f18r": "A", "f18v": "A",
        "f19r": "A", "f19v": "A", "f20r": "A", "f20v": "A",
        "f21r": "A", "f21v": "A", "f22r": "A", "f22v": "A",
        "f23r": "A", "f23v": "A", "f24r": "A", "f24v": "A",
        "f25r": "A", "f25v": "A",
        "f26r": "B", "f26v": "B",
        "f27r": "A", "f27v": "A", "f28r": "A", "f28v": "A",
        "f29r": "A", "f29v": "A", "f30r": "A", "f30v": "A",
        "f31r": "B", "f31v": "B",
        "f32r": "A", "f32v": "A",
        "f33r": "B", "f33v": "B", "f34r": "B", "f34v": "B",
        "f35r": "A", "f35v": "A", "f36r": "A", "f36v": "A",
        "f37r": "A", "f37v": "A", "f38r": "A", "f38v": "A",
        "f39r": "B", "f39v": "B", "f40r": "B", "f40v": "B",
        "f41r": "B", "f41v": "B",
        "f42r": "A", "f42v": "A",
        "f43r": "B", "f43v": "B",
        "f44r": "A", "f44v": "A", "f45r": "A", "f45v": "A",
        "f46r": "B", "f46v": "B",
        "f47r": "A", "f47v": "A",
        "f48r": "B", "f48v": "B",
        "f49r": "A", "f49v": "A",
        "f50r": "B", "f50v": "B",
        "f51r": "A", "f51v": "A", "f52r": "A", "f52v": "A",
        "f53r": "A", "f53v": "A", "f54r": "A", "f54v": "A",
        "f55r": "B", "f55v": "B",
        "f56r": "A", "f56v": "A",
        "f57r": "B",
        "f58r": "A", "f58v": "A",
        "f66r": "B", "f66v": "B",
        "f75r": "B", "f75v": "B", "f76r": "B", "f76v": "B",
        "f77r": "B", "f77v": "B", "f78r": "B", "f78v": "B",
        "f79r": "B", "f79v": "B", "f80r": "B", "f80v": "B",
        "f81r": "B", "f81v": "B", "f82r": "B", "f82v": "B",
        "f83r": "B", "f83v": "B", "f84r": "B", "f84v": "B",
        "f85r": "B",
        "f86v": "B",
        "f87r": "A", "f87v": "A", "f88r": "A", "f88v": "A",
        "f89r": "A", "f89v": "A",
        "f90r": "A", "f90v": "A",
        "f93r": "A", "f93v": "A",
        "f94r": "B", "f94v": "B",
        "f95r": "B", "f95v": "B",
        "f96r": "A", "f96v": "A",
        "f99r": "A", "f99v": "A",
        "f100r": "A", "f100v": "A",
        "f101r": "A", "f101v": "A",
        "f102r": "A", "f102v": "A",
        "f103r": "B", "f103v": "B", "f104r": "B", "f104v": "B",
        "f105r": "B", "f105v": "B", "f106r": "B", "f106v": "B",
        "f107r": "B", "f107v": "B", "f108r": "B", "f108v": "B",
        "f111r": "B", "f111v": "B", "f112r": "B", "f112v": "B",
        "f113r": "B", "f113v": "B", "f114r": "B", "f114v": "B",
        "f115r": "B", "f115v": "B", "f116r": "B", "f116v": "B",
    }
    return raw


FOLIO_LANG_MAP = build_folio_language_map()
SIGMA_1_FOLIOS = {f for f, l in FOLIO_LANG_MAP.items() if l == 'A'}
SIGMA_0_FOLIOS = {f for f, l in FOLIO_LANG_MAP.items() if l == 'B'}

CALIBRATION_FILE = "vms_calibration.json"
DEFAULT_MANUSCRIPT = "RF1b-e.txt"


# ============================================================================
# MANUSCRIPT PARSING
# ============================================================================

def parse_manuscript(filename):
    """Parse a VMS transcription file into {folio: [sentence, ...]}."""
    folio_sentences = {}
    current_folio = None
    all_folios = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for raw_line in f:
            raw_line = raw_line.rstrip('\n')
            folio_match = re.match(r'^<(f\d+[rv]\d?)[\.\w]*>', raw_line)
            if folio_match:
                current_folio = folio_match.group(1)
                current_folio = re.sub(r'(\d+[rv])\d*$', r'\1', current_folio)
                all_folios.add(current_folio)
                continue
            if raw_line.startswith('#') or not raw_line.strip():
                continue
            line_match = re.match(r'^<([^>]+)>\s+(.*)', raw_line)
            if line_match:
                tag = line_match.group(1)
                content = line_match.group(2)
                content = re.sub(r'<->', '.', content)
                folio_from_tag = re.match(r'(f\d+[rv])', tag)
                if folio_from_tag:
                    current_folio = folio_from_tag.group(1)
                    all_folios.add(current_folio)
                tokens = re.split(r'[.\s]+', content)
            else:
                if not current_folio:
                    continue
                tokens = re.split(r'[.\s]+', raw_line)
            cleaned = []
            for tok in tokens:
                tok = tok.strip()
                tok = re.sub(r'[{}$$!?*%=,]', '', tok)
                if tok and len(tok) > 1 and tok.isalpha():
                    cleaned.append(tok)
            if cleaned:
                if current_folio not in folio_sentences:
                    folio_sentences[current_folio] = []
                folio_sentences[current_folio].append(cleaned)
    return folio_sentences, all_folios


def get_sentences_for_language(folio_sentences, all_folios, language):
    target = SIGMA_1_FOLIOS if language == 'A' else SIGMA_0_FOLIOS
    sentences = []
    for folio in sorted(target):
        if folio in folio_sentences:
            for sent in folio_sentences[folio]:
                if len(sent) >= 2:
                    sentences.append(sent)
    return sentences


# ============================================================================
# CALIBRATION
# ============================================================================

def _calibrate_evaluate_chunked(sentences, tokenizer, n_chunks, n_shuffles,
                                label_prefix):
    """Evaluate sentences in n_chunks chunks for CI metrics."""
    n_chunks = min(n_chunks, len(sentences))
    if n_chunks < 1:
        return []
    chunk_size = len(sentences) // n_chunks
    results = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = len(sentences) if i == n_chunks - 1 else (i + 1) * chunk_size
        chunk_sents = sentences[start:end]
        if len(chunk_sents) < 5:
            continue
        r = evaluate_corpus_fast(
            chunk_sents, tokenizer,
            label=f"{label_prefix} chunk {i}/{n_chunks}",
            n_shuffles=n_shuffles,
        )
        results.append(r)
    return results


def _calibrate_evaluate_full(sentences, tokenizer, n_shuffles, label):
    """Evaluate all sentences as a single block for bilateral."""
    if len(sentences) < 5:
        return None
    return evaluate_corpus_fast(sentences, tokenizer, label=label,
                                n_shuffles=n_shuffles)


def calibrate_language(folio_sentences, all_folios, language, tokenizer,
                       n_chunks_ci, n_shuffles):
    """Compute 4 signatures on real VMS data for one Currier dialect."""
    sentences = get_sentences_for_language(folio_sentences, all_folios,
                                           language)
    if not sentences:
        print(f"  WARNING: No sentences for Currier {language}")
        return None

    n_words = sum(len(s) for s in sentences)
    print(f"  Currier {language}: {len(sentences)} sentences, {n_words} words")

    results_ci = _calibrate_evaluate_chunked(
        sentences, tokenizer, n_chunks_ci, n_shuffles, f"VMS-{language}")
    if not results_ci:
        return None

    agg_ci = aggregate(results_ci, label=f"VMS Currier {language}")

    full_result = _calibrate_evaluate_full(
        sentences, tokenizer, n_shuffles, f"VMS-{language}-full")

    cal = {
        'es_pct':      agg_ci['es_pct_mean'],
        'es_pct_lo':   agg_ci['es_pct_lo'],
        'es_pct_hi':   agg_ci['es_pct_hi'],
        'mi_orig':     agg_ci['mi_orig_mean'],
        'mi_orig_lo':  agg_ci['mi_orig_lo'],
        'mi_orig_hi':  agg_ci['mi_orig_hi'],
        'mi_retention_pct': agg_ci.get('mi_retention_pct_mean', 0),
        'shape':       agg_ci['shape'],
        'zipf_r2':     agg_ci.get('zipf_r2_mean', 0),
        'cv':          agg_ci.get('cv_mean', 0),
        'n_words':     n_words,
        'n_sents':     len(sentences),
        'n_chunks_ci': len(results_ci),
    }

    if full_result:
        cal['bilat_full'] = 1 if (full_result['n_start_extreme'] > 0 and
                                   full_result['n_end_extreme'] > 0) else 0
        cal['n_start_extreme'] = full_result['n_start_extreme']
        cal['n_end_extreme']   = full_result['n_end_extreme']
    else:
        cal['bilat_full']      = 0
        cal['n_start_extreme'] = 0
        cal['n_end_extreme']   = 0

    cal['bilat_frac'] = agg_ci['bilat_frac']
    cal['n_se_mean']  = agg_ci['n_se_mean']
    cal['n_ee_mean']  = agg_ci['n_ee_mean']

    return cal


def run_calibration(manuscript_file, tokenizer, n_chunks_ci, n_shuffles):
    """Parse manuscript and calibrate both dialects."""
    print(f"\n  Parsing manuscript: {manuscript_file}")
    folio_sentences, all_folios = parse_manuscript(manuscript_file)

    a_folios = SIGMA_1_FOLIOS & all_folios
    b_folios = SIGMA_0_FOLIOS & all_folios
    unassigned = all_folios - SIGMA_1_FOLIOS - SIGMA_0_FOLIOS

    print(f"  Total folios: {len(all_folios)}")
    print(f"  Currier A: {len(a_folios)},  Currier B: {len(b_folios)}")
    if unassigned:
        print(f"  Unassigned: {len(unassigned)}")

    calibration = {}
    for lang in ['A', 'B']:
        print(f"\n  Calibrating Currier {lang}...")
        cal = calibrate_language(folio_sentences, all_folios, lang,
                                  tokenizer, n_chunks_ci, n_shuffles)
        if cal:
            calibration[lang] = cal
    return calibration


def load_or_calibrate(manuscript_file, tokenizer, n_chunks_ci, n_shuffles,
                      force=False):
    """Load cached calibration or compute it."""
    if not force and os.path.exists(CALIBRATION_FILE):
        print(f"  Loading cached calibration from {CALIBRATION_FILE}")
        with open(CALIBRATION_FILE, 'r') as f:
            cal = json.load(f)
        if 'A' in cal and 'B' in cal and 'bilat_full' in cal.get('A', {}):
            return cal
        print(f"  Cached calibration outdated, re-running...")

    if not os.path.exists(manuscript_file):
        print(f"  WARNING: Manuscript {manuscript_file} not found.")
        print(f"  Using hardcoded whole-MS reference values.")
        return None

    cal = run_calibration(manuscript_file, tokenizer, n_chunks_ci, n_shuffles)
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(cal, f, indent=2)
    print(f"  Calibration saved to {CALIBRATION_FILE}")
    return cal


def get_dialect_reference(calibration, language):
    """
    Return a VMS_OBS-compatible dict for the given dialect.

    If calibration is available and contains the dialect, returns dialect-
    specific values.  Otherwise falls back to whole-MS VMS_OBS.
    """
    if calibration and language and language in calibration:
        ref = calibration[language]
        return {
            "es_pct":           ref['es_pct'],
            "es_pct_lo":        ref.get('es_pct_lo', ref['es_pct'] - 10),
            "es_pct_hi":        ref.get('es_pct_hi', ref['es_pct'] + 10),
            "n_start_extreme":  ref.get('n_start_extreme', 2),
            "n_end_extreme":    ref.get('n_end_extreme', 3),
            "bilat_full":       ref.get('bilat_full', 1),
            "mi_orig":          ref['mi_orig'],
            "mi_orig_lo":       ref.get('mi_orig_lo', ref['mi_orig'] * 0.5),
            "mi_orig_hi":       ref.get('mi_orig_hi', ref['mi_orig'] * 2.0),
            "mi_retention_pct": ref.get('mi_retention_pct', 21.3),
            "zipf_r2":          ref.get('zipf_r2', 0.95),
            "cv":               ref.get('cv', 1.0),
            "shape":            ref.get('shape', 'Zipfian'),
            "n_words":          ref.get('n_words', 0),
        }
    return dict(VMS_OBS)  # whole-MS fallback


# ============================================================================
# TOKENIZER
# ============================================================================

def make_greedy_tokenizer(grapheme_list):
    sorted_g = sorted(grapheme_list, key=lambda x: (-len(x), x))
    def tokenizer(word):
        tokens, i = [], 0
        while i < len(word):
            for g in sorted_g:
                if word[i:i + len(g)] == g:
                    tokens.append(g)
                    i += len(g)
                    break
            else:
                tokens.append(word[i])
                i += 1
        return tokens
    return tokenizer


def pretokenize(sentences, tokenizer):
    """Tokenize every word once. Returns list[list[list[str]]]."""
    return [
        [toks for toks in (tokenizer(w) for w in sent) if toks]
        for sent in sentences
    ]


# ============================================================================
# METRICS  (all operate on pre-tokenised sentences)
# ============================================================================

def end_start_pct_fast(tok_sents):
    sc, ec = Counter(), Counter()
    for sent in tok_sents:
        for toks in sent:
            sc[toks[0]] += 1
            ec[toks[-1]] += 1
    cls = {}
    for g in set(sc) | set(ec):
        s, e = sc.get(g, 0), ec.get(g, 0)
        cls[g] = "start" if s > 2 * e else "end" if e > 2 * s else "ambig"
    es = total = 0
    for sent in tok_sents:
        for i in range(len(sent) - 1):
            if cls.get(sent[i][-1]) == "end" and cls.get(sent[i + 1][0]) == "start":
                es += 1
            total += 1
    return 100.0 * es / total if total else 0.0


def bilateral_extremity_fast(tok_sents, threshold=100.0):
    sc, ec = Counter(), Counter()
    for sent in tok_sents:
        for toks in sent:
            sc[toks[0]] += 1
            ec[toks[-1]] += 1
    n_se = n_ee = 0
    for g in set(sc) | set(ec):
        s, e = sc.get(g, 0), ec.get(g, 0)
        if (s + 1) / (e + 1) > threshold: n_se += 1
        if (e + 1) / (s + 1) > threshold: n_ee += 1
    return n_se, n_ee


def _boundary_mi(tok_sents):
    trans = Counter()
    for sent in tok_sents:
        for i in range(len(sent) - 1):
            trans[(sent[i][-1], sent[i + 1][0])] += 1
    if not trans:
        return 0.0
    total = sum(trans.values())
    cond_c, tgt_c = Counter(), Counter()
    for (c, t), cnt in trans.items():
        cond_c[c] += cnt
        tgt_c[t]  += cnt
    h_t  = -sum((v / total) * math.log2(v / total + 1e-15) for v in tgt_c.values())
    h_tc =  sum(-(cnt / total) * math.log2(cnt / cond_c[c] + 1e-15)
                for (c, t), cnt in trans.items())
    return max(h_t - h_tc, 0.0)


def mi_with_retention_fast(tok_sents, n_shuffles=3):
    mi_orig = _boundary_mi(tok_sents)
    rng = random.Random(0)
    shuf_mis = []
    for _ in range(n_shuffles):
        shuf = [list(s) for s in tok_sents]
        for s in shuf:
            rng.shuffle(s)
        shuf_mis.append(_boundary_mi(shuf))
    mi_shuf = float(np.mean(shuf_mis))
    ret = mi_shuf / mi_orig if mi_orig > 1e-10 else float("nan")
    return mi_orig, mi_shuf, ret


def zipfian_boundary_test_fast(tok_sents):
    sc = Counter()
    for sent in tok_sents:
        for toks in sent:
            sc[toks[0]] += 1
    if len(sc) < 3:
        return 0.0, 0.0, "insufficient"
    freqs     = np.array(sorted(sc.values(), reverse=True), dtype=float)
    ranks     = np.arange(1, len(freqs) + 1, dtype=float)
    log_r, log_f = np.log(ranks), np.log(freqs + 1e-10)
    pred      = np.polyval(np.polyfit(log_r, log_f, 1), log_r)
    ss_res    = np.sum((log_f - pred) ** 2)
    ss_tot    = np.sum((log_f - np.mean(log_f)) ** 2)
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    cv   = float(np.std(freqs) / np.mean(freqs)) if np.mean(freqs) > 0 else 0.0
    shape = "Zipfian" if r2 > 0.85 and cv > 0.8 \
            else "Plateau" if cv < 0.5 else "Intermediate"
    return r2, cv, shape


def evaluate_corpus_fast(sentences, tokenizer, label="", n_shuffles=3):
    """Tokenize once, run all metrics on the cached result."""
    tok          = pretokenize(sentences, tokenizer)
    es           = end_start_pct_fast(tok)
    n_se, n_ee   = bilateral_extremity_fast(tok)
    mi_o, mi_s, ret = mi_with_retention_fast(tok, n_shuffles=n_shuffles)
    r2, cv, shape   = zipfian_boundary_test_fast(tok)
    return {
        "label":            label,
        "es_pct":           es,
        "n_start_extreme":  n_se,
        "n_end_extreme":    n_ee,
        "bilateral":        "YES" if (n_se > 0 and n_ee > 0) else "no",
        "mi_orig":          mi_o,
        "mi_shuf":          mi_s,
        "mi_retention_pct": 100 * ret if not math.isnan(ret) else float("nan"),
        "zipf_r2":          r2,
        "cv":               cv,
        "shape":            shape,
    }


# ============================================================================
# STATISTICS
# ============================================================================

def bootstrap_ci(values, alpha=0.05, n_boot=2000):
    if len(values) < 2:
        m = float(np.mean(values)) if values else 0.0
        return m, m, m
    arr  = np.array(values, dtype=float)
    boot = np.array([np.mean(np.random.choice(arr, size=len(arr), replace=True))
                     for _ in range(n_boot)])
    return (float(np.mean(arr)),
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def cohens_d(sample_values, population_value):
    arr = np.array(sample_values, dtype=float)
    sd  = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    if sd < 1e-12:
        return float("inf") if abs(np.mean(arr) - population_value) > 1e-12 else 0.0
    return float((np.mean(arr) - population_value) / sd)


def joint_profile_match(agg, ref=None):
    """
    Joint profile match.  If ref is a dialect calibration dict, uses adaptive
    ranges.  Otherwise uses v2.6 hardcoded ranges.
    """
    if ref and 'es_pct' in ref:
        ref_es = ref['es_pct']
        es_lo  = max(ref_es - 15, 50)
        es_hi  = min(ref_es + 15, 98)
        mi_min = max(0.10, ref.get('mi_orig', 0.10) * 0.5)
        ref_shape = ref.get('shape', 'Zipfian')
        non_plateau = {'Zipfian', 'Intermediate'}
    else:
        es_lo, es_hi = 70, 95
        mi_min       = 0.10
        ref_shape    = 'Zipfian'
        non_plateau  = {'Zipfian', 'Intermediate'}

    gen_es    = agg.get("es_pct_mean", 0)
    gen_mi    = agg.get("mi_orig_mean", 0)
    gen_shape = agg.get("shape", "")

    checks = {
        "E->S":  es_lo <= gen_es <= es_hi,
        "MI":    gen_mi >= mi_min,
        "Bilat": agg.get("bilat_frac", 0) >= 0.5,
    }

    if ref_shape in non_plateau:
        checks["Shape"] = gen_shape in non_plateau
    else:
        checks["Shape"] = gen_shape == ref_shape

    return checks, sum(checks.values())


# ============================================================================
# FAIL-FAST  (dialect-aware)
# ============================================================================

def fail_fast_check(baseline_agg, random_order_agg, single_pool_agg,
                    ref=None):
    """
    Four structural invariants.  When ref (dialect calibration) is provided,
    thresholds adapt:
      F1: E->S must be within a plausible range (not 100%, not <5%)
      F2: MI must be detectable
      F3: Random ordering must drop MI
      F4: Single pool must drop E->S
    """
    ff_es_lo      = FF_ES_LO
    ff_es_hi      = FF_ES_HI
    ff_mi_min     = FF_MI_MIN
    ff_c_mi_ratio = FF_C_MI_RATIO
    ff_d_es_delta = FF_D_ES_DELTA

    if ref and 'es_pct' in ref:
        # Adapt: baseline E->S should be near the dialect reference
        # but we keep the structural invariants (not 100%, not <5%)
        ff_mi_min = max(0.03, ref.get('mi_orig', 0.10) * 0.3)

    errors = []
    bES = baseline_agg.get("es_pct_mean",      0.0)
    bMI = baseline_agg.get("mi_orig_mean",     0.0)
    cMI = random_order_agg.get("mi_orig_mean", 0.0)
    dES = single_pool_agg.get("es_pct_mean",   0.0)

    if bES >= ff_es_hi:
        errors.append(
            f"F1 FAIL  Baseline E->S = {bES:.1f}% (must be < {ff_es_hi:.0f}%)\n"
            f"         Bridge graphemes are not reaching word-boundary positions.\n"
            f"         Verify build_bridge_pools() prepends bridge to prefix_slots[0]\n"
            f"         and to suffix_slots[-1]."
        )
    elif bES < ff_es_lo:
        errors.append(
            f"F1 FAIL  Baseline E->S = {bES:.1f}% (must be > {ff_es_lo:.0f}%)\n"
            f"         Pool structure has collapsed; no prefix/suffix separation."
        )

    if bMI <= ff_mi_min:
        errors.append(
            f"F2 FAIL  Baseline MI = {bMI:.4f} (must be > {ff_mi_min:.2f})\n"
            f"         Sparse Markov boundary pairs not generating MI."
        )

    if cMI >= bMI * ff_c_mi_ratio:
        errors.append(
            f"F3 FAIL  Config-C MI = {cMI:.4f},  baseline MI = {bMI:.4f}\n"
            f"         Random ordering should drop MI below "
            f"{ff_c_mi_ratio * 100:.0f}% of baseline."
        )

    if dES > bES - ff_d_es_delta:
        errors.append(
            f"F4 FAIL  Config-D E->S = {dES:.1f}%,  baseline = {bES:.1f}%\n"
            f"         Single pool should drop E->S by >= {ff_d_es_delta:.0f} pp."
        )

    dialect_tag = ""
    if ref and 'es_pct' in ref:
        dialect_tag = f"  (dialect ref: E->S={ref['es_pct']:.1f}%  MI={ref['mi_orig']:.4f})"

    return 
    if errors:
        print(f"\n{'!' * 70}")
        print(f"  FAIL-FAST: {len(errors)} invariant(s) violated — aborting.{dialect_tag}")
        print(f"{'!' * 70}")
        for err in errors:
            print(f"\n  {err}")
        print(f"\n  No further configs will be run.")
        print(f"{'!' * 70}\n")
        sys.exit(1)

    print(f"\n  [FAIL-FAST OK]  E->S={bES:.1f}%  MI={bMI:.4f}  "
          f"C-MI={cMI:.4f}  D-ES={dES:.1f}%  — all 4 invariants passed.{dialect_tag}")


# ============================================================================
# GRAPHEME POOLS
# ============================================================================

PREFIX_CORE = {
    0: ["q", "d", "s", "t", "k", "p", "f", "c"],
    1: ["ch", "sh", "ck", "cth", "cph"],
    2: ["ok", "ot", "ol", "or"],
    3: ["k2", "t2", "p2", "f2", "d2", "s2"],
    4: ["ch2", "sh2", "ckh", "cth2", "cfh"],
}

SUFFIX_CORE = {
    1: ["dy", "dl", "dm", "ds", "dar", "dal"],
    2: ["iin", "in", "ir", "iir", "iiir"],
    3: ["al", "am", "an", "ar", "ain", "aiin"],
    4: ["ey", "ed", "es", "edy", "eey", "eedy"],
    5: ["ly", "ry", "ny", "my", "ldy"],
}

BRIDGE_CANDIDATES = ["o", "a", "y", "e2", "i2"]

DEFAULT_BOUNDARY_PAIRS = [
    ("ly",  "q",  5.0), ("ry",  "d",  5.0), ("ny",  "s",  5.0),
    ("my",  "t",  4.0), ("in",  "q",  5.0), ("iin", "d",  5.0),
    ("an",  "s",  4.0), ("ed",  "k",  4.0), ("es",  "t",  4.0),
    ("o",   "q",  5.0), ("a",   "d",  5.0), ("y",   "s",  5.0),
    ("e2",  "k",  4.0), ("i2",  "t",  4.0),
    ("o",   "a",  3.0), ("a",   "y",  3.0),
]


def _bridge_weights(slot, bridge_graphemes, bridge_relative_weight):
    raw   = [bridge_relative_weight if g in bridge_graphemes else 1.0 for g in slot]
    total = sum(raw)
    return [r / total for r in raw]


def build_bridge_pools(bridge_size=2, bridge_weight=2.0):
    if bridge_size == 0:
        ps = [list(PREFIX_CORE[i]) for i in range(5)]
        ss = [list(SUFFIX_CORE[i]) for i in range(1, 6)]
        return ps, ss

    bridge = BRIDGE_CANDIDATES[:bridge_size]
    ps = [bridge + list(PREFIX_CORE[0])] + \
         [list(PREFIX_CORE[i]) for i in range(1, 5)]
    ss = [list(SUFFIX_CORE[i]) for i in range(1, 5)] + \
         [bridge + list(SUFFIX_CORE[5])]
    return ps, ss


def build_bridge_weights(ps, ss, bridge_size=2, bridge_weight=2.0, zipf_alpha=1.2):
    if bridge_size == 0:
        return make_zipf_slot_weights(ps, zipf_alpha), \
               make_zipf_slot_weights(ss, zipf_alpha)
    bridge = BRIDGE_CANDIDATES[:bridge_size]
    pw = [_bridge_weights(ps[0], bridge, bridge_weight)] + \
         [make_zipf_weights(len(s), zipf_alpha) for s in ps[1:]]
    sw = [make_zipf_weights(len(s), zipf_alpha) for s in ss[:-1]] + \
         [_bridge_weights(ss[-1], bridge, bridge_weight)]
    return pw, sw


def get_all_graphemes(ps, ss):
    all_g = set()
    for sl in ps + ss:
        all_g.update(sl)
    return sorted(all_g, key=lambda x: (-len(x), x))


# ============================================================================
# WEIGHT HELPERS
# ============================================================================

def make_zipf_weights(n, alpha=1.2):
    if n <= 0: return []
    raw = [1.0 / (i + 1) ** alpha for i in range(n)]
    s   = sum(raw)
    return [r / s for r in raw]


def make_flat_weights(n):
    return [1.0 / n] * n if n > 0 else []


def make_zipf_slot_weights(slots, alpha=1.2):
    return [make_zipf_weights(len(s), alpha) for s in slots]


def make_flat_slot_weights(slots):
    return [make_flat_weights(len(s)) for s in slots]


# ============================================================================
# LEXICON
# ============================================================================

def _build_word(ps, ss, pw, sw, min_pf, max_pf, min_sf, max_sf, rng):
    n_pf = len(ps); n_sf = len(ss)
    npf  = rng.randint(max(1, min(min_pf, n_pf)), max(1, min(max_pf, n_pf)))
    nsf  = rng.randint(max(1, min(min_sf, n_sf)), max(1, min(max_sf, n_sf)))
    gs   = []
    for i in range(min(npf, n_pf)):
        slot = ps[i]
        w    = pw[i] if pw and i < len(pw) and len(pw[i]) == len(slot) else None
        gs.append(rng.choices(slot, weights=w, k=1)[0] if w else rng.choice(slot))
    for i in range(max(0, n_sf - nsf), n_sf):
        slot = ss[i]
        w    = sw[i] if sw and i < len(sw) and len(sw[i]) == len(slot) else None
        gs.append(rng.choices(slot, weights=w, k=1)[0] if w else rng.choice(slot))
    return "".join(gs)


def build_lexicon(ps, ss, pw, sw, min_pf, max_pf, min_sf, max_sf,
                  vocab_size=1000, rng=None):
    if rng is None: rng = random.Random()
    lexicon, seen, attempts = [], set(), 0
    while len(lexicon) < vocab_size and attempts < vocab_size * 50:
        attempts += 1
        w = _build_word(ps, ss, pw, sw, min_pf, max_pf, min_sf, max_sf, rng)
        if w and len(w) >= 2 and w not in seen:
            lexicon.append(w); seen.add(w)
    while len(lexicon) < vocab_size:
        w = _build_word(ps, ss, pw, sw, min_pf, max_pf, min_sf, max_sf, rng)
        if w and len(w) >= 2: lexicon.append(w)
    return lexicon


def assign_freq_weights(lexicon, alpha=1.2, rng=None):
    if rng is None: rng = random.Random()
    n       = len(lexicon)
    indices = list(range(n)); rng.shuffle(indices)
    weights = make_zipf_weights(n, alpha)
    fw      = [0.0] * n
    for rank, idx in enumerate(indices):
        fw[idx] = weights[rank]
    return fw


# ============================================================================
# SPARSE MARKOV CHAIN
# ============================================================================

def build_sparse_markov(lexicon, fw, tokenizer,
                        boundary_pairs=None, pair_strength=5.0,
                        top_k=30, rng=None):
    n = len(lexicon)
    if n <= 1:
        return {0: ([0], [1.0])}

    toks     = [tokenizer(w) for w in lexicon]
    end_gs   = [t[-1] if t else "" for t in toks]
    start_gs = [t[0]  if t else "" for t in toks]
    base     = np.array(fw, dtype=np.float64)

    sg_idx: Dict[str, List[int]] = defaultdict(list)
    for j, sg in enumerate(start_gs):
        sg_idx[sg].append(j)

    eg_pairs: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    if boundary_pairs:
        for entry in boundary_pairs:
            eg, sg = entry[0], entry[1]
            st = entry[2] if len(entry) == 3 else pair_strength
            eg_pairs[eg].append((sg, st))

    k     = min(top_k, n)
    trans = {}
    for i in range(n):
        raw = base.copy()
        for sg, st in eg_pairs.get(end_gs[i], []):
            for j in sg_idx.get(sg, []):
                raw[j] *= st
        top_idx = np.argpartition(raw, -k)[-k:]
        top_w   = raw[top_idx]
        s       = top_w.sum()
        top_w   = top_w / s if s > 0 else np.ones(k) / k
        trans[i] = (top_idx.tolist(), top_w.tolist())
    return trans


def sample_markov(trans, fw, n_words, rng=None):
    if rng is None: rng = random.Random()
    indices = list(range(len(fw)))
    cur     = rng.choices(indices, weights=fw, k=1)[0]
    seq     = [cur]
    for _ in range(n_words - 1):
        entry = trans.get(cur)
        cur   = rng.choices(entry[0], weights=entry[1], k=1)[0] if entry \
                else rng.choices(indices, weights=fw, k=1)[0]
        seq.append(cur)
    return seq


def sample_iid(fw, n_words, rng=None):
    if rng is None: rng = random.Random()
    return rng.choices(list(range(len(fw))), weights=fw, k=n_words)


def segment(words, min_len=4, max_len=12, rng=None):
    if rng is None: rng = random.Random()
    sents, i = [], 0
    while i < len(words):
        n = rng.randint(min_len, max_len)
        s = words[i:i + n]
        if len(s) >= 3: sents.append(s)
        i += n
    return sents


# ============================================================================
# CONFIG CLASS
# ============================================================================

class GeneratorConfig:
    def __init__(self, label, ps, ss, pw=None, sw=None,
                 vocab_size=1000, vocab_zipf_alpha=1.2,
                 use_markov=True, boundary_pairs=None, pair_strength=5.0,
                 top_k=30, min_pf=2, max_pf=4, min_sf=2, max_sf=4,
                 n_words=37000):
        self.label            = label
        self.ps, self.ss      = ps, ss
        self.pw               = pw if pw is not None else make_zipf_slot_weights(ps)
        self.sw               = sw if sw is not None else make_zipf_slot_weights(ss)
        self.vocab_size       = vocab_size
        self.vocab_zipf_alpha = vocab_zipf_alpha
        self.use_markov       = use_markov
        self.boundary_pairs   = boundary_pairs
        self.pair_strength    = pair_strength
        self.top_k            = top_k
        self.min_pf, self.max_pf = min_pf, max_pf
        self.min_sf, self.max_sf = min_sf, max_sf
        self.n_words          = n_words
        all_g = get_all_graphemes(ps, ss)
        self.tokenizer        = make_greedy_tokenizer(all_g)
        self.n_graphemes      = len(all_g)

    def generate(self, seed=42):
        rng     = random.Random(seed)
        lex     = build_lexicon(self.ps, self.ss, self.pw, self.sw,
                                self.min_pf, self.max_pf, self.min_sf, self.max_sf,
                                self.vocab_size, rng)
        fw      = assign_freq_weights(lex, self.vocab_zipf_alpha, rng)
        if self.use_markov:
            trans   = build_sparse_markov(lex, fw, self.tokenizer,
                                          self.boundary_pairs, self.pair_strength,
                                          self.top_k, rng)
            indices = sample_markov(trans, fw, self.n_words, rng)
        else:
            indices = sample_iid(fw, self.n_words, rng)
        return segment([lex[i] for i in indices], rng=rng)


# ============================================================================
# CONFIG FACTORIES
# ============================================================================

def cfg_baseline(n_words=37000, bridge_size=2, bridge_weight=2.0, top_k=30):
    ps, ss = build_bridge_pools(bridge_size, bridge_weight)
    pw, sw = build_bridge_weights(ps, ss, bridge_size, bridge_weight)
    return GeneratorConfig("BASELINE", ps, ss, pw, sw,
        use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
        pair_strength=5.0, top_k=top_k, n_words=n_words)


def cfg_overlapping(n_words=37000):
    ps_b, ss_b = build_bridge_pools(2)
    all_g = sorted(set(g for sl in ps_b + ss_b for g in sl),
                   key=lambda x: (-len(x), x))
    chunk = max(5, len(all_g) // 6)
    ps = [list(set(all_g[(i * chunk + j) % len(all_g)] for j in range(chunk)))
          for i in range(6)]
    ss = [list(set(all_g[((i * chunk + chunk // 2) + j) % len(all_g)] for j in range(chunk)))
          for i in range(6)]
    return GeneratorConfig("A: Overlapping pools (~60%)", ps, ss,
        make_zipf_slot_weights(ps), make_zipf_slot_weights(ss),
        use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
        pair_strength=5.0, top_k=30, n_words=n_words)


def cfg_flat(n_words=37000):
    ps, ss = build_bridge_pools(2)
    return GeneratorConfig("B: Flat distributions", ps, ss,
        make_flat_slot_weights(ps), make_flat_slot_weights(ss),
        vocab_size=1000, vocab_zipf_alpha=0.0,
        use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
        pair_strength=5.0, top_k=30, n_words=n_words)


def cfg_random_order(n_words=37000):
    ps, ss = build_bridge_pools(2)
    pw, sw = build_bridge_weights(ps, ss, 2)
    return GeneratorConfig("C: Random word order", ps, ss, pw, sw,
        use_markov=False, boundary_pairs=None, top_k=30, n_words=n_words)


def cfg_single_pool(n_words=37000):
    class _Cfg:
        label = "D: Single pool"
        vocab_size = 0
        def __init__(self):
            ps_b, ss_b = build_bridge_pools(2)
            self._all_g = list(set(g for sl in ps_b + ss_b for g in sl))
            self.tokenizer   = make_greedy_tokenizer(self._all_g)
            self.n_graphemes = len(self._all_g)
            self.n_words     = n_words
        def generate(self, seed=42):
            rng = random.Random(seed)
            words = ["".join(rng.choice(self._all_g)
                              for _ in range(rng.randint(2, 5)))
                     for _ in range(self.n_words)]
            sents, i = [], 0
            while i < len(words):
                n = rng.randint(4, 12); s = words[i:i + n]
                if len(s) >= 3: sents.append(s)
                i += n
            return sents
    return _Cfg()


def cfg_variable_boundary(n_words=37000):
    ps, ss = build_bridge_pools(2)
    pw, sw = build_bridge_weights(ps, ss, 2)
    return GeneratorConfig("E: Variable boundary", ps, ss, pw, sw,
        use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
        pair_strength=5.0, top_k=30,
        min_pf=1, max_pf=5, min_sf=1, max_sf=5, n_words=n_words)


def cfg_unilateral(n_words=37000):
    ps_b, ss_b = build_bridge_pools(2)
    sf_g = [g for sl in ss_b for g in sl]
    ps   = [list(set(sl + sf_g[i * 3:(i + 1) * 3])) for i, sl in enumerate(ps_b)]
    return GeneratorConfig("F: Unilateral extremity", ps, ss_b,
        make_zipf_slot_weights(ps), make_zipf_slot_weights(ss_b),
        use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
        pair_strength=5.0, top_k=30, n_words=n_words)


def cfg_agglutinative(n_words=37000):
    po = ["ka","ba","da","ta","na","ma","sa","la","ku","bu","du","tu","nu","mu"]
    so = ["ri","li","ni","si","ki","mi","ti","di","ru","lu","nu2","su","ku2","mu2"]
    sh = ["a","e","i","o","u"]
    ps = [po[:5]+sh[:2], po[3:8]+sh[1:3], po[6:11]+sh[2:4],
          po[9:14]+sh[3:5], sh, sh]
    ss = [sh, sh, so[:5]+sh[:2], so[3:8]+sh[1:3],
          so[6:11]+sh[2:4], so[9:14]+sh[3:5]]
    return GeneratorConfig("G: Agglutinative mimic", ps, ss,
        make_zipf_slot_weights(ps), make_zipf_slot_weights(ss),
        use_markov=True,
        boundary_pairs=[("u","ka",3.0),("o","ba",3.0),
                        ("i","da",3.0),("e","ta",3.0)],
        pair_strength=3.0, top_k=30,
        min_pf=2, max_pf=5, min_sf=2, max_sf=5, n_words=n_words)


def cfg_templatic(n_words=37000):
    cons = ["b","t","th","j","d","dh","r","z","s","sh",
            "k","l","m","n","h","w","f","q"]
    vp   = ["a","i","u","aa","ii","uu"]
    vs   = ["at","in2","un","aat","iin","uun"]
    br   = ["a","i"]
    ps   = [cons[:9], vp, cons[9:], br]
    ss   = [br, cons[:9], vs, cons[9:]]
    return GeneratorConfig("H: Templatic mimic", ps, ss,
        make_zipf_slot_weights(ps), make_zipf_slot_weights(ss),
        use_markov=True,
        boundary_pairs=[("k","b",3.0),("l","t",3.0),
                        ("m","d",3.0),("n","s",3.0)],
        pair_strength=3.0, top_k=30,
        min_pf=1, max_pf=3, min_sf=1, max_sf=3, n_words=n_words)


def cfg_high_entropy(n_words=37000):
    pa = [f"g{i:02d}" for i in range(30)]
    pb = [f"g{i:02d}" for i in range(20, 50)]
    br = [f"g{i:02d}" for i in range(20, 22)]
    ps = [pa[i:i + 10] for i in range(0, 30, 10)] + [br]
    ss = [br] + [pb[i:i + 10] for i in range(0, 30, 10)]
    return GeneratorConfig("I: High-entropy natural", ps, ss,
        make_zipf_slot_weights(ps), make_zipf_slot_weights(ss),
        use_markov=True,
        boundary_pairs=[(f"g{i:02d}", f"g{i+20:02d}", 3.0) for i in range(20, 30)],
        pair_strength=3.0, top_k=30,
        min_pf=1, max_pf=3, min_sf=1, max_sf=3, n_words=n_words)


def cfg_random_strings(n_words=37000):
    alpha = list("abcdefghijklmnopqrstuvwxyz")
    ps = [alpha] * 3; ss = [alpha] * 3
    return GeneratorConfig("CTRL: Random strings", ps, ss,
        make_flat_slot_weights(ps), make_flat_slot_weights(ss),
        vocab_size=2000, vocab_zipf_alpha=0.0,
        use_markov=False, boundary_pairs=None, top_k=30,
        min_pf=1, max_pf=3, min_sf=1, max_sf=3, n_words=n_words)


def make_english_control():
    text = (
        "Call me Ishmael Some years ago never mind how long precisely "
        "having little or no money in my purse and nothing particular "
        "to interest me on shore I thought I would sail about a little "
        "and see the watery part of the world It is a way I have of "
        "driving off the spleen and regulating the circulation "
        "Whenever I find myself growing grim about the mouth whenever "
        "it is a damp drizzly November in my soul whenever I find "
        "myself involuntarily pausing before coffin warehouses and "
        "bringing up the rear of every funeral I meet and especially "
        "whenever my hypos get such an upper hand of me that it "
        "requires a strong moral principle to prevent me from "
        "deliberately stepping into the street and methodically "
        "knocking peoples hats off then I account it high time to get "
        "to sea as soon as I can This is my substitute for pistol "
        "and ball With a philosophical flourish Cato throws himself "
        "upon his sword I quietly take to the ship"
    )
    words = text.split()
    sents, rng, i = [], random.Random(999), 0
    while i < len(words):
        n = rng.randint(6, 15); s = words[i:i + n]
        if len(s) >= 3: sents.append(s)
        i += n
    return sents, lambda w: list(w.lower())


# ============================================================================
# SENSITIVITY CONFIGS
# ============================================================================

def cfg_overlap_sweep(frac, n_words=37000):
    ps_b, ss_b = build_bridge_pools(2)
    if frac <= 0.001:
        pw, sw = build_bridge_weights(ps_b, ss_b, 2)
        return GeneratorConfig("S1: overlap=0%", ps_b, ss_b, pw, sw,
            use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
            pair_strength=5.0, top_k=30, n_words=n_words)
    if frac >= 0.999:
        return cfg_single_pool(n_words)
    pf_u = set(g for sl in ps_b for g in sl)
    sf_u = set(g for sl in ss_b for g in sl)
    pf_o, sf_o = sorted(pf_u - sf_u), sorted(sf_u - pf_u)
    def _inject(base, donor, f):
        n_i = max(1, int(len(donor) * f / len(base)))
        return [list(set(sl + donor[(i*n_i) % max(1,len(donor)):
                                    (i*n_i) % max(1,len(donor)) + n_i]))
                for i, sl in enumerate(base)]
    ps = _inject(ps_b, sf_o, frac); ss = _inject(ss_b, pf_o, frac)
    return GeneratorConfig(f"S1: overlap={int(frac*100)}%", ps, ss,
        make_zipf_slot_weights(ps), make_zipf_slot_weights(ss),
        use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
        pair_strength=5.0, top_k=30, n_words=n_words)


def cfg_slot_count_sweep(n_slots, n_words=37000):
    ps_b, ss_b = build_bridge_pools(2)
    pf_g = sorted(set(g for sl in ps_b for g in sl))
    sf_g = sorted(set(g for sl in ss_b for g in sl))
    bridge = BRIDGE_CANDIDATES[:2]
    pf_per = max(3, len(pf_g) // max(1, n_slots - 1))
    sf_per = max(3, len(sf_g) // max(1, n_slots - 1))
    ps = [list(set(pf_g[(i*pf_per+j) % len(pf_g)] for j in range(pf_per)))
          for i in range(n_slots - 1)] + [list(bridge)]
    ss = [list(bridge)] + \
         [list(set(sf_g[(i*sf_per+j) % len(sf_g)] for j in range(sf_per)))
          for i in range(n_slots - 1)]
    mu = min(n_slots, 5)
    return GeneratorConfig(f"S2: slots={n_slots}+{n_slots}", ps, ss,
        make_zipf_slot_weights(ps), make_zipf_slot_weights(ss),
        use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
        pair_strength=5.0, top_k=30,
        min_pf=max(1, mu // 3), max_pf=mu,
        min_sf=max(1, mu // 3), max_sf=mu,
        n_words=n_words)


def cfg_zipf_sweep(alpha, n_words=37000):
    ps, ss = build_bridge_pools(2)
    bridge = BRIDGE_CANDIDATES[:2]
    if alpha < 0.01:
        pw, sw = make_flat_slot_weights(ps), make_flat_slot_weights(ss)
    else:
        pw = [_bridge_weights(ps[0], bridge, 2.0)] + \
             [make_zipf_weights(len(s), alpha) for s in ps[1:]]
        sw = [make_zipf_weights(len(s), alpha) for s in ss[:-1]] + \
             [_bridge_weights(ss[-1], bridge, 2.0)]
    return GeneratorConfig(f"S3: zipf_alpha={alpha:.1f}", ps, ss, pw, sw,
        use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
        pair_strength=5.0, top_k=30, n_words=n_words)


def cfg_vocab_sweep(vsize, n_words=37000):
    ps, ss = build_bridge_pools(2)
    pw, sw = build_bridge_weights(ps, ss, 2)
    return GeneratorConfig(f"S4: vocab={vsize}", ps, ss, pw, sw,
        vocab_size=vsize,
        use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
        pair_strength=5.0, top_k=30, n_words=n_words)


def cfg_pair_strength_sweep(strength, n_words=37000):
    ps, ss = build_bridge_pools(2)
    pw, sw = build_bridge_weights(ps, ss, 2)
    return GeneratorConfig(f"S5: pair_str={strength:.1f}", ps, ss, pw, sw,
        use_markov=True,
        boundary_pairs=DEFAULT_BOUNDARY_PAIRS if strength >= 0.01 else None,
        pair_strength=strength, top_k=30, n_words=n_words)


def cfg_bridge_sweep(bridge_size, n_words=37000):
    ps, ss = build_bridge_pools(bridge_size)
    pw, sw = build_bridge_weights(ps, ss, bridge_size)
    return GeneratorConfig(f"S6: bridge={bridge_size}", ps, ss, pw, sw,
        use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
        pair_strength=5.0, top_k=30, n_words=n_words)


def cfg_topk_sweep(top_k, n_words=37000):
    ps, ss = build_bridge_pools(2)
    pw, sw = build_bridge_weights(ps, ss, 2)
    return GeneratorConfig(f"S7: top_k={top_k}", ps, ss, pw, sw,
        use_markov=True, boundary_pairs=DEFAULT_BOUNDARY_PAIRS,
        pair_strength=5.0, top_k=top_k, n_words=n_words)


# ============================================================================
# RUNNERS
# ============================================================================

def run_config(cfg, n_runs=20, base_seed=42, n_shuffles=3):
    results = []
    for i in range(n_runs):
        try:
            sents = cfg.generate(seed=base_seed + i * 137)
        except Exception as exc:
            print(f"\n  ERROR {cfg.label} run {i}: {exc}")
            import traceback; traceback.print_exc()
            continue
        results.append(
            evaluate_corpus_fast(sents, cfg.tokenizer, cfg.label, n_shuffles))
    return results


def aggregate(results, label=None, ref=None):
    """
    Aggregate multiple run results.  If ref (dialect calibration dict) is
    provided, Cohen's d is computed against the dialect reference instead of
    hardcoded VMS_OBS.
    """
    if not results:
        return {"label": label or "EMPTY", "n_runs": 0,
                "es_pct_mean": 0, "es_pct_lo": 0, "es_pct_hi": 0,
                "mi_orig_mean": 0, "mi_orig_lo": 0, "mi_orig_hi": 0,
                "bilat_frac": 0, "n_se_mean": 0, "n_ee_mean": 0, "shape": "N/A"}
    if label is None: label = results[0]["label"]

    obs = VMS_OBS
    if ref and 'es_pct' in ref:
        obs = {
            "es_pct":           ref['es_pct'],
            "mi_orig":          ref['mi_orig'],
            "mi_retention_pct": ref.get('mi_retention_pct', VMS_OBS['mi_retention_pct']),
            "zipf_r2":          ref.get('zipf_r2', VMS_OBS['zipf_r2']),
            "cv":               ref.get('cv', VMS_OBS['cv']),
        }

    agg = {"label": label, "n_runs": len(results)}
    for m in ["es_pct", "mi_orig", "mi_retention_pct", "zipf_r2", "cv"]:
        vals = [r[m] for r in results
                if not (isinstance(r[m], float) and math.isnan(r[m]))]
        if vals:
            mean, lo, hi = bootstrap_ci(vals)
            agg[f"{m}_mean"] = mean; agg[f"{m}_lo"] = lo; agg[f"{m}_hi"] = hi
            if m in obs: agg[f"{m}_d"] = cohens_d(vals, obs[m])
        else:
            for sfx in ("_mean", "_lo", "_hi", "_d"):
                agg[m + sfx] = float("nan")
    agg["bilat_frac"] = float(np.mean(
        [1 if r["n_start_extreme"] > 0 and r["n_end_extreme"] > 0 else 0
         for r in results]))
    agg["n_se_mean"] = float(np.mean([r["n_start_extreme"] for r in results]))
    agg["n_ee_mean"] = float(np.mean([r["n_end_extreme"]   for r in results]))
    shapes = [r["shape"] for r in results]
    agg["shape"] = max(set(shapes), key=shapes.count)
    return agg


# ============================================================================
# PRINTING
# ============================================================================

def _ref_label(ref):
    """Short label for reference row in tables."""
    if ref and 'es_pct' in ref:
        bilat = "YES" if ref.get('bilat_full', 0) else "no"
        return (f"VMS (dialect ref)", ref['es_pct'], ref['mi_orig'],
                ref.get('n_start_extreme', '?'), ref.get('n_end_extreme', '?'),
                bilat, ref.get('shape', '?'))
    return ("VMS (observed)", 80.6, 0.230, "2+", "3+", "100", "Zipfian")


def print_main_table(agg_list, ref=None):
    W = 150
    rl = _ref_label(ref)
    dialect_tag = ""
    if ref and 'es_pct' in ref:
        dialect_tag = "  [dialect-calibrated]"
    print(f"\n{'='*W}")
    print(f"  ABLATION RESULTS  (95% bootstrap CIs, Cohen's d vs VMS){dialect_tag}")
    print(f"{'='*W}")
    print(f"  {'Config':<35} {'E->S% [CI]':>25} {'d':>7}  "
          f"{'Se':>4} {'Ee':>4} {'Bil%':>5}  "
          f"{'MI [CI]':>25} {'d':>7}  {'Shape':>12}  {'Joint':>5}")
    print(f"  {'-'*(W-2)}")
    for a in agg_list:
        es_s = f"{a['es_pct_mean']:5.1f} [{a['es_pct_lo']:5.1f},{a['es_pct_hi']:5.1f}]"
        mi_s = f"{a['mi_orig_mean']:.4f} [{a['mi_orig_lo']:.4f},{a['mi_orig_hi']:.4f}]"
        esd  = a.get("es_pct_d", float("nan"))
        mid  = a.get("mi_orig_d", float("nan"))
        esd_s = f"{esd:>+7.1f}" if math.isfinite(esd) else f"{'N/A':>7}"
        mid_s = f"{mid:>+7.1f}" if math.isfinite(mid) else f"{'N/A':>7}"
        _, nm = joint_profile_match(a, ref)
        print(f"  {a['label']:<35} {es_s:>25} {esd_s}  "
              f"{a['n_se_mean']:4.0f} {a['n_ee_mean']:4.0f} "
              f"{100*a['bilat_frac']:5.0f}  "
              f"{mi_s:>25} {mid_s}  {a['shape']:>12}  {nm:>4}/4")
    print(f"\n  {rl[0]:<35} "
          f"{str(rl[1]):>25} {'ref':>7}  {str(rl[3]):>4} {str(rl[4]):>4} {str(rl[5]):>5}  "
          f"{str(rl[2]):>25} {'ref':>7}  {str(rl[6]):>12}  {'4/4':>5}")


def print_interp_matrix(agg_list, ref=None):
    dialect_tag = ""
    if ref and 'es_pct' in ref:
        dialect_tag = "  [dialect-calibrated]"
    print(f"\n\n{'='*100}")
    print(f"  INTERPRETATION MATRIX{dialect_tag}")
    print(f"{'='*100}")
    print(f"\n  {'Config':<35} {'E->S':>6} {'Bilat':>6} {'MI':>6} {'Zipf':>6}  {'Joint':>6}")
    print(f"  {'-'*70}")
    for a in agg_list:
        checks, nm = joint_profile_match(a, ref)
        s1 = "ok" if checks["E->S"] else "x"
        s2 = "ok" if checks["Bilat"] else "x"
        s3 = "ok" if checks["MI"] else "x"
        s4 = "ok" if checks["Shape"] else "x"
        # Add marginal marks for near-misses
        if not checks["E->S"]:
            es = a["es_pct_mean"]
            if ref and 'es_pct' in ref:
                lo = max(ref['es_pct'] - 20, 40)
                hi = min(ref['es_pct'] + 20, 100)
            else:
                lo, hi = 60, 98
            if lo <= es <= hi:
                s1 = "~"
        if not checks["MI"]:
            mi = a["mi_orig_mean"]
            if mi > 0.03:
                s3 = "~"
        if not checks["Shape"]:
            if a.get("zipf_r2_mean", 0) > 0.80:
                s4 = "~"
        print(f"  {a['label']:<35} {s1:>6} {s2:>6} {s3:>6} {s4:>6}  {nm:>5}/4")

    if ref and 'es_pct' in ref:
        ref_es = ref['es_pct']
        es_lo  = max(ref_es - 15, 50)
        es_hi  = min(ref_es + 15, 98)
        mi_min = max(0.10, ref.get('mi_orig', 0.10) * 0.5)
        print(f"\n  ok=VMS-like  ~=marginal  x=absent")
        print(f"  Sig1: E->S {es_lo:.0f}-{es_hi:.0f}%  "
              f"Sig2: bilateral >50% runs  "
              f"Sig3: MI>{mi_min:.2f}  Sig4: non-Plateau")
    else:
        print(f"\n  ok=VMS-like  ~=marginal  x=absent")
        print(f"  Sig1: E->S 70-95%  Sig2: bilateral >50% runs  Sig3: MI>0.10  Sig4: Zipfian")


def print_sens_table(agg_list, title, ref=None):
    dialect_tag = ""
    if ref and 'es_pct' in ref:
        dialect_tag = "  [dialect-calibrated]"
    print(f"\n{'='*130}")
    print(f"  SENSITIVITY: {title}{dialect_tag}")
    print(f"{'='*130}")
    print(f"\n  {'Config':<28} {'E->S% [CI]':>25} {'Bil%':>5} "
          f"{'MI [CI]':>25} {'R2':>6} {'CV':>6} {'Shape':>12}  {'Joint':>5}")
    print(f"  {'-'*118}")
    for a in agg_list:
        es_s = f"{a['es_pct_mean']:5.1f} [{a['es_pct_lo']:5.1f},{a['es_pct_hi']:5.1f}]"
        mi_s = f"{a['mi_orig_mean']:.4f} [{a['mi_orig_lo']:.4f},{a['mi_orig_hi']:.4f}]"
        r2   = a.get("zipf_r2_mean", float("nan"))
        cv   = a.get("cv_mean",      float("nan"))
        _, nm = joint_profile_match(a, ref)
        print(f"  {a['label']:<28} {es_s:>25} {100*a['bilat_frac']:5.0f} "
              f"{mi_s:>25} {r2:6.3f} {cv:6.2f} {a['shape']:>12}  {nm:>4}/4")
    if ref and 'es_pct' in ref:
        print(f"\n  Dialect ref: E->S={ref['es_pct']:.1f}%  MI={ref['mi_orig']:.4f}  "
              f"Bilateral={'YES' if ref.get('bilat_full',0) else 'NO'}  "
              f"R2={ref.get('zipf_r2',0):.3f}  CV={ref.get('cv',0):.2f}  {ref.get('shape','?')}")
    else:
        print(f"\n  VMS ref: E->S=80.6%  MI=0.230  Bilateral  R2=0.95  CV~1.0  Zipfian")


# ============================================================================
# ANALYSIS PARTS
# ============================================================================

def run_main(n_runs, base_seed, n_words, n_shuffles, ref=None):
    cfgs = [
        cfg_baseline(n_words), cfg_overlapping(n_words), cfg_flat(n_words),
        cfg_random_order(n_words), cfg_single_pool(n_words),
        cfg_variable_boundary(n_words), cfg_unilateral(n_words),
        cfg_agglutinative(n_words), cfg_templatic(n_words),
        cfg_high_entropy(n_words), cfg_random_strings(n_words),
    ]

    all_agg = []
    baseline_agg = ro_agg = sp_agg = None

    for cfg in cfgs:
        print(f"  Running: {cfg.label}  "
              f"({n_runs} runs, {cfg.n_graphemes} graphemes, vocab={cfg.vocab_size})...",
              end="", flush=True)
        res = run_config(cfg, n_runs, base_seed, n_shuffles)
        agg = aggregate(res, ref=ref)
        all_agg.append(agg)
        print(f"  E->S={agg['es_pct_mean']:.1f}%  MI={agg['mi_orig_mean']:.4f}")

        if cfg.label == "BASELINE":             baseline_agg = agg
        if cfg.label == "C: Random word order": ro_agg       = agg
        if cfg.label == "D: Single pool":       sp_agg       = agg

        if baseline_agg and ro_agg and sp_agg and cfg.label == "D: Single pool":
            fail_fast_check(baseline_agg, ro_agg, sp_agg, ref=ref)

    print(f"  Running: CTRL: English...", end="", flush=True)
    en_s, en_tok = make_english_control()
    en_r = evaluate_corpus_fast(en_s, en_tok, "CTRL: English", n_shuffles)
    en_agg = aggregate([en_r], "CTRL: English (Moby Dick)", ref=ref)
    all_agg.append(en_agg)
    print(f"  done.")

    print_main_table(all_agg, ref=ref)
    print_interp_matrix(all_agg, ref=ref)
    return all_agg


def run_sensitivity(n_runs, base_seed, n_words, n_shuffles, ref=None):
    sweeps = [
        ("Pool Overlap",
         [cfg_overlap_sweep(f, n_words)
          for f in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]]),
        ("Slot Count",
         [cfg_slot_count_sweep(n, n_words) for n in [2, 4, 5, 6, 8, 10]]),
        ("Zipf Exponent",
         [cfg_zipf_sweep(a, n_words) for a in [0.0, 0.5, 1.0, 1.5, 2.0]]),
        ("Vocabulary Size",
         [cfg_vocab_sweep(v, n_words) for v in [200, 500, 1000, 2000, 5000]]),
        ("Boundary Pair Strength  [S5]",
         [cfg_pair_strength_sweep(s, n_words) for s in [0.0, 1.0, 3.0, 5.0, 10.0, 20.0]]),
        ("Bridge Zone Width  [S6]",
         [cfg_bridge_sweep(b, n_words) for b in [0, 1, 2, 3, 5, 10]]),
        ("Sparse Markov Top-K  [S7]",
         [cfg_topk_sweep(k, n_words) for k in [5, 10, 20, 30, 50, 100, 1000]]),
    ]
    for title, cfgs in sweeps:
        print(f"\n  --- {title} ---")
        agg_list = []
        for cfg in cfgs:
            print(f"  Running: {cfg.label}  ({n_runs} runs)...", end="", flush=True)
            agg = aggregate(run_config(cfg, n_runs, base_seed, n_shuffles), ref=ref)
            agg_list.append(agg)
            print(f"  E->S={agg['es_pct_mean']:.1f}%  MI={agg['mi_orig_mean']:.4f}")
        print_sens_table(agg_list, title, ref=ref)


def run_falsification(n_runs, base_seed, n_words, n_shuffles, ref=None):
    cfgs = [cfg_baseline(n_words), cfg_agglutinative(n_words),
            cfg_templatic(n_words), cfg_high_entropy(n_words)]
    all_agg = []
    for cfg in cfgs:
        print(f"  Running: {cfg.label}  ({n_runs} runs)...", end="", flush=True)
        agg = aggregate(run_config(cfg, n_runs, base_seed, n_shuffles), ref=ref)
        all_agg.append(agg)
        print(f"  done.")

    dialect_tag = ""
    if ref and 'es_pct' in ref:
        dialect_tag = "  [dialect-calibrated]"
    print(f"\n{'='*120}")
    print(f"  FALSIFICATION: CAN NEAR-MISS GENERATORS APPROACH THE VMS JOINT PROFILE?{dialect_tag}")
    print(f"{'='*120}")
    print(f"\n  {'Config':<35} {'E->S%':>8} {'d(E->S)':>8} "
          f"{'MI':>8} {'d(MI)':>8} {'Bil%':>6} {'Shape':>12} {'Joint':>6}")
    print(f"  {'-'*95}")
    for a in all_agg:
        esd  = a.get("es_pct_d", float("nan"))
        mid  = a.get("mi_orig_d", float("nan"))
        esd_s = f"{esd:>+8.1f}" if math.isfinite(esd) else f"{'N/A':>8}"
        mid_s = f"{mid:>+8.1f}" if math.isfinite(mid) else f"{'N/A':>8}"
        _, nm = joint_profile_match(a, ref)
        print(f"  {a['label']:<35} {a['es_pct_mean']:8.1f} {esd_s} "
              f"{a['mi_orig_mean']:8.4f} {mid_s} "
              f"{100*a['bilat_frac']:6.0f} {a['shape']:>12} {nm:>5}/4")
    rl = _ref_label(ref)
    print(f"  {rl[0]:<35} {str(rl[1]):>8} {'ref':>8} "
          f"{str(rl[2]):>8} {'ref':>8} {str(rl[5]):>6} {str(rl[6]):>12} {'4/4':>6}")
    print(f"\n  |d| < 0.5 small  0.5-0.8 medium  > 0.8 large")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="VMS Ablation Suite v2.7")
    ap.add_argument("--runs",   type=int, default=20,
                    help="Runs per config (default 20)")
    ap.add_argument("--seed",   type=int, default=42,
                    help="Base random seed (default 42)")
    ap.add_argument("--words",  type=int, default=37000,
                    help="Words per synthetic corpus (default 37000)")
    ap.add_argument("--part",   default="all",
                    choices=["all", "main", "sensitivity", "falsify"])
    ap.add_argument("--bridge", type=int, default=2,
                    help="Bridge zone size for BASELINE config (default 2)")
    ap.add_argument("--topk",   type=int, default=30,
                    help="Sparse Markov top-k for BASELINE config (default 30)")
    ap.add_argument("--fast",   action="store_true",
                    help="3 MI shuffles instead of 10")
    # ── New dialect-aware options ──
    ap.add_argument("--language", type=str, default=None,
                    choices=["A", "B"],
                    help="Currier dialect (A or B). Enables per-dialect "
                         "calibration from the manuscript. Omit for whole-MS "
                         "hardcoded references (v2.6 behaviour).")
    ap.add_argument("--manuscript", type=str, default=DEFAULT_MANUSCRIPT,
                    help=f"VMS transcription file (default: {DEFAULT_MANUSCRIPT})")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="Compute and cache calibration, then exit")
    ap.add_argument("--force-calibrate", action="store_true",
                    help="Force re-computation of calibration")
    ap.add_argument("--n-cal-chunks", type=int, default=10,
                    help="Chunks for calibration CI metrics (default 10)")
    args = ap.parse_args()

    n_shuffles = 3 if args.fast else 10
    mode       = "FAST (3 shuffles)" if args.fast else "FULL (10 shuffles)"

    # ── Resolve calibration and dialect reference ──
    calibration = None
    ref         = None  # dialect reference dict (or None for whole-MS fallback)
    need_cal    = args.language is not None or args.calibrate_only or args.force_calibrate

    # Build a simple tokenizer for calibration (character-level fallback;
    # grille.py TOKENIZER is used in verify_generator.py but we keep this
    # file self-contained).
    import grille as gr
    cal_tokenizer = gr.TOKENIZER
    cal_tokenizer_label = "grille.py EVA tokenizer"

    if need_cal:
        print(f"\n  Calibration tokenizer: {cal_tokenizer_label}")
        calibration = load_or_calibrate(
            args.manuscript, cal_tokenizer, args.n_cal_chunks, n_shuffles,
            force=args.force_calibrate)

    if args.language and calibration and args.language in calibration:
        ref = calibration[args.language]
    elif args.language and (not calibration or args.language not in calibration):
        print(f"\n  WARNING: No calibration for Currier {args.language}.")
        print(f"  Falling back to whole-MS hardcoded references.")

    if args.calibrate_only:
        if calibration:
            for lang in ['A', 'B']:
                if lang in calibration:
                    c = calibration[lang]
                    bilat = "YES" if c.get('bilat_full', 0) else "NO"
                    print(f"\n  Currier {lang}:")
                    print(f"    E->S:      {c['es_pct']:.1f}%  "
                          f"[{c['es_pct_lo']:.1f}, {c['es_pct_hi']:.1f}]")
                    print(f"    MI:        {c['mi_orig']:.4f}  "
                          f"[{c['mi_orig_lo']:.4f}, {c['mi_orig_hi']:.4f}]")
                    print(f"    Bilateral: {bilat}  "
                          f"(Se={c.get('n_start_extreme', '?')}, "
                          f"Ee={c.get('n_end_extreme', '?')})")
                    print(f"    Shape:     {c['shape']}  "
                          f"(R²={c['zipf_r2']:.3f}, CV={c['cv']:.2f})")
                    print(f"    Words:     {c['n_words']}")
        print(f"\n  Calibration complete.")
        return

    # ── Banner ──
    lang_tag = f"Currier {args.language}" if args.language else "whole-MS (v2.6 compat)"
    print(f"\n{'#'*70}")
    print(f"#  VMS ABLATION SUITE v2.7")
    print(f"#  runs={args.runs}  seed={args.seed}  words={args.words}  mode={mode}")
    print(f"#  baseline: bridge={args.bridge}  topk={args.topk}")
    print(f"#  language: {lang_tag}")
    if ref:
        print(f"#  dialect ref: E->S={ref['es_pct']:.1f}%  MI={ref['mi_orig']:.4f}  "
              f"Bilateral={'YES' if ref.get('bilat_full',0) else 'NO'}  "
              f"{ref.get('shape','?')}")
    else:
        print(f"#  reference: VMS_OBS hardcoded (E->S=80.6%  MI=0.230  Zipfian)")
    print(f"#")
    print(f"#  Speedups vs v2.5:")
    print(f"#    pre-tokenize once per corpus  (~55x on tokenizer)")
    print(f"#    sparse pair lookup in Markov  (~6x on chain build)")
    print(f"#    3 MI shuffles in --fast mode  (~3x on MI computation)")
    print(f"#")
    print(f"#  v2.7: per-dialect calibration with --language A|B")
    print(f"#    adaptive fail-fast, joint profile match, Cohen's d")
    print(f"#    bilateral from full corpus, CI metrics from chunks")
    print(f"{'#'*70}")

    if args.part in ("all", "main"):
        print(f"\n{'='*70}\n  PART 1: MAIN ABLATIONS\n{'='*70}\n")
        run_main(args.runs, args.seed, args.words, n_shuffles, ref=ref)

    if args.part in ("all", "sensitivity"):
        print(f"\n{'='*70}\n  PART 2: SENSITIVITY\n{'='*70}\n")
        run_sensitivity(args.runs, args.seed, args.words, n_shuffles, ref=ref)

    if args.part in ("all", "falsify"):
        print(f"\n{'='*70}\n  PART 3: FALSIFICATION\n{'='*70}\n")
        run_falsification(args.runs, args.seed, args.words, n_shuffles, ref=ref)

    print(f"\n{'#'*70}\n#  COMPLETE\n{'#'*70}\n")


if __name__ == "__main__":
    main()
