#!/usr/bin/env python3
"""
verify_generator.py — Verify generated Voynich text against the four
structural signatures from Parisel (2025).

Uses signatures_v27.py (same pipeline as 0_scribal_falsify.py).

Bilateral extremity (Sig2) is evaluated on the FULL corpus (1 chunk)
because the >100:1 ratio threshold requires large sample sizes.
Other signatures use multiple chunks for confidence intervals.

Usage:
  python verify_generator.py --words words.txt --language A
  python verify_generator.py --words words.txt --language B --fast
  python verify_generator.py --calibrate-only --language A
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import signatures_v27 as ev
    import grille as gr
    TOKENIZER = gr.TOKENIZER
except ImportError as e:
    sys.exit(
        f"ERROR: {e}\n"
        "signatures_v26.py and grille.py must be in the same directory."
    )

# ============================================================================
# GRAPHEME POOLS  (reuse from signatures_v27 for comparability)
# ============================================================================

PREFIX_GRAPHEMES = (
    ev.BRIDGE_CANDIDATES[:2] +               # o, a  (bridge)
    ev.PREFIX_CORE[0] +                      # q d s t k p f c
    ev.PREFIX_CORE[1] +                      # ch sh ck cth cph
    ev.PREFIX_CORE[2] +                      # ok ot ol or
    ev.PREFIX_CORE[3] +                      # k2 t2 p2 f2 d2 s2
    ev.PREFIX_CORE[4]                        # ch2 sh2 ckh cth2 cfh
)

SUFFIX_GRAPHEMES = (
    ev.BRIDGE_CANDIDATES[:2] +               # o, a  (bridge)
    ev.SUFFIX_CORE[1] +                      # dy dl dm ds dar dal
    ev.SUFFIX_CORE[2] +                      # iin in ir iir iiir
    ev.SUFFIX_CORE[3] +                      # al am an ar ain aiin
    ev.SUFFIX_CORE[4] +                      # ey ed es edy eey eedy
    ev.SUFFIX_CORE[5]                        # ly ry ny my ldy
)

ALL_GRAPHEMES = sorted(
    set(PREFIX_GRAPHEMES) | set(SUFFIX_GRAPHEMES),
    key=lambda x: (-len(x), x)
)


TOKENIZER = ev.make_greedy_tokenizer(ALL_GRAPHEMES)

# ── Folio assignments ──────────────────────────────────────────────────

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
CHCLASS_FILE = "chclass.txt"
MANUSCRIPT_FILE= "RF1b-e.txt"


# ================================================================
# MANUSCRIPT PARSING
# ================================================================

def parse_manuscript(filename):
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


def get_all_words(folio_sentences, all_folios, language):
    target = SIGMA_1_FOLIOS if language == 'A' else SIGMA_0_FOLIOS
    words = set()
    for folio in target:
        if folio in folio_sentences:
            for sent in folio_sentences[folio]:
                words.update(sent)
    return words


# ================================================================
# EVALUATION HELPERS
# ================================================================

def evaluate_chunked(sentences, n_chunks, n_shuffles, label_prefix):
    """Evaluate sentences in n_chunks chunks. Returns list of result dicts."""
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
        r = ev.evaluate_corpus_fast(
            chunk_sents, TOKENIZER,
            label=f"{label_prefix} chunk {i}/{n_chunks}",
            n_shuffles=n_shuffles,
        )
        results.append(r)
    return results


def evaluate_full(sentences, n_shuffles, label):
    """Evaluate all sentences as single block. Returns one result dict."""
    if len(sentences) < 5:
        return None
    return ev.evaluate_corpus_fast(
        sentences, TOKENIZER,
        label=label,
        n_shuffles=n_shuffles,
    )


# ================================================================
# CALIBRATION
# ================================================================

def calibrate_language(folio_sentences, all_folios, language,
                       n_chunks_ci, n_shuffles):
    """Compute 4 signatures on real VMS data for one Currier dialect.
    
    Uses n_chunks_ci chunks for CI metrics (E->S, MI, Shape).
    Uses 1 chunk (full corpus) for bilateral.
    """
    sentences = get_sentences_for_language(folio_sentences, all_folios,
                                           language)
    if not sentences:
        print(f"  WARNING: No sentences for Currier {language}")
        return None

    n_words = sum(len(s) for s in sentences)
    print(f"  Currier {language}: {len(sentences)} sentences, {n_words} words")

    # CI metrics from chunks
    results_ci = evaluate_chunked(sentences, n_chunks_ci, n_shuffles,
                                   f"VMS-{language}")
    if not results_ci:
        return None

    agg_ci = ev.aggregate(results_ci, label=f"VMS Currier {language}")

    # Bilateral from full corpus
    full_result = evaluate_full(sentences, n_shuffles,
                                 f"VMS-{language}-full")

    # Build calibration combining both
    cal = {
        'es_pct': agg_ci['es_pct_mean'],
        'es_pct_lo': agg_ci['es_pct_lo'],
        'es_pct_hi': agg_ci['es_pct_hi'],
        'mi_orig': agg_ci['mi_orig_mean'],
        'mi_orig_lo': agg_ci['mi_orig_lo'],
        'mi_orig_hi': agg_ci['mi_orig_hi'],
        'shape': agg_ci['shape'],
        'zipf_r2': agg_ci.get('zipf_r2_mean', 0),
        'cv': agg_ci.get('cv_mean', 0),
        'n_words': n_words,
        'n_sents': len(sentences),
        'n_chunks_ci': len(results_ci),
    }

    # Bilateral: from full corpus evaluation
    if full_result:
        cal['bilat_full'] = 1 if (full_result['n_start_extreme'] > 0 and
                                   full_result['n_end_extreme'] > 0) else 0
        cal['n_se_full'] = full_result['n_start_extreme']
        cal['n_ee_full'] = full_result['n_end_extreme']
    else:
        cal['bilat_full'] = 0
        cal['n_se_full'] = 0
        cal['n_ee_full'] = 0

    # Also store chunk-based bilateral for reference
    cal['bilat_frac'] = agg_ci['bilat_frac']
    cal['n_se_mean'] = agg_ci['n_se_mean']
    cal['n_ee_mean'] = agg_ci['n_ee_mean']

    return cal


def run_calibration(manuscript_file, n_chunks_ci, n_shuffles):
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
                                  n_chunks_ci, n_shuffles)
        if cal:
            calibration[lang] = cal
    return calibration


def load_or_calibrate(manuscript_file, n_chunks_ci, n_shuffles, force=False):
    if not force and os.path.exists(CALIBRATION_FILE):
        print(f"  Loading cached calibration from {CALIBRATION_FILE}")
        with open(CALIBRATION_FILE, 'r') as f:
            cal = json.load(f)
        # Check for new bilat_full field
        if 'A' in cal and 'B' in cal and 'bilat_full' in cal.get('A', {}):
            return cal
        print(f"  Cached calibration outdated, re-running...")

    cal = run_calibration(manuscript_file, n_chunks_ci, n_shuffles)
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(cal, f, indent=2)
    print(f"  Calibration saved to {CALIBRATION_FILE}")
    return cal


# ================================================================
# ADAPTIVE THRESHOLD LOGIC
# ================================================================

def compute_adaptive_checks(gen_agg, gen_bilat_full, ref):
    """
    Adaptive thresholds calibrated to per-dialect VMS reference.
    
    Sig2 (bilateral) uses full-corpus evaluation for both ref and gen.
    Other signatures use chunk-based CIs.
    """
    ref_es = ref['es_pct']
    ref_mi = ref['mi_orig']
    ref_bilat_full = ref.get('bilat_full', 0)
    ref_shape = ref['shape']

    gen_es = gen_agg.get('es_pct_mean', 0)
    gen_mi = gen_agg.get('mi_orig_mean', 0)
    gen_shape = gen_agg.get('shape', '')

    checks = {}
    reasons = {}

    # Sig1: E->S within ±15pp
    es_lo = max(ref_es - 15, 50)
    es_hi = min(ref_es + 15, 98)
    checks['E->S'] = es_lo <= gen_es <= es_hi
    reasons['E->S'] = f"range [{es_lo:.0f}, {es_hi:.0f}]%"

    # Sig2: Bilateral — full-corpus comparison
    ref_has = ref_bilat_full == 1
    gen_has = gen_bilat_full == 1
    if ref_has:
        # Reference has bilateral — generator must too
        checks['Bilat'] = gen_has
        reasons['Bilat'] = "must have bilateral (ref has it)"
    else:
        # Reference lacks bilateral — generator should also lack it
        checks['Bilat'] = not gen_has
        reasons['Bilat'] = "must lack bilateral (ref lacks it)"

    # Sig3: MI in reasonable range
    mi_lo = max(0.10, ref_mi * 0.5)
    mi_hi = ref_mi * 2.0
    checks['MI'] = mi_lo <= gen_mi <= mi_hi
    reasons['MI'] = f"range [{mi_lo:.3f}, {mi_hi:.3f}]"

    # Sig4: Shape — match or both non-Plateau
    non_plateau = {'Zipfian', 'Intermediate'}
    if ref_shape in non_plateau:
        checks['Shape'] = gen_shape in non_plateau
        reasons['Shape'] = f"non-Plateau (ref={ref_shape})"
    else:
        checks['Shape'] = gen_shape == ref_shape
        reasons['Shape'] = f"match '{ref_shape}'"

    return checks, reasons


# ================================================================
# GENERATED TEXT LOADING
# ================================================================

def load_generated_words(filepath):
    if not os.path.exists(filepath):
        sys.exit(f"ERROR: file not found: {filepath}")
    with open(filepath, encoding='utf-8-sig', errors='replace') as f:
        raw = f.read()
    raw = raw.replace('\r\n', '\n').replace('\r', '\n').strip()
    if not raw:
        sys.exit(f"ERROR: file is empty: {filepath}")

    file_lines = [ln.strip() for ln in raw.split('\n') if ln.strip()]
    if len(file_lines) > 1:
        sentences = []
        all_words = []
        for ln in file_lines:
            words = ln.split()
            all_words.extend(words)
            if len(words) >= 2:
                sentences.append(words)
        return all_words, sentences
    else:
        words = raw.split()
        return words, None


def words_to_sentences(words, min_len=4, max_len=12, seed=42):
    import random as _rng
    r = _rng.Random(seed)
    sentences = []
    i = 0
    while i < len(words):
        n = r.randint(min_len, max_len)
        s = words[i:i + n]
        if len(s) >= 2:
            sentences.append(s)
        i += n
    return sentences


# ================================================================
# ATTESTATION
# ================================================================

def check_attestation(gen_words, manuscript_file, language,
                      manuscript2=None):
    results = {}
    for label, fname in [('chclass.txt', manuscript_file),
                          ('RF1b-e.txt', manuscript2)]:
        if fname is None or not os.path.exists(fname):
            continue
        folio_sents, all_folios = parse_manuscript(fname)
        ms_words = get_all_words(folio_sents, all_folios, language)
        ms_words_all = set()
        for sents in folio_sents.values():
            for sent in sents:
                ms_words_all.update(sent)
        gen_set = set(gen_words)
        results[label] = {
            'ms_types_dialect': len(ms_words),
            'ms_types_all': len(ms_words_all),
            'tok_dialect': sum(1 for w in gen_words if w in ms_words),
            'tok_all': sum(1 for w in gen_words if w in ms_words_all),
            'typ_dialect': sum(1 for w in gen_set if w in ms_words),
            'typ_all': sum(1 for w in gen_set if w in ms_words_all),
            'tok_total': len(gen_words),
            'typ_total': len(gen_set),
            'unattested_sample': sorted(gen_set - ms_words_all)[:20],
        }
    return results


# ================================================================
# REPORT
# ================================================================

def print_report(gen_agg, gen_bilat_full, gen_se_full, gen_ee_full,
                 cal, language, filepath, attestation=None):
    W = 70
    ref = cal[language]

    checks, reasons = compute_adaptive_checks(gen_agg, gen_bilat_full, ref)
    n_pass = sum(checks.values())
    checks_paper, _ = ev.joint_profile_match(gen_agg)

    fname = os.path.basename(filepath)

    print(f"\n{'='*W}")
    print(f"  GENERATOR VERIFICATION — {fname}")
    print(f"  Language: Currier {language}")
    print(f"  {gen_agg.get('n_runs', '?')} CI chunks, "
          f"{gen_agg.get('n_words', '?')} words, "
          f"{gen_agg.get('n_sents', '?')} sentences")
    print(f"{'='*W}")

    # Attestation
    if attestation:
        print(f"\n  WORD ATTESTATION")
        print(f"  {'─'*55}")
        for label, att in attestation.items():
            pct_tok_d = 100 * att['tok_dialect'] / att['tok_total'] if att['tok_total'] else 0
            pct_tok_a = 100 * att['tok_all'] / att['tok_total'] if att['tok_total'] else 0
            pct_typ_d = 100 * att['typ_dialect'] / att['typ_total'] if att['typ_total'] else 0
            pct_typ_a = 100 * att['typ_all'] / att['typ_total'] if att['typ_total'] else 0
            print(f"\n  Against {label}:")
            print(f"    Dialect tokens: {att['tok_dialect']}/{att['tok_total']} "
                  f"({pct_tok_d:.1f}%)")
            print(f"    Dialect types:  {att['typ_dialect']}/{att['typ_total']} "
                  f"({pct_typ_d:.1f}%)")
            print(f"    All-MS tokens:  {att['tok_all']}/{att['tok_total']} "
                  f"({pct_tok_a:.1f}%)")
            print(f"    All-MS types:   {att['typ_all']}/{att['typ_total']} "
                  f"({pct_typ_a:.1f}%)")
            if att['unattested_sample']:
                print(f"    Unattested:     {', '.join(att['unattested_sample'][:15])}")

    # Signatures
    print(f"\n  FOUR-SIGNATURE EVALUATION")
    print(f"  {'─'*55}")

    print(f"\n  Sig1  E->S:      {gen_agg['es_pct_mean']:.1f}%  "
          f"[{gen_agg['es_pct_lo']:.1f}, {gen_agg['es_pct_hi']:.1f}]")
    print(f"        VMS ref:   {ref['es_pct']:.1f}%  "
          f"[{ref['es_pct_lo']:.1f}, {ref['es_pct_hi']:.1f}]")
    print(f"        Criterion: {reasons['E->S']}")

    # Sig2 — full corpus
    ref_bilat_full = ref.get('bilat_full', 0)
    print(f"\n  Sig2  Bilateral (full corpus):")
    print(f"        Generator: Se={gen_se_full}, Ee={gen_ee_full}  "
          f"→ {'YES' if gen_bilat_full else 'NO'}")
    print(f"        VMS ref:   Se={ref.get('n_se_full', '?')}, "
          f"Ee={ref.get('n_ee_full', '?')}  "
          f"→ {'YES' if ref_bilat_full else 'NO'}")
    print(f"        Criterion: {reasons['Bilat']}")
    # Also show chunk-based for context
    print(f"        (chunks:   gen={gen_agg['bilat_frac']*100:.0f}%  "
          f"ref={ref.get('bilat_frac', 0)*100:.0f}%)")

    # Sig3
    print(f"\n  Sig3  MI:        {gen_agg['mi_orig_mean']:.4f}  "
          f"[{gen_agg['mi_orig_lo']:.4f}, {gen_agg['mi_orig_hi']:.4f}]")
    print(f"        VMS ref:   {ref['mi_orig']:.4f}  "
          f"[{ref['mi_orig_lo']:.4f}, {ref['mi_orig_hi']:.4f}]")
    print(f"        Criterion: {reasons['MI']}")
    mi_ratio = gen_agg['mi_orig_mean'] / ref['mi_orig'] if ref['mi_orig'] > 0 else 0
    if 0.8 <= mi_ratio <= 1.2:
        mi_quality = "excellent match"
    elif 0.5 <= mi_ratio <= 2.0:
        mi_quality = "acceptable range"
    else:
        mi_quality = f"{'overshoot' if mi_ratio > 1 else 'undershoot'} ({mi_ratio:.1f}x)"
    print(f"        Quality:   {mi_quality} ({mi_ratio:.2f}x reference)")

    # Sig4
    print(f"\n  Sig4  Shape:     {gen_agg['shape']}  "
          f"R²={gen_agg.get('zipf_r2_mean', 0):.3f}  "
          f"CV={gen_agg.get('cv_mean', 0):.2f}")
    print(f"        VMS ref:   {ref['shape']}  "
          f"R²={ref['zipf_r2']:.3f}  CV={ref['cv']:.2f}")
    print(f"        Criterion: {reasons['Shape']}")

    # Verdict
    print(f"\n  {'─'*55}")
    print(f"  VERDICT (calibrated to Currier {language}):")
    print(f"  {'─'*55}")
    for sig in ['E->S', 'Bilat', 'MI', 'Shape']:
        passed = checks[sig]
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {sig:12s}: {status:10s}  [{reasons[sig]}]")
    print(f"  {'─'*55}")
    print(f"    Joint (adaptive):   {n_pass}/4")

    print(f"\n  Reference values:")
    print(f"    VMS Currier {language}: "
          f"E->S={ref['es_pct']:.1f}%  MI={ref['mi_orig']:.4f}  "
          f"Bilateral={'YES' if ref_bilat_full else 'NO'}  {ref['shape']}")

    if n_pass == 4:
        print(f"\n  ★★★ ALL 4 SIGNATURES PASSED (Currier {language}) ★★★")
    elif n_pass == 3:
        print(f"\n  ★★ 3/4 SIGNATURES PASSED")
    elif n_pass >= 2:
        print(f"\n  ★ {n_pass}/4 signatures passed")
    else:
        print(f"\n  {n_pass}/4 signatures passed")

    # Diagnostics
    print(f"\n  {'─'*55}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"  {'─'*55}")
    diagnostics = []
    if not checks['E->S']:
        diagnostics.append(
            f"  E->S: gen {gen_agg['es_pct_mean']:.1f}% vs ref {ref['es_pct']:.1f}%")
    if not checks['Bilat']:
        diagnostics.append(
            f"  Bilat: gen Se={gen_se_full},Ee={gen_ee_full} "
            f"vs ref Se={ref.get('n_se_full','?')},Ee={ref.get('n_ee_full','?')}")
    if not checks['MI']:
        diagnostics.append(
            f"  MI: gen {gen_agg['mi_orig_mean']:.4f} vs ref {ref['mi_orig']:.4f} "
            f"({mi_quality})")
    if not checks['Shape']:
        diagnostics.append(
            f"  Shape: gen {gen_agg['shape']} vs ref {ref['shape']}")
    if diagnostics:
        print(f"  Failures:")
        for d in diagnostics:
            print(f"    {d}")
    else:
        print(f"  All signatures match VMS Currier {language} reference.")

    print(f"{'='*W}\n")
    return checks, n_pass


# ================================================================
# MAIN
# ================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Verify generated Voynich text against 4-signature criterion')
    ap.add_argument('--words', type=str, default=None)
    ap.add_argument('--language', type=str, required=True,
                    choices=['A', 'B'])
    ap.add_argument('--manuscript', type=str, default=MANUSCRIPT_FILE)
    ap.add_argument('--manuscript2', type=str, default=None)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--n-shuffles', type=int, default=None)
    ap.add_argument('--n-chunks', type=int, default=10,
                    help='Chunks for CI metrics (default: 10)')
    ap.add_argument('--calibrate-only', action='store_true')
    ap.add_argument('--force-calibrate', action='store_true')
    args = ap.parse_args()

    n_shuffles = args.n_shuffles or (3 if args.fast else 10)

    print(f"\n{'#'*70}")
    print(f"#  VOYNICH GENERATOR VERIFICATION")
    print(f"#  Language: Currier {args.language}")
    print(f"#  MI shuffles: {n_shuffles}")
    print(f"#  CI chunks: {args.n_chunks}  (bilateral: full corpus)")
    print(f"#  Thresholds: adaptive (calibrated per dialect)")
    print(f"{'#'*70}")

    # Calibration
    print(f"\n  Calibration...")
    cal = load_or_calibrate(
        args.manuscript, args.n_chunks, n_shuffles,
        force=args.force_calibrate)

    if args.language not in cal:
        sys.exit(f"ERROR: No calibration for Currier {args.language}")

    ref = cal[args.language]
    print(f"\n  VMS Currier {args.language} reference:")
    print(f"    E->S:      {ref['es_pct']:.1f}%")
    print(f"    MI:        {ref['mi_orig']:.4f}")
    print(f"    Bilateral: {'YES' if ref.get('bilat_full', 0) else 'NO'} "
          f"(Se={ref.get('n_se_full', '?')}, Ee={ref.get('n_ee_full', '?')})")
    print(f"    Shape:     {ref['shape']} "
          f"(R²={ref['zipf_r2']:.3f}, CV={ref['cv']:.2f})")
    print(f"    Words:     {ref['n_words']}")

    if args.calibrate_only:
        print(f"\n  Calibration complete.")
        return

    if args.words is None:
        sys.exit("ERROR: --words required (unless --calibrate-only)")

    # Load generated text
    print(f"\n  Loading: {args.words}")
    gen_words, file_sentences = load_generated_words(args.words)
    print(f"    Words: {len(gen_words)}")
    print(f"    Unique: {len(set(gen_words))}")
    print(f"    Sample: {' '.join(gen_words[:20])}")

    if file_sentences:
        sentences = file_sentences
        print(f"    Sentences (from lines): {len(sentences)}")
    else:
        sentences = words_to_sentences(gen_words)
        print(f"    Sentences (auto): {len(sentences)}")

    n_words_total = sum(len(s) for s in sentences)

    # Attestation
    print(f"\n  Checking attestation...")
    attestation = check_attestation(
        gen_words, args.manuscript, args.language,
        manuscript2=args.manuscript2)

    # Evaluation: chunks for CI metrics
    print(f"\n  Running CI evaluation ({args.n_chunks} chunks)...")
    results_ci = evaluate_chunked(sentences, args.n_chunks, n_shuffles,
                                   f"Gen-{args.language}")
    for i, r in enumerate(results_ci):
        if i % max(1, len(results_ci)//5) == 0 or i == len(results_ci) - 1:
            print(f"    Chunk {i:2d}/{len(results_ci)}: "
                  f"E->S={r['es_pct']:.1f}%  MI={r['mi_orig']:.4f}  "
                  f"Shape={r['shape']}")

    if not results_ci:
        sys.exit("ERROR: no chunks large enough.")

    gen_agg = ev.aggregate(
        results_ci,
        label=f"Generator ({os.path.basename(args.words)})")
    gen_agg['n_words'] = n_words_total
    gen_agg['n_sents'] = len(sentences)
    gen_agg['n_runs'] = len(results_ci)

    # Evaluation: full corpus for bilateral
    print(f"\n  Running full-corpus bilateral evaluation...")
    full_result = evaluate_full(sentences, n_shuffles,
                                 f"Gen-{args.language}-full")
    if full_result:
        gen_se_full = full_result['n_start_extreme']
        gen_ee_full = full_result['n_end_extreme']
        gen_bilat_full = 1 if (gen_se_full > 0 and gen_ee_full > 0) else 0
        print(f"    Se={gen_se_full}, Ee={gen_ee_full}  "
              f"→ Bilateral={'YES' if gen_bilat_full else 'NO'}")
    else:
        gen_se_full = gen_ee_full = 0
        gen_bilat_full = 0

    # Report
    print_report(gen_agg, gen_bilat_full, gen_se_full, gen_ee_full,
                 cal, args.language, args.words, attestation)


if __name__ == '__main__':
    main()
