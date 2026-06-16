Support material for https://arxiv.org/pdf/2604.19762

## calibration (need RF1b-e.txt in local dir)

python3 verifier.py --calibrate-only --language A

  Calibration...
  Loading cached calibration from vms_calibration.json

  VMS Currier A reference:
    E->S:      71.0%
    MI:        0.5856
    Bilateral: YES (Se=3, Ee=3)
    Shape:     Zipfian (R²=0.863, CV=1.45)
    Words:     9892

  Calibration complete.

  python3 verifier.py --calibrate-only --language B

    Calibration...
  Loading cached calibration from vms_calibration.json

  VMS Currier B reference:
    E->S:      64.3%
    MI:        0.4980
    Bilateral: YES (Se=3, Ee=7)
    Shape:     Intermediate (R²=0.805, CV=1.53)
    Words:     19640

  Calibration complete.


## verification (source: any space separated source file, here we use words.txt)

python3 verifier.py --words words.txt --language A


VOYNICH GENERATOR VERIFICATION
Language: Currier A
MI shuffles: 10
CI chunks: 10  (bilateral: full corpus)
Thresholds: adaptive (calibrated per dialect)

  Calibration...
  Loading cached calibration from vms_calibration.json

  VMS Currier A reference:
    E->S:      71.0%
    MI:        0.5856
    Bilateral: YES (Se=3, Ee=3)
    Shape:     Zipfian (R²=0.863, CV=1.45)
    Words:     9892

  Loading: words.txt
    Words: 10000
    Unique: 2135
    Sample: lchdy qokol olo chekcheor chol tchor chol qoaiin qoteol cho dy tchory chopchal chody tos kcheey ainy chaiin ydar cho
    Sentences (from lines): 1521

  Checking attestation...

  Running CI evaluation (10 chunks)...
    Chunk  0/10: E->S=66.1%  MI=0.4728  Shape=Intermediate
    Chunk  2/10: E->S=69.1%  MI=0.4610  Shape=Zipfian
    Chunk  4/10: E->S=65.6%  MI=0.4651  Shape=Zipfian
    Chunk  6/10: E->S=66.6%  MI=0.4253  Shape=Zipfian
    Chunk  8/10: E->S=66.0%  MI=0.4714  Shape=Zipfian
    Chunk  9/10: E->S=64.7%  MI=0.4954  Shape=Intermediate

  Running full-corpus bilateral evaluation...
    Se=3, Ee=2  → Bilateral=YES

  GENERATOR VERIFICATION — words.txt
  Language: Currier A
  10 CI chunks, 9806 words, 1521 sentences

  WORD ATTESTATION
  ───────────────────────────────────────────────────────

  Against chclass.txt:
    Dialect tokens: 9405/10000 (94.0%)
    Dialect types:  1819/2135 (85.2%)
    All-MS tokens:  9506/10000 (95.1%)
    All-MS types:   1861/2135 (87.2%)
    Unattested:     alfchy, apchey, chaikhy, charal, chcheaiin, chctych, chdqoty, cheas, chee, cheekal, cheekchody, cheekol, cheetchy, chekcheor, chekeeschy

  FOUR-SIGNATURE EVALUATION
  ───────────────────────────────────────────────────────

  Sig1  E->S:      66.4%  [65.4, 67.8]
        VMS ref:   71.0%  [67.2, 74.8]
        Criterion: range [56, 86]%

  Sig2  Bilateral (full corpus):
        Generator: Se=3, Ee=2  → YES
        VMS ref:   Se=3, Ee=3  → YES
        Criterion: must have bilateral (ref has it)
        (chunks:   gen=0%  ref=10%)

  Sig3  MI:        0.4631  [0.4503, 0.4748]
        VMS ref:   0.5856  [0.5413, 0.6315]
        Criterion: range [0.293, 1.171]
        Quality:   acceptable range (0.79x reference)

  Sig4  Shape:     Zipfian  R²=0.856  CV=1.65
        VMS ref:   Zipfian  R²=0.863  CV=1.45
        Criterion: non-Plateau (ref=Zipfian)

  ───────────────────────────────────────────────────────
  VERDICT (calibrated to Currier A):
  ───────────────────────────────────────────────────────
    E->S        : ✓ PASS      [range [56, 86]%]
    Bilat       : ✓ PASS      [must have bilateral (ref has it)]
    MI          : ✓ PASS      [range [0.293, 1.171]]
    Shape       : ✓ PASS      [non-Plateau (ref=Zipfian)]
  ───────────────────────────────────────────────────────
    Joint (adaptive):   4/4

  Reference values:
    VMS Currier A: E->S=71.0%  MI=0.5856  Bilateral=YES  Zipfian

  ★★★ ALL 4 SIGNATURES PASSED (Currier A) ★★★


## signatures (support material for the paper)
Note: each run can take up to 45 minutes, the results are uploaded so you dont have to run the sigs yourself.

python3 signatures_v27.py --language A --manuscript RF1b-e.txt --fast > signatures_Currier_A.txt
python3 signatures_v27.py --language B --manuscript RF1b-e.txt --fast > signatures_Currier_B.txt
