"""
gate_sweep_fixedpoint.py

Q8.8 fixed-point version of the gate validation.

*** READ THIS BEFORE TRUSTING THE OUTPUT ***
If you already have a bit-exact Q8.8 simulation harness from earlier in
this project, use THAT instead of this script, or cross-check this
script's numbers against it line by line. A second, independently
written fixed-point simulator is exactly how two "same" numbers end up
silently disagreeing -- this project already has one unresolved case of
that (26% vs 1.8% pruned). This script is a faithful-effort Q8.8
implementation following the conventions below. It is NOT verified
against softmax_pipe_deep2.v. If any assumption below doesn't match the
RTL, these numbers will be wrong in a way that looks right.

ASSUMPTIONS -- verify each against your RTL before trusting results:
  - Q8.8 = 16-bit signed, 8 integer bits + 8 fractional bits.
    Resolution = 1/256 = 0.00390625. Range = [-128.0, 127.99609375].
  - Rounding: round-to-nearest on quantization, saturate (clamp) on
    overflow -- not wraparound.
  - Attention scores are quantized to Q8.8 before any softmax logic
    runs (the format named in your paper title).
  - GATE and the eta-derived prune threshold are also quantized to
    Q8.8 before comparison, so the compare itself is bit-exact, not
    float-vs-fixed.
  - The exp() LUT index is computed from the Q8.8-quantized (s - m)
    value, matching your idx>>3 indexing scheme.
  - LUT *output* values are stored at a WIDER fixed precision than
    Q8.8 by default (--lut-frac-bits, default 16). Q8.8's 1/256
    resolution is too coarse to represent small exp() outputs like
    exp(-8)=0.000335 meaningfully -- CONFIRM your RTL's actual LUT
    output width and set --lut-frac-bits to match.
  - The sum accumulator uses --acc-frac-bits (default = lut-frac-bits)
    to avoid overflow across up to 8 summed terms. Confirm against RTL.
  - Final division (normalization) is computed at full float precision
    on the fixed-point accumulator values -- NOT bit-modeled as a
    specific hardware divider or reciprocal LUT. If your RTL uses one,
    that is a source of additional error THIS SCRIPT DOES NOT CAPTURE.

Usage:
    python gate_sweep_fixedpoint.py --data attn_data.npz --gates 6 8 12
"""
import argparse
import numpy as np
import torch

Q_INT_BITS = 8
Q_FRAC_BITS = 8
Q_SCALE = 1 << Q_FRAC_BITS
Q_MIN = -(1 << (Q_INT_BITS + Q_FRAC_BITS - 1))
Q_MAX = (1 << (Q_INT_BITS + Q_FRAC_BITS - 1)) - 1


def to_q88(x):
    raw = np.round(np.asarray(x, dtype=np.float64) * Q_SCALE)
    return np.clip(raw, Q_MIN, Q_MAX).astype(np.int64)

def from_q88(raw):
    return raw.astype(np.float64) / Q_SCALE

def true_sm(s):
    return torch.softmax(torch.as_tensor(s, dtype=torch.float64), dim=-1).numpy()

def kl(p, q, e=1e-9):
    p = p + e
    q = q + e
    return float(np.sum(p * np.log(p / q)))

def build_exp_lut(lut_frac_bits, lo=-8.0, hi=0.0, n=256):
    vals = np.exp(np.linspace(lo, hi, n))
    scale = 1 << lut_frac_bits
    raw = np.clip(np.round(vals * scale), 0, (1 << 40) - 1).astype(np.int64)
    return raw, scale

def lut_index(x_q88_raw, lo=-8.0, hi=0.0, n=256):
    x = from_q88(x_q88_raw)
    idx = np.floor((x - lo) * n / (hi - lo)).astype(np.int64)
    return np.clip(idx, 0, n - 1)

def sg_fixed(s_float, gate_float, lut_raw, lut_scale, acc_frac_bits, eta=0.5):
    s_q = to_q88(s_float)
    m_q, mn_q = int(s_q.max()), int(s_q.min())
    gate_q = int(to_q88(gate_float))
    spread_q = m_q - mn_q

    if spread_q < gate_q:
        keep = np.ones(len(s_q), dtype=bool)
    else:
        spread_f = spread_q / Q_SCALE
        thresh_f = (m_q / Q_SCALE) - eta * spread_f
        thresh_q = int(to_q88(thresh_f))
        keep = s_q >= thresh_q

    shifted_q = np.clip(s_q - m_q, int(to_q88(-8.0)), int(to_q88(0.0)))
    idx = lut_index(shifted_q)
    e_raw = lut_raw[idx].astype(np.int64) * keep.astype(np.int64)

    acc_scale = 1 << acc_frac_bits
    e_acc = np.round(e_raw.astype(np.float64) / lut_scale * acc_scale).astype(np.int64)
    total = e_acc.sum()
    out = np.zeros(len(s_q)) if total == 0 else e_acc.astype(np.float64) / total
    return out, int(keep.sum())

def per_window_kl_fixed(windows, gate, lut_raw, lut_scale, acc_frac_bits):
    ks, ps = [], []
    for w in windows:
        w = np.asarray(w, dtype=np.float64)
        if len(w) < 2:
            ks.append(0.0); ps.append(0.0)
            continue
        o, k = sg_fixed(w, gate, lut_raw, lut_scale, acc_frac_bits)
        ks.append(kl(true_sm(w), o))
        ps.append(1 - k / len(w))
    return np.array(ks), np.array(ps)

def sentence_clustered_bootstrap(ks_a, ks_b, sentence_ids, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sentence_ids)
    diffs = ks_a - ks_b
    boot_means = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        vals = [diffs[sentence_ids == sid].mean() for sid in picked]
        boot_means.append(np.mean(vals))
    boot_means = np.array(boot_means)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return dict(n_clusters=len(uniq), mean_diff=float(diffs.mean()),
                ci95=(float(ci_lo), float(ci_hi)),
                p_a_better=float(np.mean(boot_means < 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gates", type=float, nargs="+", default=[6.0, 8.0, 12.0])
    ap.add_argument("--lut-frac-bits", type=int, default=16,
                     help="fractional bits for LUT exp() output storage (default 16 -- CHANGE to match RTL)")
    ap.add_argument("--acc-frac-bits", type=int, default=16,
                     help="fractional bits for the sum accumulator (default = lut-frac-bits)")
    args = ap.parse_args()

    print("*** Q8.8 fixed-point simulation -- verify assumptions in the file")
    print("*** docstring against softmax_pipe_deep2.v before trusting these numbers.\n")

    d = np.load(args.data, allow_pickle=True)
    windows, sentence_ids = d["windows"], d["sentence_ids"]
    categories = d["categories"] if "categories" in d else None
    n_sentences = len(np.unique(sentence_ids))
    print(f"Loaded {len(windows)} windows from {n_sentences} sentences")
    print(f"Q8.8: resolution=1/{Q_SCALE}={1/Q_SCALE:.6f}, range=[{Q_MIN/Q_SCALE:.3f}, {Q_MAX/Q_SCALE:.3f}]")
    print(f"LUT output precision: {args.lut_frac_bits} frac bits; accumulator: {args.acc_frac_bits} frac bits\n")

    lut_raw, lut_scale = build_exp_lut(args.lut_frac_bits)

    results, pruned = {}, {}
    print(f"{'gate':>6} | {'mean_KL':>9} | {'p95_KL':>9} | {'max_KL':>9} | {'pruned%':>8}")
    for gate in args.gates:
        ks, ps = per_window_kl_fixed(windows, gate, lut_raw, lut_scale, args.acc_frac_bits)
        results[gate] = ks
        pruned[gate] = ps
        print(f"{gate:>6} | {ks.mean():9.4f} | {np.percentile(ks,95):9.4f} | "
              f"{ks.max():9.4f} | {ps.mean()*100:7.1f}%")

    if categories is not None:
        print("\n=== Per-category breakdown (fixed-point) ===")
        for gate in args.gates:
            print(f"\n-- gate={gate} --")
            for cat in sorted(set(categories)):
                mask = categories == cat
                print(f"   {cat:>18} | mean_KL={results[gate][mask].mean():.4f} | "
                      f"max_KL={results[gate][mask].max():.4f}")

    print("\n=== Sentence-clustered significance test (fixed-point) ===")
    best = min(args.gates, key=lambda g: results[g].mean())
    for g in args.gates:
        if g == best:
            continue
        t = sentence_clustered_bootstrap(results[best], results[g], sentence_ids)
        print(f"gate={best} vs gate={g}: n_clusters={t['n_clusters']}  "
              f"mean_diff={t['mean_diff']:.4f}  95% CI={t['ci95']}  "
              f"P(gate={best} better)={t['p_a_better']:.3f}")

    print("\nCompare mean_KL above to your earlier FLOATING-POINT run (gate_sweep_final.py).")
    print("A large gap between the two means quantization is adding real error your")
    print("floating-point validation never saw -- report BOTH numbers, not just one.")


if __name__ == "__main__":
    main()
