"""
Idea B: Numerically-stable, max-aware fused SoftMax for SWAT-style accelerators.

This single file runs the full before/after demonstration:
  B1 - show SWAT's fused SoftMax (NO max-subtraction) is unstable
  B2 - implement the max-subtracted fix
  B3 - re-run the same sweep and show the error collapses

Run:  python3 idea_B_softmax_stability.py
"""

import numpy as np
import torch


# ----------------------------------------------------------------------
# Shared helper: build an exp() lookup table over a given range.
# This mimics how SWAT implements exp() in hardware (Script 4 style):
# a fixed-size table, NOT a real exp() unit. Anything outside the
# table's range gets clipped to the nearest edge entry.
# ----------------------------------------------------------------------
def exp_via_lut(values, lut_lo, lut_hi, n_entries=256):
    """Approximate exp(values) using a fixed LUT over [lut_lo, lut_hi]."""
    v = values.numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    # map each value to a table index, CLIPPING anything out of range
    idx = ((v - lut_lo) / (lut_hi - lut_lo) * (n_entries - 1)).astype(int)
    idx = np.clip(idx, 0, n_entries - 1)          # <-- clipping = the bug source
    table_inputs = np.linspace(lut_lo, lut_hi, n_entries)
    exp_table = np.exp(table_inputs)
    return exp_table[idx]


# ----------------------------------------------------------------------
# B1: SWAT's SoftMax  -- NO max subtraction, LUT range (-4, +4)
# ----------------------------------------------------------------------
def swat_softmax_no_maxsub(scores, lut_lo=-4, lut_hi=4):
    e = exp_via_lut(scores, lut_lo, lut_hi)       # exp via LUT, no shift
    return e / e.sum()


# ----------------------------------------------------------------------
# B2: The fix -- subtract the running max FIRST, then exp via LUT.
# After subtracting the max, every score is <= 0, so the exp input
# is always in (-inf, 0]. The LUT only needs to cover (-8, 0] and
# NOTHING ever clips on the high side. This is the standard
# numerically-stable SoftMax, which SWAT's Eq.1 omits.
# ----------------------------------------------------------------------
def stable_lut_softmax(scores, lut_lo=-8, lut_hi=0):
    m = scores.max()                              # the running max (one comparator)
    shifted = scores - m                          # now all <= 0  -> never clips high
    e = exp_via_lut(shifted, lut_lo, lut_hi)
    return e / e.sum()


# ----------------------------------------------------------------------
# Ground truth: the mathematically correct SoftMax (full precision).
# ----------------------------------------------------------------------
def true_softmax(scores):
    return torch.softmax(scores, dim=-1).numpy()


# ----------------------------------------------------------------------
# The experiment: sweep score magnitude from +-1 up to +-16
# (+-16 is the realistic Q.K range you measured in Script 2).
# For each scale, measure how wrong each method is vs. true SoftMax.
# ----------------------------------------------------------------------
def run_sweep(seed=0, n=64, scales=(1, 3, 6, 10, 16), trials=200):
    rng = torch.Generator().manual_seed(seed)
    print(f"{'scale':>7} | {'% clipped':>10} | "
          f"{'SWAT err (no maxsub)':>22} | {'stable err (fix)':>18}")
    print("-" * 70)
    for scale in scales:
        swat_errs, stable_errs, clip_fracs = [], [], []
        for _ in range(trials):
            s = torch.randn(n, generator=rng) * scale
            ref = true_softmax(s)

            swat = swat_softmax_no_maxsub(s)
            stable = stable_lut_softmax(s)

            swat_errs.append(np.mean(np.abs(ref - swat)))
            stable_errs.append(np.mean(np.abs(ref - stable)))
            clip_fracs.append(float(((s < -4) | (s > 4)).float().mean()))

        print(f"{'+-'+str(scale):>7} | "
              f"{np.mean(clip_fracs)*100:9.0f}% | "
              f"{np.mean(swat_errs)*100:21.2f}% | "
              f"{np.mean(stable_errs)*100:17.4f}%")


# ----------------------------------------------------------------------
# Optional: show the OVERFLOW failure that pure error % hides.
# In true low precision, exp(big) doesn't just clip -- it overflows.
# ----------------------------------------------------------------------
def overflow_demo():
    print("\nOverflow check (what happens with a real exp, low precision):")
    for scale in (4, 16, 30):
        s = torch.randn(64) * scale
        # simulate fp16 exp WITHOUT max subtraction
        raw = torch.exp(s.half())
        has_inf = torch.isinf(raw).any().item()
        # WITH max subtraction
        safe = torch.exp((s - s.max()).half())
        safe_inf = torch.isinf(safe).any().item()
        print(f"  scale +-{scale:2d}:  no-maxsub overflow(inf)? {has_inf:<5}  "
              f"|  with-maxsub overflow? {safe_inf}")


if __name__ == "__main__":
    print("=" * 70)
    print("IDEA B: SoftMax stability  --  before (SWAT) vs after (fix)")
    print("=" * 70)
    run_sweep()
    overflow_demo()
    print("\nReading the table:")
    print("  * SWAT column: error jumps once scores leave the (-4,+4) LUT range.")
    print("  * stable column: error stays tiny at EVERY scale, incl. +-16.")
    print("  * 'fix' = subtract running max before exp  (SWAT's Eq.1 omits this).")
    
def stable_softmax_with_pruning(scores, eta=0.5):
    m = scores.max(); mn = scores.min()
    threshold = m - eta*(m - mn)          # LAPA's DDF, Eq.5
    keep = scores >= threshold            # prune weak tokens
    # stable softmax only over kept tokens
    kept = scores[keep] - m
    e = torch.exp(kept)
    out = torch.zeros_like(scores)
    out[keep] = e/e.sum()
    return out, keep.sum().item()         # also return how many survived
