"""
08_sweep_gate_threshold_g.py -- sweeps SG-DDF's gate threshold g on the
CORRECTED (8-wide chunked) engine, to find whether any g reopens the gate
usefully now that spread is computed per-8-element-chunk instead of over
the whole row.

WHY THIS EXISTS
  softmax_engine_CURRENT.py's default gate (g=12.0, i.e. GATE_Q=3072 in
  Q8.8) was tuned against the OLD, buggy full-row spread computation.
  Chunk-local spread at L=8 is much smaller than full-row spread at
  L=100+ (see the earlier extreme-value-statistics estimate: mean
  spread ~15 at L=100 vs ~8.5 at L=8), so g=12 barely ever opens the
  gate on real 8-wide chunks -- confirmed empirically: SG-DDF's cut-n
  collapsed from 34.4% (old, wrong) to 0.9% (new, correct) on a real
  GPT-2 Medium / WikiText-103 run.

  This script sweeps g and reports, for each value:
    - SG-DDF's cut-n% (how often the gate actually opens)
    - SG-DDF's PPL delta vs exact baseline (does it stay safe?)
    - next-token agreement vs baseline
  alongside Stable (g=infinity, gate never opens, the safety floor) and
  DDF (g=0, gate always open, the pruning ceiling) as fixed reference
  rows, so you can see exactly where each g lands between those two
  bounds.

USAGE
  python3 08_sweep_gate_threshold_g.py --model gpt2-medium --dataset wikitext103 --limit_tokens 20000
  python3 08_sweep_gate_threshold_g.py --model qwen --dataset c4 --limit_tokens 20000 --gates 1,2,3,4,6,8,12

NOTE ON COST: this runs one full perplexity pass per gate value tested,
on top of the baseline and the two reference rows. Keep --limit_tokens
modest (10-20k) for the sweep itself; once you've picked a promising g,
re-run it at full --limit_tokens through 01_validate_llm_perplexity_TABLE4.py
for the number that actually goes in the paper.
"""
import argparse, csv, math, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from softmax_engine_CURRENT import HWSoftmax

MODEL_PRESETS = {
    'gpt2-medium': dict(model_id='gpt2-medium', gated=False),
    'llama':       dict(model_id='unsloth/Llama-3.2-1B', gated=False),
    'qwen':        dict(model_id='Qwen/Qwen2.5-0.5B', gated=False),
}

FALLBACK_TEXT = (
    "The history of natural language processing generally started in the 1950s, "
    "although work can be found from earlier periods. In 1950, Alan Turing published "
    "an article titled Computing Machinery and Intelligence which proposed what is now "
    "called the Turing test as a criterion of intelligence. The Georgetown experiment "
    "in 1954 involved fully automatic translation of more than sixty Russian sentences "
    "into English. Real progress was much slower than predicted, and funding for "
    "machine translation was dramatically reduced after the ALPAC report in 1966."
) * 60


def load_text(dataset, limit_tokens, local_file=None):
    if local_file:
        return open(local_file, encoding='utf-8').read()
    try:
        from datasets import load_dataset
        name, subset = {
            'wikitext103': ('wikitext', 'wikitext-103-raw-v1'),
            'wikitext2':   ('wikitext', 'wikitext-2-raw-v1'),
            'c4':          ('allenai/c4', 'en'),
        }[dataset]
        if dataset == 'c4':
            ds = load_dataset(name, subset, split='validation', streaming=True)
            texts, n = [], 0
            for ex in ds:
                texts.append(ex['text']); n += len(ex['text'])
                if n > limit_tokens * 6:
                    break
            return "\n\n".join(texts)
        ds = load_dataset(name, subset, split='validation')
        return "\n\n".join(ds['text'])[:limit_tokens * 6]
    except Exception as e:
        print(f"[warn] dataset load failed ({e}); using built-in fallback -- "
              "fine for a smoke test, not for a reported number")
        return FALLBACK_TEXT


@torch.no_grad()
def perplexity_and_next_token(model, tok, text, device, limit_tokens, max_len=512,
                               baseline_next_tok=None):
    ids = tok(text, return_tensors="pt").input_ids
    if limit_tokens:
        ids = ids[:, :limit_tokens]
    ids = ids.to(device)
    nll, cnt, next_preds = 0.0, 0, []
    for i in range(0, ids.size(1) - 1, max_len):
        c = ids[:, i:i + max_len]
        if c.size(1) < 2:
            continue
        out = model(c, labels=c)
        if torch.isnan(out.loss) or torch.isinf(out.loss):
            return float('nan'), []
        nll += out.loss.item() * (c.size(1) - 1)
        cnt += c.size(1) - 1
        next_preds.append(int(out.logits[0, -1].argmax()))
    ppl = math.exp(nll / cnt) if cnt else float('nan')
    agree = None
    if baseline_next_tok is not None:
        agree = 100.0 * sum(a == b for a, b in zip(next_preds, baseline_next_tok)) / max(len(next_preds), 1)
    return ppl, next_preds, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_PRESETS))
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--dataset", default="wikitext103",
                    choices=["wikitext103", "wikitext2", "c4"])
    ap.add_argument("--wikitext_file", default=None)
    ap.add_argument("--limit_tokens", type=int, default=20_000)
    ap.add_argument("--gates", default="1,2,3,4,6,8,12",
                    help="comma-separated g values in REAL units (Q8.8 conversion "
                         "done internally: g_q88 = g_real * 256)")
    ap.add_argument("--module", default="mul255", choices=["mul255", "shift3", "soft"])
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--out", default="gate_sweep_results.csv")
    a = ap.parse_args()
    device = "cuda" if (a.cuda and torch.cuda.is_available()) else "cpu"

    preset = MODEL_PRESETS[a.model]
    model_id = a.model_id or preset['model_id']
    gates_real = [float(x) for x in a.gates.split(",")]

    print(f"\n{'='*76}\n{a.model} ({model_id}) on {a.dataset}  --  gate sweep: {gates_real}\n{'='*76}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager", torch_dtype=torch.float32
    ).to(device).eval()

    text = load_text(a.dataset, a.limit_tokens, a.wikitext_file)

    orig = F.softmax
    hits = [0]
    def probe(x, dim=-1, dtype=None, **kw):
        hits[0] += 1
        return orig(x, dim=dim, dtype=dtype) if dtype else orig(x, dim=dim)
    F.softmax = probe
    with torch.no_grad():
        model(tok(text[:500], return_tensors="pt").input_ids.to(device))
    F.softmax = orig
    assert hits[0] > 0, "softmax patch never fired -- not using eager attention"
    print(f"[info] patch verified: {hits[0]} attention softmax calls in a 500-token probe\n")

    t0 = time.time()
    F.softmax = orig
    base_ppl, base_next, _ = perplexity_and_next_token(model, tok, text, device, a.limit_tokens)
    print(f"BASELINE (exact fp32 softmax): PPL = {base_ppl:.3f}   [{time.time()-t0:.0f}s]\n")

    hdr = f"{'g (real)':>9}|{'g (Q8.8)':>9}|{'PPL':>10}|{'delta':>9}|{'next-tok agree%':>16}|{'cut-n%':>7}|{'dead%':>6}"
    print(hdr); print("-" * len(hdr))
    rows = [dict(g_real='baseline', g_q88='-', ppl=base_ppl, d_ppl=0.0,
                 next_tok_agree_pct=100.0, cut_n_pct=0.0, dead_pct=0.0)]

    # Reference floor: Stable (gate effectively infinite, never opens)
    hw = HWSoftmax(a.module, 1, device=device, gate=1 << 30)
    F.softmax = hw
    ppl, _, agree = perplexity_and_next_token(model, tok, text, device, a.limit_tokens,
                                              baseline_next_tok=base_next)
    F.softmax = orig
    st = hw.stats()
    print(f"{'Stable':>9}|{'(floor)':>9}|{ppl:10.3f}|{ppl-base_ppl:+9.3f}|"
          f"{agree:16.2f}|{st['cut_n_pct']:7.1f}|{st['dead_pct']:6.1f}")
    rows.append(dict(g_real='Stable(floor)', g_q88='-', ppl=ppl, d_ppl=ppl-base_ppl,
                     next_tok_agree_pct=agree, cut_n_pct=st['cut_n_pct'], dead_pct=st['dead_pct']))

    # Reference ceiling: DDF (gate always open, g=0)
    hw = HWSoftmax(a.module, 2, device=device)
    F.softmax = hw
    ppl, _, agree = perplexity_and_next_token(model, tok, text, device, a.limit_tokens,
                                              baseline_next_tok=base_next)
    F.softmax = orig
    st = hw.stats()
    print(f"{'DDF':>9}|{'(ceiling)':>9}|{ppl:10.3f}|{ppl-base_ppl:+9.3f}|"
          f"{agree:16.2f}|{st['cut_n_pct']:7.1f}|{st['dead_pct']:6.1f}")
    rows.append(dict(g_real='DDF(ceiling)', g_q88='-', ppl=ppl, d_ppl=ppl-base_ppl,
                     next_tok_agree_pct=agree, cut_n_pct=st['cut_n_pct'], dead_pct=st['dead_pct']))
    print("-" * len(hdr))

    # The actual sweep: SG-DDF at each g
    for g_real in gates_real:
        g_q88 = int(round(g_real * 256))
        hw = HWSoftmax(a.module, 3, device=device, gate=g_q88)
        F.softmax = hw
        ppl, _, agree = perplexity_and_next_token(model, tok, text, device, a.limit_tokens,
                                                  baseline_next_tok=base_next)
        F.softmax = orig
        st = hw.stats()
        print(f"{g_real:9.1f}|{g_q88:9d}|{ppl:10.3f}|{ppl-base_ppl:+9.3f}|"
              f"{agree:16.2f}|{st['cut_n_pct']:7.1f}|{st['dead_pct']:6.1f}")
        rows.append(dict(g_real=g_real, g_q88=g_q88, ppl=ppl, d_ppl=ppl-base_ppl,
                         next_tok_agree_pct=agree, cut_n_pct=st['cut_n_pct'], dead_pct=st['dead_pct']))
    print("-" * len(hdr))

    cols = ['g_real', 'g_q88', 'ppl', 'd_ppl', 'next_tok_agree_pct', 'cut_n_pct', 'dead_pct']
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")
    print("\nHow to read this: look for the SMALLEST g whose PPL delta and next-token")
    print("agreement stay close to Stable's (the floor row) while cut-n%% is clearly")
    print("above 0 -- that's a candidate replacement for g=12. If PPL delta jumps")
    print("sharply toward DDF's (the ceiling row) before cut-n%% becomes meaningful,")
    print("that is evidence 8-wide chunking may be too fine-grained for this gate")
    print("to find a useful middle ground -- a real, reportable finding either way.")


if __name__ == "__main__":
    main()
