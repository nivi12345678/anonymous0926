"""
validate_llm.py -- validate hardware softmax variants on GPT-2 Medium,
LLaMA, and Qwen, without any fine-tuning.

These are causal language models, not classifiers, so the metric here is
PERPLEXITY on held-out text (matching Fig. 6 of the reference paper /
your own Table III), not accuracy. Reuses softmax_hw.HWSoftmax exactly as
validate_glue.py does: patch nn.functional.softmax at inference time,
never touch the weights.

MODEL PRESETS
  gpt2-medium : "gpt2-medium"                    ungated, 355M, 24 layers
  llama       : "meta-llama/Llama-3.2-1B"         GATED -- see note below
  qwen        : "Qwen/Qwen2.5-0.5B"                ungated

  Override any preset with --model_id to match the exact paper-reported
  size (e.g. --model llama --model_id meta-llama/Llama-3.2-3B).

GATING
  LLaMA checkpoints require accepting the license at the model page and
  either `huggingface-cli login` or an HF_TOKEN env var. If loading fails
  with a 401/403, that's why -- Qwen and GPT-2 need neither.

CRITICAL: attn_implementation="eager"
  Modern Transformers routes attention through a shared eager_attention_
  forward() across GPT-2, LLaMA, and Qwen alike, which calls
  nn.functional.softmax directly -- but ONLY when attn_implementation=
  "eager" is set. The default fused SDPA/flash kernel is invisible to
  Python-level patching, and every "variant" below would silently report
  the exact baseline number. This script asserts the patch actually fires
  before trusting any result; do not remove that assertion.

USAGE
  python3 validate_llm.py --model gpt2-medium --limit_tokens 100000
  python3 validate_llm.py --model qwen        --limit_tokens 100000
  python3 validate_llm.py --model llama       --limit_tokens 100000 --cuda
  python3 validate_llm.py --all --limit_tokens 50000     # sweep all three

  --dataset wikitext103 (default, matches the paper) | wikitext2 | c4
  --wikitext_file /path/to/local.txt   # bypass the Hub entirely
"""
import argparse, csv, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from softmax_hw import HWSoftmax

MODULES = ('mul255', 'shift3', 'soft')
MODES = {0: 'SWAT', 1: 'Stable', 2: 'DDF', 3: 'SG-DDF'}

MODEL_PRESETS = {
    'gpt2-medium': dict(model_id='gpt2-medium', gated=False),
    'llama':       dict(model_id='meta-llama/Llama-3.2-1B', gated=True),
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
) * 40


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
        ds = load_dataset(name, subset, split='validation' if dataset != 'c4' else 'validation',
                          streaming=(dataset == 'c4'))
        if dataset == 'c4':
            texts, n = [], 0
            for ex in ds:
                texts.append(ex['text']); n += len(ex['text'])
                if n > limit_tokens * 6:   # ~6 chars/token, generous margin
                    break
            return "\n\n".join(texts)
        return "\n\n".join(ds['text'])[:limit_tokens * 6]
    except Exception as e:
        print(f"[warn] dataset load failed ({e}); using built-in fallback passage "
              "-- fine for a smoke test, not for a reported number")
        return FALLBACK_TEXT


@torch.no_grad()
def perplexity_and_agreement(model, tok, text, device, limit_tokens, max_len=512,
                              baseline_next_tok=None):
    """Windowed perplexity (matches part3_perplexity_fixed.py), plus optional
    next-token argmax agreement against a supplied baseline prediction list."""
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

    import math
    ppl = math.exp(nll / cnt) if cnt else float('nan')
    agree = None
    if baseline_next_tok is not None:
        agree = 100.0 * sum(a == b for a, b in zip(next_preds, baseline_next_tok)) / max(len(next_preds), 1)
    return ppl, next_preds, agree


def run_one(model_key, model_id, dataset, limit_tokens, local_file, device):
    print(f"\n{'='*70}\n{model_key}  ({model_id})  on {dataset}\n{'='*70}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", torch_dtype=torch.float32
        ).to(device).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()

    text = load_text(dataset, limit_tokens, local_file)

    # ---- verify the patch actually reaches attention ------------------
    orig = F.softmax
    hits = [0]

    def probe(x, dim=-1, dtype=None, **kw):
        hits[0] += 1
        return orig(x, dim=dim, dtype=dtype) if dtype else orig(x, dim=dim)
    F.softmax = probe
    with torch.no_grad():
        ids = tok(text[:500], return_tensors="pt").input_ids.to(device)
        model(ids)
    F.softmax = orig
    assert hits[0] > 0, (
        f"softmax patch never fired for {model_id} -- it is not routing through "
        "eager attention. Every variant below would silently report the baseline "
        "number. Check the transformers version supports attn_implementation="
        "'eager' for this architecture.")
    print(f"[info] patch verified: {hits[0]} attention softmax calls in a 500-token probe\n")

    t0 = time.time()
    base_ppl, base_next, _ = perplexity_and_agreement(model, tok, text, device, limit_tokens)
    print(f"BASELINE (exact fp32 softmax): PPL = {base_ppl:.3f}   [{time.time()-t0:.0f}s]\n")

    hdr = (f"{'module':>7}|{'mode':>7}|{'PPL':>10}|{'delta':>9}|{'next-tok agree%':>16}"
           f"|{'cut-n%':>7}|{'dead%':>6}|{'sumdev%':>8}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for module in MODULES:
        for mode in (0, 1, 2, 3):
            hw = HWSoftmax(module, mode, device=device)
            F.softmax = hw
            ppl, _, agree = perplexity_and_agreement(
                model, tok, text, device, limit_tokens, baseline_next_tok=base_next)
            F.softmax = orig
            st = hw.stats()
            note = "  <-- DIVERGED" if ppl != ppl else ""  # nan check
            print(f"{module:>7}|{MODES[mode]:>7}|{ppl:10.3f}|{ppl-base_ppl:+9.3f}"
                  f"|{agree:16.2f}|{st['cut_n_pct']:7.1f}|{st['dead_pct']:6.1f}"
                  f"|{st['sum_dev_pct']:8.2f}{note}")
            rows.append(dict(model=model_key, dataset=dataset, module=module,
                             mode=MODES[mode], ppl=ppl, d_ppl=ppl - base_ppl,
                             next_tok_agree_pct=agree, cut_n_pct=st['cut_n_pct'],
                             dead_pct=st['dead_pct'], sum_dev_pct=st['sum_dev_pct']))
        print("-" * len(hdr))

    base_row = dict(model=model_key, dataset=dataset, module='-', mode='BASELINE',
                    ppl=base_ppl, d_ppl=0.0, next_tok_agree_pct=100.0,
                    cut_n_pct=0.0, dead_pct=0.0, sum_dev_pct=0.0)
    return [base_row] + rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS))
    ap.add_argument("--model_id", default=None, help="override the preset checkpoint")
    ap.add_argument("--all", action="store_true", help="run gpt2-medium, llama, qwen")
    ap.add_argument("--dataset", default="wikitext103",
                    choices=["wikitext103", "wikitext2", "c4"])
    ap.add_argument("--wikitext_file", default=None, help="local .txt, bypasses the Hub")
    ap.add_argument("--limit_tokens", type=int, default=100_000)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--out", default="llm_softmax_results.csv")
    a = ap.parse_args()
    if not a.all and not a.model:
        sys.exit("pass --model {gpt2-medium,llama,qwen}, or --all")
    device = "cuda" if (a.cuda and torch.cuda.is_available()) else "cpu"

    keys = list(MODEL_PRESETS) if a.all else [a.model]
    all_rows = []
    for k in keys:
        preset = MODEL_PRESETS[k]
        model_id = a.model_id or preset['model_id']
        if preset['gated'] and a.model_id is None:
            print(f"[info] {k} uses a gated checkpoint ({model_id}). If loading fails "
                  "with 401/403, accept the license on the model page and run "
                  "`huggingface-cli login` or set HF_TOKEN.")
        try:
            all_rows += run_one(k, model_id, a.dataset, a.limit_tokens,
                                a.wikitext_file, device)
        except Exception as e:
            print(f"[SKIPPED] {k}: {e}")

    cols = ['model', 'dataset', 'module', 'mode', 'ppl', 'd_ppl',
            'next_tok_agree_pct', 'cut_n_pct', 'dead_pct', 'sum_dev_pct']
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(all_rows)
    print(f"\nwrote {a.out}  ({len(all_rows)} rows)")
    print("\nReminder: unlike the CNN case, perplexity here CAN move for every mode, "
          "including Stable -- the softmax output feeds a weighted sum (x V) and then "
          "a residual add, so magnitude errors (e.g. the reciprocal's sum-deviation) "
          "propagate forward instead of being absorbed by an argmax.")


if __name__ == "__main__":
    main()