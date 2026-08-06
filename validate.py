"""
validate_cnn.py -- validate hardware softmax variants on pretrained CNN
classifiers, without retraining.

MODELS  : resnet50, densenet121          (via timm, weights hosted on HF Hub)
DATASETS: cifar10, cifar100, oxford_pets (37-class breeds), svhn (10-class digits)

WHY THIS EXPERIMENT CANNOT MOVE TOP-1 THE WAY THE ATTENTION EXPERIMENT DID
---------------------------------------------------------------------------
ResNet/DenseNet apply softmax nowhere inside the model -- they return raw
logits, and softmax is something you apply afterward to read off a
probability. Because the exponential LUT is monotone and the reciprocal is
one positive scalar shared by the whole row, argmax(HWSoftmax(logits)) ==
argmax(logits) for every mode that keeps the maximum -- which is Stable,
DDF and SG-DDF by construction (DDF's threshold is max-(max-min)/2, and the
max always satisfies score >= threshold, so it is never pruned).

The one mode that CAN move top-1 is SWAT: with no max-subtraction, a wide
logit range can push the exponential sum past 2^16, the reciprocal
collapses to 0, and the unit emits an all-zero vector -- at which point
argmax silently returns index 0 and the prediction is simply wrong. This
script reports that as the DEAD-INDUCED ACCURACY LOSS column, which is the
only genuine top-1 result available here; everything else in this file is
a calibration story (NLL, ECE, Brier, JSD), not an accuracy story. See the
printed summary and the CSV notes column -- do not report a "SG-DDF
improves CNN accuracy" claim from this data, because SG-DDF cannot
mathematically change CNN top-1.

USAGE
  pip install timm --break-system-packages          # for every checkpoint here
  pip install detectors --break-system-packages     # registers the CIFAR-10/100 timm ids

  python3 validate_cnn.py --model resnet50    --dataset cifar10      --limit 500
  python3 validate_cnn.py --model densenet121 --dataset cifar100     --limit 500
  python3 validate_cnn.py --model resnet50    --dataset oxford_pets  --limit 500
  python3 validate_cnn.py --model resnet50    --dataset svhn         --limit 2000

  # sweep everything into one CSV (skips the two unavailable combos -- see below):
  python3 validate_cnn.py --all --limit 500

NOTE ON densenet121 + oxford_pets / densenet121 + svhn
  No verified fine-tuned checkpoint for either pair was found on the Hub.
  Requesting one raises a clear error rather than guessing a repo name that
  might not exist. Add your own entry to CKPT below if you have one.

CHECKPOINTS USED (all pretrained weights, no fine-tuning here)
  resnet50    + cifar10      : timm 'resnet50_cifar10'          (edadaltocg, HF Hub)
  densenet121 + cifar10      : timm 'densenet121_cifar10'       (edadaltocg, HF Hub)
  resnet50    + cifar100     : timm 'resnet50_cifar100'         (edadaltocg, HF Hub)
  densenet121 + cifar100     : timm 'densenet121_cifar100'      (edadaltocg, HF Hub)
  resnet50    + oxford_pets  : timm 'hf-hub:nateraw/resnet50-oxford-iiit-pet' (37 breeds)
  resnet50    + svhn         : timm 'resnet50_svhn'             (edadaltocg, HF Hub)
  densenet121 + oxford_pets  : NOT AVAILABLE -- see load_model()'s error message
  densenet121 + svhn         : NOT AVAILABLE -- see load_model()'s error message
    Neither has a verified fine-tuned checkpoint on the Hub. Requesting
    either raises a clear error instead of guessing a repo name that might
    404. Add your own entry to CKPT below if you have one.
  The CIFAR-10/100 + SVHN checkpoints require `import detectors` before
  `timm.create_model` (it registers the architecture variant) -- this
  script does that for you.
"""
import argparse, csv, sys, time
import numpy as np
import torch
import torch.nn.functional as F

from softmax_hw import HWSoftmax

MODULES = ('mul255', 'shift3', 'soft')
MODES = {0: 'SWAT', 1: 'Stable', 2: 'DDF', 3: 'SG-DDF'}

CKPT = {
    ('resnet50', 'cifar10'):       dict(timm_name='resnet50_cifar10',        need_detectors=True,  num_classes=10),
    ('densenet121', 'cifar10'):    dict(timm_name='densenet121_cifar10',     need_detectors=True,  num_classes=10),
    ('resnet50', 'cifar100'):      dict(timm_name='resnet50_cifar100',       need_detectors=True,  num_classes=100),
    ('densenet121', 'cifar100'):   dict(timm_name='densenet121_cifar100',    need_detectors=True,  num_classes=100),
    ('resnet50', 'oxford_pets'):   dict(timm_name='hf-hub:nateraw/resnet50-oxford-iiit-pet',
                                        need_detectors=False, num_classes=37),
    ('resnet50', 'svhn'):          dict(timm_name='resnet50_svhn',           need_detectors=True,  num_classes=10),
}


def load_model(model_name, dataset, device):
    import timm
    if (model_name, dataset) not in CKPT:
        raise SystemExit(
            f"No verified checkpoint for {model_name}/{dataset}. See the "
            f"CHECKPOINTS USED note at the top of this file -- densenet121 "
            f"currently has no known fine-tuned checkpoint for oxford_pets "
            f"or svhn. Add your own entry to CKPT if you have one.")
    cfg = CKPT[(model_name, dataset)]
    if cfg['need_detectors']:
        try:
            import detectors  # noqa: F401  -- side effect: registers CIFAR-10 timm ids
        except ImportError:
            raise SystemExit(
                f"'{cfg['timm_name']}' needs the `detectors` package to register with "
                f"timm. Install with: pip install detectors --break-system-packages")
    model = timm.create_model(cfg['timm_name'], pretrained=True).to(device).eval()
    data_cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_cfg)
    return model, transform, cfg['num_classes']


def load_data(dataset, transform, limit):
    if dataset in ('cifar10', 'cifar100'):
        from datasets import load_dataset
        # NOTE: cifar100's HF dataset uses different column names than cifar10
        # (fine_label + coarse_label, not label) -- a naive shared code path
        # here would KeyError or silently grab the wrong column.
        hf_name = 'cifar100' if dataset == 'cifar100' else 'cifar10'
        label_col = 'fine_label' if dataset == 'cifar100' else 'label'
        ds = load_dataset(hf_name, split="test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        imgs = [transform(x.convert('RGB')) for x in ds['img']]
        labels = ds[label_col]
        return torch.stack(imgs), torch.tensor(labels)

    if dataset == 'oxford_pets':
        from datasets import load_dataset
        # NOTE: this dataset's image column is 'image', not 'img' like
        # cifar10/100 -- a third distinct naming convention. Verified
        # against the timm/oxford-iiit-pet dataset card before writing this.
        ds = load_dataset("timm/oxford-iiit-pet", split="test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        imgs = [transform(x.convert('RGB')) for x in ds['image']]
        labels = ds['label']
        return torch.stack(imgs), torch.tensor(labels)

    if dataset == 'svhn':
        from datasets import load_dataset
        # NOTE: SVHN ships as multiple configs (cropped_digits vs
        # full_numbers) under one repo -- the config name is a required
        # positional arg, not optional, or you silently get the wrong task
        # (object detection labels instead of a single class id).
        ds = load_dataset("ufldl-stanford/svhn", "cropped_digits", split="test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        imgs = [transform(x.convert('RGB')) for x in ds['image']]
        labels = ds['label']
        return torch.stack(imgs), torch.tensor(labels)

    raise ValueError(f"no data-loading path for dataset={dataset!r}")


def expected_calibration_error(conf, correct, n_bins=15):
    """Standard ECE: weighted gap between confidence and accuracy per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


@torch.no_grad()
def batched_logits(model, imgs, device, bs=64):
    out = []
    for i in range(0, len(imgs), bs):
        out.append(model(imgs[i:i + bs].to(device)).cpu())
    return torch.cat(out)


def evaluate_variant(logits, labels, hw_or_none, n_classes):
    """hw_or_none=None -> exact float softmax baseline."""
    if hw_or_none is None:
        probs = F.softmax(logits, dim=-1)
        cut_n = cut_mass = dead_pct = sum_dev = 0.0
    else:
        probs = hw_or_none(logits.double())  # HWSoftmax quantises internally
        st = hw_or_none.stats()
        cut_n, dead_pct, sum_dev = st['cut_n_pct'], st['dead_pct'], st['sum_dev_pct']
        cut_mass = float('nan')  # not tracked per-element in HWSoftmax; see softmax_hw.py

    preds = probs.argmax(-1)
    correct = (preds == labels).numpy().astype(float)
    top1 = 100.0 * correct.mean()

    k = min(5, n_classes)
    top5 = 100.0 * (probs.topk(k, dim=-1).indices == labels[:, None]).any(-1).float().mean().item()

    p = probs.clamp(min=1e-9)
    nll = float(F.nll_loss(p.log(), labels).item())
    brier = float(((probs - F.one_hot(labels, n_classes)) ** 2).sum(-1).mean().item())

    conf, _ = probs.max(-1)
    ece = expected_calibration_error(conf.numpy(), correct)

    ref = F.softmax(logits, dim=-1).clamp(min=1e-12)
    q = probs.clamp(min=1e-12)
    m = 0.5 * (ref + q)
    jsd = float((0.5 * (ref * (ref / m).log()).sum(-1) +
                 0.5 * (q * (q / m).log()).sum(-1)).mean().item() / np.log(2))

    return dict(top1=top1, top5=top5, nll=nll, ece=100 * ece, brier=brier, jsd=jsd,
                cut_n_pct=cut_n, dead_pct=dead_pct, sum_dev_pct=sum_dev)


def run_one(model_name, dataset, limit, device):
    print(f"\n{'='*70}\n{model_name} on {dataset}\n{'='*70}")
    model, transform, n_classes = load_model(model_name, dataset, device)
    imgs, labels = load_data(dataset, transform, limit)
    print(f"[info] {len(imgs)} images, {n_classes}-way classifier")

    t0 = time.time()
    logits = batched_logits(model, imgs, device)
    print(f"[info] inference done in {time.time()-t0:.0f}s")

    rows = []
    base = evaluate_variant(logits, labels, None, n_classes)
    base['model'], base['dataset'], base['module'], base['mode'] = model_name, dataset, '-', 'BASELINE'
    rows.append(base)
    print(f"BASELINE: top1={base['top1']:.2f}  top5={base['top5']:.2f}  "
          f"nll={base['nll']:.4f}  ece={base['ece']:.2f}%")

    print(f"{'module':>8}|{'mode':>7}|{'top1':>7}|{'d-top1':>7}|{'top5':>7}"
          f"|{'nll':>7}|{'ece%':>6}|{'brier':>7}|{'jsd':>7}|{'dead%':>6}|{'sumdev%':>8}")
    for module in MODULES:
        for mode in (0, 1, 2, 3):
            hw = HWSoftmax(module, mode, device='cpu')
            r = evaluate_variant(logits, labels, hw, n_classes)
            r['model'], r['dataset'], r['module'], r['mode'] = model_name, dataset, module, MODES[mode]
            rows.append(r)
            note = ""
            if mode == 0 and abs(r['top1'] - base['top1']) > 0.05:
                note = "  <-- SWAT overflow cost real accuracy"
            elif mode != 0 and abs(r['top1'] - base['top1']) > 0.5:
                # A few tenths of a point of drift from baseline is expected:
                # Q8.8 rounds logits to 1/256, and rows whose top-1/top-2 gap
                # is below that quantum can flip regardless of pruning. This
                # hits every non-SWAT mode identically -- it is quantization
                # noise, not a pruning artifact. Only flag a LARGE gap here.
                note = "  !! check: drift beyond quantization-tie noise"
            print(f"{module:>8}|{MODES[mode]:>7}|{r['top1']:7.2f}|{r['top1']-base['top1']:+7.2f}"
                  f"|{r['top5']:7.2f}|{r['nll']:7.4f}|{r['ece']:6.2f}|{r['brier']:7.4f}"
                  f"|{r['jsd']:7.4f}|{r['dead_pct']:6.1f}|{r['sum_dev_pct']:8.2f}{note}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet50", "densenet121"])
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "oxford_pets", "svhn"])
    ap.add_argument("--all", action="store_true", help="run every model x dataset pair in CKPT")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--out", default="cnn_softmax_results.csv")
    a = ap.parse_args()
    if not a.all and not (a.model and a.dataset):
        sys.exit("pass --model + --dataset, or --all")
    device = "cuda" if (a.cuda and torch.cuda.is_available()) else "cpu"

    combos = (list(CKPT.keys())   # every verified {model, dataset} pair
              if a.all else [(a.model, a.dataset)])

    all_rows = []
    for model_name, dataset in combos:
        try:
            all_rows += run_one(model_name, dataset, a.limit, device)
        except Exception as e:
            print(f"[SKIPPED] {model_name}/{dataset}: {e}")

    cols = ['model', 'dataset', 'module', 'mode', 'top1', 'top5', 'nll', 'ece',
            'brier', 'jsd', 'cut_n_pct', 'dead_pct', 'sum_dev_pct']
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(all_rows)
    print(f"\nwrote {a.out}  ({len(all_rows)} rows)")
    print("\nReminder: Stable/DDF/SG-DDF should sit within a few tenths of a point of "
          "BASELINE top1 -- small drift is expected from Q8.8 rounding on near-tied "
          "logits (top1/top2 gap below 1/256) and hits every non-SWAT mode identically, "
          "since none of them can prune the argmax. SWAT's drop is the one genuine "
          "pruning/overflow effect here. A large gap on a non-SWAT mode is a bug; a "
          "fraction of a point is not.")


if __name__ == "__main__":
    main()
