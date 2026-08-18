#!/usr/bin/env python3
"""
Computational cost of both inference modes — for the paper's deployment table.

Measures, on the current GPU (or CPU):
    parameters, single-pass (β-NLL) latency, D4 TTA (8×) latency,
    throughput and peak GPU memory, at the requested batch sizes.

Latency = median over --iters timed runs after --warmup warm-up runs, with
torch.cuda.synchronize() around each timed region.

Usage:
    python scripts/eval/benchmark_inference.py --checkpoint runs/cv_x/fold_0_ForestEnv/best.pth
    python scripts/eval/benchmark_inference.py --model unet          # fresh weights, configs/cv_resnet34.toml
    python scripts/eval/benchmark_inference.py --model pconv --batch_sizes 1 8
"""
import argparse
import json
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path
from elevcomp.paths import CONFIG_DIR


import numpy as np
import torch

from elevcomp.inference import nll_predict, tta_predict
from elevcomp.model import build_model, count_parameters
from elevcomp.utils import environment_info, write_csv


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def time_fn(fn, inp, device, warmup: int, iters: int) -> float:
    """Median wall-clock seconds of fn(inp)."""
    for _ in range(warmup):
        fn(inp)
    _sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(inp)
        _sync(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    p = argparse.ArgumentParser(description='Benchmark β-NLL vs TTA inference cost.')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--checkpoint', help='.pth checkpoint (uses its stored config)')
    src.add_argument('--model', choices=['pconv', 'unet', 'attunet'],
                     help='fresh model built from configs/cv_resnet34.toml')
    p.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 8, 32])
    p.add_argument('--input_size', type=int, default=256)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default=str(Path(__file__).resolve().parent.parent
                                        / 'reports'))
    args = p.parse_args()

    device = torch.device(args.device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = ckpt['config']
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt['model'])
        source = args.checkpoint
    else:
        with open(CONFIG_DIR / 'cv_resnet34.toml', 'rb') as f:
            cfg = tomllib.load(f)
        cfg['model'] = args.model
        model = build_model(cfg).to(device)
        source = f'fresh weights, configs/cv_resnet34.toml, model={args.model}'
    model.eval()

    uncertainty = cfg.get('uncertainty', False)
    n_params = count_parameters(model)
    gpu = (torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU')

    print(f'Model       : {cfg["model"]}  ({n_params:,} params)')
    print(f'Source      : {source}')
    print(f'Device      : {gpu}')
    print(f'Input       : (B, 2, {args.input_size}, {args.input_size})  |  '
          f'median of {args.iters} iters after {args.warmup} warm-up\n')

    rows = []
    hdr = (f'{"batch":>5} {"mode":>6} {"latency [ms]":>13} {"ms/sample":>10} '
           f'{"samples/s":>10} {"peak mem [MB]":>14}')
    print(hdr)
    print('─' * len(hdr))

    for bs in args.batch_sizes:
        inp = torch.randn(bs, 2, args.input_size, args.input_size, device=device)
        # Channel 1 is a binary mask in real data — keep it plausible for pconv.
        inp[:, 1] = (inp[:, 1] > 0).float()

        for mode, fn in (('nll', lambda x: nll_predict(model, x, uncertainty)),
                         ('tta', lambda x: tta_predict(model, x))):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            try:
                lat = time_fn(fn, inp, device, args.warmup, args.iters)
            except torch.cuda.OutOfMemoryError:
                print(f'{bs:>5} {mode:>6} {"OOM":>13}')
                continue
            mem = (torch.cuda.max_memory_allocated(device) / 2**20
                   if device.type == 'cuda' else float('nan'))
            row = {'model': cfg['model'], 'batch': bs, 'mode': mode,
                   'latency_ms': lat * 1e3, 'ms_per_sample': lat * 1e3 / bs,
                   'samples_per_s': bs / lat, 'peak_mem_mb': mem}
            rows.append(row)
            print(f'{bs:>5} {mode:>6} {lat*1e3:>13.2f} {lat*1e3/bs:>10.2f} '
                  f'{bs/lat:>10.1f} {mem:>14.0f}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'benchmark_{cfg["model"]}'
    with open(out_dir / f'{stem}.json', 'w') as f:
        json.dump({'generated_at': datetime.now().isoformat(timespec='seconds'),
                   'source': source, 'device': gpu, 'params': n_params,
                   'input_size': args.input_size, 'uncertainty': uncertainty,
                   'iters': args.iters, 'results': rows,
                   'environment_info': environment_info()}, f, indent=2)
    write_csv(out_dir / f'{stem}.csv', rows)

    # LaTeX rows for the compute table (batch = 1, the deployment case).
    b1 = {r['mode']: r for r in rows if r['batch'] == 1}
    if b1:
        lines = [
            '% Auto-generated by scripts/eval/benchmark_inference.py',
            f'% {gpu}, input 2x{args.input_size}x{args.input_size}, batch 1, '
            f'median of {args.iters} runs',
            r'\begin{tabular}{lrrrr}',
            r'\toprule',
            r'Uncertainty mode & Params & Forward passes & Latency [ms] & Peak mem [MB] \\',
            r'\midrule',
            f'Single-pass $\\beta$-NLL $\\hat{{\\sigma}}$ & {n_params/1e6:.1f}M & 1 & '
            f'{b1["nll"]["latency_ms"]:.1f} & {b1["nll"]["peak_mem_mb"]:.0f}' + r' \\',
            f'TTA ensemble $\\hat{{\\sigma}}$ (D4) & {n_params/1e6:.1f}M & 8 & '
            f'{b1["tta"]["latency_ms"]:.1f} & {b1["tta"]["peak_mem_mb"]:.0f}' + r' \\',
            r'\bottomrule', r'\end{tabular}',
        ]
        (out_dir / f'table_compute_{cfg["model"]}.tex').write_text('\n'.join(lines) + '\n')

    print(f'\nWritten → {out_dir}/{stem}.{{json,csv}} + table_compute_{cfg["model"]}.tex')


if __name__ == '__main__':
    main()
