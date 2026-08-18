#!/usr/bin/env python
"""
Print the full nn.Module tree of any model (classic print(model)).

Usage:
    python scripts/eval/print_model.py --model segformer
    python scripts/eval/print_model.py --model attunet --no-uncertainty
    python scripts/eval/print_model.py --model unet_pytorch --out reports/resnet34_arch.txt

Model names match config / build_model:
    pconv | unet | attunet | unet_pytorch | segformer
Encoder can be overridden with --encoder_name (default per model).
"""
import argparse
import sys
from pathlib import Path

# allow running from anywhere (add repo root to path)

from elevcomp.model import build_model, count_parameters


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True,
                   choices=['pconv', 'unet', 'attunet', 'unet_pytorch',
                            'segformer'])
    p.add_argument('--encoder_name', default=None,
                   help='override encoder (default per model)')
    p.add_argument('--encoder_weights', default='none',
                   help="'imagenet' | 'none' (default: none — no download just to print)")
    p.add_argument('--no-uncertainty', dest='uncertainty', action='store_false',
                   help='build without the β-NLL log_var head')
    p.add_argument('--out', default=None, metavar='FILE',
                   help='also write the printout to this file')
    args = p.parse_args()

    cfg = {'model': args.model, 'uncertainty': args.uncertainty,
           'encoder_weights': args.encoder_weights}
    if args.encoder_name:
        cfg['encoder_name'] = args.encoder_name

    model = build_model(cfg)
    header = (f'# model={args.model}  uncertainty={args.uncertainty}  '
              f'params={count_parameters(model)/1e6:.2f}M\n')
    text = header + str(model) + '\n'

    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f'\n# written to {args.out}')


if __name__ == '__main__':
    main()
