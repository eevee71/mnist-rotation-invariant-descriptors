"""Turn ``results/sweep2d.jsonl`` into the tables you actually want to read.

    python scripts/analyze_sweep2d.py
    python scripts/analyze_sweep2d.py --metric qda_acc
"""

import argparse
import json

import numpy as np
import pandas as pd


def load(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', default='results/sweep2d.jsonl')
    p.add_argument('--metric', default='mlp_acc', choices=['mlp_acc', 'qda_acc'])
    args = p.parse_args()

    df = load(args.path)
    m = args.metric
    df = df.dropna(subset=[m])

    agg = (df.groupby(['method', 'degree', 'moment_scale'])[m]
             .agg(['mean', 'std', 'count']).reset_index())

    for method in sorted(agg['method'].unique()):
        sub = agg[agg['method'] == method]
        piv = sub.pivot(index='degree', columns='moment_scale', values='mean')
        print(f"\n=== {m} | method = {method}  (rows: max_degree, cols: moment_scale)")
        print((piv * 100).round(2).to_string())
        best = sub.loc[sub['mean'].idxmax()]
        print(f"  best: degree={int(best['degree'])} "
              f"moment_scale={best['moment_scale']} "
              f"-> {best['mean'] * 100:.2f}% +- {best['std'] * 100:.2f}")

    if {'point', 'integral'} <= set(agg['method'].unique()):
        wide = agg.pivot_table(index=['degree', 'moment_scale'],
                               columns='method', values='mean')
        wide['delta_pp'] = (wide['integral'] - wide['point']) * 100
        piv = wide.reset_index().pivot(index='degree', columns='moment_scale',
                                       values='delta_pp')
        print(f"\n=== {m}: integral MINUS point, percentage points")
        print(piv.round(2).to_string())

        # is the difference bigger than seed noise?
        noise = df.groupby(['method', 'degree', 'moment_scale'])[m].std().mean()
        print(f"\n  mean across-seed std: {noise * 100:.2f} pp")
        print(f"  mean |delta|:         {piv.abs().to_numpy()[~np.isnan(piv.to_numpy())].mean():.2f} pp")
        print(f"  mean signed delta:    {np.nanmean(piv.to_numpy()):+.2f} pp")
        print("\n  If |delta| sits inside the seed noise, the point-mass "
              "treatment costs you nothing and you should keep it.")

    print("\n=== dimensionality by degree")
    print(df.groupby('degree')['dim'].first().to_string())


if __name__ == '__main__':
    main()
