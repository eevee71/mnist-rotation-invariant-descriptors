from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.metrics import cluster_accuracy


def _score(y, pred):

    return dict(
        acc=cluster_accuracy(y, pred),
        ari=adjusted_rand_score(y, pred),
        nmi=normalized_mutual_info_score(y, pred),
    )


def print_compare(out):

    fmt = "{:<32} {:>7.4f} {:>7.4f} {:>7.4f}"
    print(f"{'method':<32} {'acc':>7} {'ari':>7} {'nmi':>7}")
    for name, s in out.items():
        print(fmt.format(name, s['acc'], s['ari'], s['nmi']))

    best = max(out.items(), key=lambda x: x[1]['acc'])
    print(f"\nBEST: {best[0]} -> acc={best[1]['acc']:.4f}")


def print_report(results, top=10):
    fmt = "{:>3} {:>3} {:>6.2f} {:>4} {:>7.4f} {:>7.4f} {:>7.4f}"

    print(f"\n{'deg':>3} {'K':>3} {'alpha':>6} {'dim':>4} {'acc':>7} {'ari':>7} {'nmi':>7}")
    for r in sorted(results, key=lambda x: x['acc'], reverse=True)[:top]:
        print(fmt.format(r['deg'], r['K'], r['alpha'], r['dim'], r['acc'], r['ari'], r['nmi']))

    best = max(results, key=lambda x: x['acc'])
    print(f"\nBEST: deg={best['deg']}, K={best['K']}, alpha={best['alpha']:.2f} -> "
          f"acc={best['acc']:.4f}, ari={best['ari']:.4f}, nmi={best['nmi']:.4f}")

    return best