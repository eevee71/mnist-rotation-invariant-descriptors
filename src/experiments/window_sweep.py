import math
import torch
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from data.dataloader import load_data
from src.models.mlp import train_mlp
from src.moment_transforms import MomentTransform
from src.dataset_preparation import prepare_pipeline


# rotation-invariant radial window functions r = ||x||
WINDOW_FUNCTIONS = {

    # Standard Gaussian
    "01_Standard_Gaussian_r2": lambda r, x: torch.exp(-(r**2) / 2.0),
    "02_Gaussian_Narrow_r2_scale0.5": lambda r, x: torch.exp(-(r**2)),
    "03_Gaussian_Wide_r2_scale2.0": lambda r, x: torch.exp(-(r**2) / 4.0),

    # Ring / Donut Gaussian
    "04_Ring_Gaussian_r1.0": lambda r, x: torch.exp(-((r - 1.0) ** 2) / 2.0),
    "05_Ring_Gaussian_r0.5": lambda r, x: torch.exp(-((r - 0.5) ** 2) / 2.0),
    "06_Ring_Gaussian_r1.5": lambda r, x: torch.exp(-((r - 1.5) ** 2) / 2.0),
    "07_Ring_Gaussian_r2.0": lambda r, x: torch.exp(-((r - 2.0) ** 2) / 2.0),

    # Exponential Power / Laplace
    "08_Laplace_r1": lambda r, x: torch.exp(-r),
    "09_Laplace_r1_scale0.5": lambda r, x: torch.exp(-2.0 * r),
    "10_ExpPower_r1.5": lambda r, x: torch.exp(-(r**1.5)),
    "11_ExpPower_r2.5": lambda r, x: torch.exp(-(r**2.5)),
    "12_ExpPower_r3.0": lambda r, x: torch.exp(-(r**3.0)),

    "13_Hybrid_Center_Plus_Ring1.0": lambda r, x: 0.5 * torch.exp(-(r**2) / 2.0)
    + 0.5 * torch.exp(-((r - 1.0) ** 2) / 2.0),
    "14_Hybrid_Center_Plus_Ring1.5": lambda r, x: 0.5 * torch.exp(-(r**2) / 2.0)
    + 0.5 * torch.exp(-((r - 1.5) ** 2) / 2.0),
    "15_Cauchy_1_over_1_plus_r2": lambda r, x: 1.0 / (1.0 + r**2),
    "16_HeavyTail_1_over_1_plus_r4": lambda r, x: 1.0 / (1.0 + r**4),
    "17_Compact_Cutoff_r1.5": lambda r, x: torch.where(
        r <= 1.5, torch.exp(-(r**2) / 2.0), torch.tensor(0.0, device=r.device)
    ),
    "18_Compact_Cutoff_r2.0": lambda r, x: torch.where(
        r <= 2.0, torch.exp(-(r**2) / 2.0), torch.tensor(0.0, device=r.device)
    ),
    "19_Cosine_Window_R2.0": lambda r, x: torch.where(
        r <= 2.0,
        torch.cos((torch.pi * r) / 4.0) ** 2,
        torch.tensor(0.0, device=r.device),
    ),
    "20_Logistic_Sigmoid_Window": lambda r, x: 1.0 / (1.0 + torch.exp(2.0 * (r - 1.0))),
}


def create_custom_complex_coefficients(window_fn):
    """Dynamic override for complex_coefficients using specified radial window."""

    def custom_complex_coefficients(self, images):
        dm, xc, yc = self.prepare_density_center(images)
        trace = self.covariance_invariants(dm, xc, yc)[:, 0]
        xn, yn = self.normalization(xc, yc, trace)

        z = torch.complex(xn, yn)
        z_bar = torch.conj(z)
        z_abs2 = xn**2 + yn**2
        r = torch.sqrt(z_abs2 + 1e-9)

        window = window_fn(r, z)

        coeffs_list = []
        index = []
        d = self.max_degree

        for n in range(d + 1):
            for m in range(n + 1):
                if n + m <= d:
                    H_nm = torch.zeros_like(z)
                    for k in range(min(n, m) + 1):
                        coeff = (
                            ((-1) ** k)
                            * math.factorial(n)
                            * math.factorial(m)
                            / (
                                math.factorial(k)
                                * math.factorial(n - k)
                                * math.factorial(m - k)
                            )
                        )
                        z_pow = z ** (n - k) if (n - k) > 0 else torch.ones_like(z)
                        z_bar_pow = z_bar ** (m - k) if (m - k) > 0 else torch.ones_like(z)
                        H_nm = H_nm + coeff * z_pow * z_bar_pow

                    norm = 1.0 / math.sqrt(
                        math.pi * (2 ** (n + m)) * math.factorial(n) * math.factorial(m)
                    )
                    psi_nm = norm * H_nm * window

                    c_nm = (dm * torch.conj(psi_nm)).sum(dim=(-2, -1))
                    coeffs_list.append(c_nm)
                    index.append((n, m))

        coeffs = torch.stack(coeffs_list, dim=-1)
        return coeffs, index

    return custom_complex_coefficients


def run_window_sweep(data, targets, degree=9, k=9, epochs=40, eval_n=5000):
    """Evaluate window functions on QDA and MLP classifiers."""

    results = []

    print(f"\n=== SWEEP OVER {len(WINDOW_FUNCTIONS)} WINDOW FUNCTIONS ===")
    original_coeffs_fn = MomentTransform.complex_coefficients

    for idx, (name, win_fn) in enumerate(WINDOW_FUNCTIONS.items(), 1):
        print(f"[{idx:02d}/{len(WINDOW_FUNCTIONS)}] Testing: {name} ...")

        MomentTransform.complex_coefficients = create_custom_complex_coefficients(win_fn)

        try:
            Xtr, Xte, ytr, yte, _ = prepare_pipeline(data, targets, degree=degree, k=k, eval_n=eval_n)

            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            qda = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(Xtr_s, ytr)
            qda_acc = (qda.predict(Xte_s) == yte).mean()

            _, mlp_acc, _ = train_mlp(
                data,
                targets,
                degree=degree,
                k=k,
                epochs=epochs,
                batch_size=64,
                seed=0,
                log_interval=999,
            )

            results.append(
                {
                    "name": name,
                    "qda_acc": qda_acc,
                    "mlp_acc": mlp_acc,
                    "avg_acc": (qda_acc + mlp_acc) / 2.0,
                }
            )

            print(f"    --> QDA: {qda_acc * 100:.2f}% | MLP: {mlp_acc * 100:.2f}%\n")

        except Exception as e:
            print(f"    [ERROR] {name}: {e}\n")

    MomentTransform.complex_coefficients = original_coeffs_fn
    print_results_table(results)


def print_results_table(results):
    """Print sorted results summary."""

    results_sorted = sorted(results, key=lambda x: x["avg_acc"], reverse=True)

    print("\n" + "=" * 75)
    print(f"{'WINDOW FUNCTION NAME':<35} | {'QDA ACC':<10} | {'MLP ACC':<10} | {'AVG ACC':<10}")
    print("=" * 75)

    for r in results_sorted:
        print(
            f"{r['name']:<35} | {r['qda_acc']*100:6.2f}%    | {r['mlp_acc']*100:6.2f}%    | {r['avg_acc']*100:6.2f}%"
        )

    print("=" * 75)

    best_qda = max(results, key=lambda x: x["qda_acc"])
    best_mlp = max(results, key=lambda x: x["mlp_acc"])
    best_overall = results_sorted[0]

    print(f"\nBEST QDA     : {best_qda['name']} ({best_qda['qda_acc']*100:.2f}%)")
    print(f"BEST MLP     : {best_mlp['name']} ({best_mlp['mlp_acc']*100:.2f}%)")
    print(f"BEST OVERALL : {best_overall['name']} (Avg: {best_overall['avg_acc']*100:.2f}%)\n")


if __name__ == "__main__":
    data, targets = load_data(full_dataset=True)
    run_window_sweep(data, targets, degree=9, k=9, epochs=60, eval_n=5000)