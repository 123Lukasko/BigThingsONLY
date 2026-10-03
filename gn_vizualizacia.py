"""
Vizualizácia Gauss-Newtona (single shooting) na modeloch z DP.

    python gn_vizualizacia.py                       # lineárny model
    python gn_vizualizacia.py --model bernoulli
    python gn_vizualizacia.py --p0 0.5 20 --sigma 2

Výstup (vysledky_grafy/):
    gn_<model>_cesta_kontury.png   dráha p_k na vrstevniciach S(α, β)
    gn_<model>_konvergencia.png    S(p_k) a ||Δp_k|| cez iterácie
    gn_<model>_fit_iteracie.png    krivky y(t; p_k) oproti dátam
"""

import argparse

import matplotlib
matplotlib.use("Agg")

import numpy as np

import viz
from gauss_newton import (gauss_newton, model_bernoulli_gn, model_linearny_gn,
                          simuluj_s_citlivostami, single_shooting)
from modely import bernoulli_analyticka, bernoulli_singularita, linearna_analyticka


def kontrola_jakobianu(model, t, p, h=1e-6):
    """Citlivosti vs. centrálne diferencie — rýchly test, že J sedí."""
    _, J = simuluj_s_citlivostami(model, t, p)
    J_fd = np.empty_like(J)
    for j in range(len(p)):
        e = np.zeros(len(p)); e[j] = h
        yp, _ = simuluj_s_citlivostami(model, t, p + e)
        ym, _ = simuluj_s_citlivostami(model, t, p - e)
        J_fd[:, j] = (yp - ym) / (2 * h)
    return np.max(np.abs(J - J_fd)) / np.max(np.abs(J_fd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["linearny", "bernoulli"], default="linearny")
    ap.add_argument("--p0", type=float, nargs=2, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--output-dir", default=viz.VYSTUP)
    args = ap.parse_args()

    if args.model == "linearny":
        model, p_true = model_linearny_gn(), np.array([3.0, 9.0])
        predikcia = lambda t, p: linearna_analyticka(t, p, y0=model.y0)
        T = args.T or 1.0
        sigma = 1.5 if args.sigma is None else args.sigma
        p0 = np.array(args.p0 or (0.5, 20.0))
        okno = ((-1.0, 5.0), (-10.0, 30.0))
    else:
        model, p_true = model_bernoulli_gn(), np.array([1.0, 2.0])
        predikcia = lambda t, p: bernoulli_analyticka(t, p, y0=model.y0)
        T = args.T or 0.8 * bernoulli_singularita(p_true)
        sigma = 0.05 if args.sigma is None else args.sigma
        p0 = np.array(args.p0 or (0.2, 1.0))
        okno = ((-0.5, 2.5), (-0.5, 4.0))

    # dáta: CRN konvencia  y_obs = y_true + σ·z
    rng = np.random.default_rng(args.seed)
    t = np.linspace(0.0, T, args.n)
    y_true = predikcia(t, p_true)
    y_obs = y_true + sigma * rng.standard_normal(t.size)

    print(f"model: {model.nazov}, p_true={p_true}, T={T:.4f}, n={args.n}, σ={sigma}")
    print(f"relatívna chyba J (citlivosti vs diferencie): "
          f"{kontrola_jakobianu(model, t, p0):.2e}")

    rj = single_shooting(model, t, y_obs)
    bounds = (np.array([okno[0][0], okno[1][0]]), np.array([okno[0][1], okno[1][1]]))
    drahy = {
        "Čistý GN": gauss_newton(rj, p0, max_iter=args.max_iter, bounds=bounds),
        "GN s polením kroku": gauss_newton(rj, p0, max_iter=args.max_iter,
                                           tlmenie=True, bounds=bounds),
    }
    for label, res in drahy.items():
        print(f"{label:20s}: p̂={res.p}, S={res.historia_S[-1]:.4g}, "
              f"it={res.iteracie}, {res.sprava}")

    # vrstevnice S(α, β) — analytické riešenie, nech je mriežka rýchla
    a = np.linspace(*okno[0], 160)
    b = np.linspace(*okno[1], 160)
    A, B = np.meshgrid(a, b)
    with np.errstate(all="ignore"):
        S = np.array([[np.sum((predikcia(t, (ai, bi)) - y_obs) ** 2)
                       for ai, bi in zip(ra, rb)] for ra, rb in zip(A, B)])

    viz.nastav_styl()
    pref = f"gn_{args.model}"
    nadpis = f"{model.nazov} model | σ={sigma}, T={T:.2f}, n={args.n}"
    viz.gn_cesta_na_konturach(A, B, S, drahy, p_true=p_true,
                              title=f"Dráha GN | {nadpis}",
                              output_dir=args.output_dir,
                              filename=f"{pref}_cesta_kontury", show=False)
    viz.gn_konvergencia(drahy, title=f"Konvergencia GN | {nadpis}",
                        output_dir=args.output_dir,
                        filename=f"{pref}_konvergencia", show=False)
    viz.gn_fit_iteracie(t, y_obs, y_true, drahy["GN s polením kroku"], predikcia,
                        title=f"Fit počas iterácií (GN s polením kroku) | {nadpis}",
                        output_dir=args.output_dir,
                        filename=f"{pref}_fit_iteracie", show=False)


if __name__ == "__main__":
    main()
