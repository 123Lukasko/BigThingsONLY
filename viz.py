"""
Vizualizácie pre odhad parametrov dynamického modelu.

Kánon prevzatý z notebookov DP (`vykresli_graf`, `viz_*`,
`xy_plot_over_sigma_compare*`) a dotiahnutý do znovupoužiteľného modulu.
Signatúry pôvodných funkcií sú zachované, aby sa dal modul nasadiť do
existujúcich notebookov bez prepisovania volaní.

Pravidlá, ktoré tento modul vynucuje:
  * matplotlib + seaborn(style="whitegrid"), žiadna iná knižnica
  * jedna figure = jedna myšlienka; viac metód sa porovnáva v jednej osi
    (`..._compare`) alebo v dvoch podgrafoch so zdieľanou osou x (`..._subplots`)
  * slovenské popisky, σ v popiskoch osí, α/β s číselnou hodnotou v legende
  * pevná sémantika farieb (FARBY nižšie) — tá istá vec má vždy tú istú farbu
  * ax.grid(alpha=0.3), plt.tight_layout(), legenda vždy
  * scatter surových dát sa preriedi (`point_downsample`), inak sa čiary stratia
  * výstupy do `vysledky_grafy/` s deterministickým názvom (bez medzier)
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn

# ----------------------------------------------------------------------
# Konvencie
# ----------------------------------------------------------------------
FARBY = {
    "true": "g",            # skutočné / bezšumové riešenie
    "data": "k",            # namerané (zašumené) body
    "analytic": "r",        # fit cez analytické riešenie
    "rk45": "b",            # fit cez RK45
    "metoda1": "tab:green",
    "metoda2": "tab:blue",
    "referencia": "k",      # vodorovná čiara "pravda"
    "gn": "tab:orange",     # dráha čistého Gauss-Newtona
    "gn_tlmeny": "tab:purple",  # dráha GN s polením kroku
}
STYLY = {"true": "-", "analytic": "--", "rk45": ":"}
VYSTUP = "vysledky_grafy"
GRID_ALPHA = 0.3
DPI = 150


def nastav_styl(kontext: str = "notebook"):
    """Zavolaj raz na začiatku notebooku/skriptu."""
    seaborn.set(style="whitegrid", context=kontext)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = GRID_ALPHA


def uloz_graf(fig, nazov: str, output_dir: str | None = VYSTUP, zavri: bool = False):
    """Deterministický názov, bez medzier a diakritických pascí v ceste."""
    if output_dir is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    nazov = nazov.replace(" ", "_").replace("/", "-")
    if not nazov.lower().endswith((".png", ".pdf", ".svg")):
        nazov += ".png"
    cesta = os.path.join(output_dir, nazov)
    fig.savefig(cesta, dpi=DPI)
    if zavri:
        plt.close(fig)
    print(f">>> graf uložený: {cesta}")
    return cesta


def _dokonci(fig, ax_or_axes, nazov_suboru, output_dir, show):
    fig.tight_layout()
    cesta = uloz_graf(fig, nazov_suboru, output_dir) if nazov_suboru else None
    if show:
        plt.show()
    else:
        plt.close(fig)
    return cesta


# ======================================================================
# 1. Základná kreslička (pôvodné `vykresli_graf`, rozšírené)
# ======================================================================
def vykresli_graf(t, curves: Iterable[tuple], title: str,
                  xlabel: str = "t", ylabel: str = "y(t)",
                  grid_alpha: float = GRID_ALPHA,
                  point_downsample: int | None = 20,
                  figsize=(10, 6), ax=None,
                  nazov_suboru: str | None = None,
                  output_dir: str | None = VYSTUP,
                  show: bool = True):
    """Vykreslí viacero kriviek naraz.

    curves = [(typ, y, opts), ...] kde typ ∈ {"line", "scatter"} a opts sú
    kwargs pre ax.plot / ax.scatter (color, lw, ls, label, s, alpha, ...).

    point_downsample preriedi LEN scatter — pri n=10000 by inak body prekryli
    všetky čiary a graf by nič nepovedal.
    """
    t = np.asarray(t)
    vlastna_fig = ax is None
    if vlastna_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for typ, y, opts in curves:
        y = np.asarray(y)
        if typ == "line":
            ax.plot(t, y, **opts)
        elif typ == "scatter":
            if point_downsample is None or point_downsample <= 1:
                ax.scatter(t, y, **opts)
            else:
                ax.scatter(t[::point_downsample], y[::point_downsample], **opts)
        else:
            raise ValueError(f"Neznámy typ krivky: {typ!r}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=grid_alpha)
    ax.legend()

    if vlastna_fig:
        return _dokonci(fig, ax, nazov_suboru, output_dir, show)
    return ax


# ======================================================================
# 2. Tri štandardné pohľady na jeden scenár
#    item = {"Dataset","Scenario","sigma","eps","t","y_true","y_obs",
#            "y_fit_exp","y_fit_rk","a_hat","b_hat","a_hat_rk","b_hat_rk"}
# ======================================================================
def viz_sum(item: Mapping, **kw):
    """Bezšumové riešenie vs. to, čo naozaj "vidíme". Prvý graf každej štúdie."""
    curves = [
        ("line", item["y_true"],
         {"color": FARBY["true"], "lw": 2, "label": "Bez šumu"}),
        ("scatter", item["y_obs"],
         {"color": FARBY["data"], "s": 15, "alpha": 0.4,
          "label": f"Zašumené dáta (σ={item['sigma']}, ε={item.get('eps', 0)})"}),
    ]
    return vykresli_graf(item["t"], curves,
                         f"{item['Dataset']} – {item['Scenario']} – Šum",
                         nazov_suboru=kw.pop("nazov_suboru",
                                             f"sum_{item['Dataset']}_{item['Scenario']}"),
                         **kw)


def viz_explicit_vs_numericke(item: Mapping, **kw):
    """Analytický fit vs. RK45 fit. Ak sa prekrývajú, numerika nič nekazí."""
    curves = [
        ("line", item["y_fit_exp"],
         {"color": FARBY["true"], "lw": 2, "label": "Analytický fit"}),
        ("line", item["y_fit_rk"],
         {"color": FARBY["rk45"], "lw": 2, "ls": "--", "label": "RK45 fit"}),
    ]
    return vykresli_graf(item["t"], curves,
                         f"{item['Dataset']} – {item['Scenario']} – Porovnanie fitov",
                         nazov_suboru=kw.pop("nazov_suboru",
                                             f"fity_{item['Dataset']}_{item['Scenario']}"),
                         **kw)


def viz_fitovanie(item: Mapping, **kw):
    """Pravda + dáta + oba fity v jednom. Hodnoty α̂, β̂ patria do legendy."""
    curves = [
        ("line", item["y_true"],
         {"color": FARBY["true"], "lw": 2, "label": "Skutočné riešenie"}),
        ("scatter", item["y_obs"],
         {"color": FARBY["data"], "s": 15, "alpha": 0.4, "label": "Zašumené dáta"}),
        ("line", item["y_fit_exp"],
         {"color": FARBY["analytic"], "lw": 2, "ls": STYLY["analytic"],
          "label": f"Analytický fit (α={item['a_hat']:.2f}, β={item['b_hat']:.2f})"}),
        ("line", item["y_fit_rk"],
         {"color": FARBY["rk45"], "lw": 2, "ls": STYLY["rk45"],
          "label": f"RK45 fit (α={item['a_hat_rk']:.2f}, β={item['b_hat_rk']:.2f})"}),
    ]
    return vykresli_graf(item["t"], curves,
                         f"{item['Dataset']} – {item['Scenario']} – Fitovanie parametrov",
                         nazov_suboru=kw.pop("nazov_suboru",
                                             f"fit_{item['Dataset']}_{item['Scenario']}"),
                         **kw)


# ======================================================================
# 3. Metrika cez faktor (σ, T, n) — jadro výsledkovej kapitoly
#    results_by_x[x][metoda][param][metrika]
# ======================================================================
def xy_plot_over_sigma_compare(results_by_sigma: Mapping, methods: Sequence[str],
                               param: str, metric: str,
                               dataset_label: str = "medium",
                               output_dir: str | None = VYSTUP,
                               with_errorbar: bool = True,
                               true_val: float | None = None,
                               xlabel: str = "σ (šum)",
                               logy: bool = False,
                               filename: str | None = None,
                               show: bool = True):
    """Viac metód v JEDNEJ osi. Použi, keď sa krivky neprekrývajú."""
    xs = sorted(results_by_sigma.keys())
    x = np.array(xs, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, method in enumerate(methods):
        y = np.array([results_by_sigma[s][method][param][metric] for s in xs], float)
        yerr = None
        if with_errorbar and metric == "mean":
            yerr = np.array([results_by_sigma[s][method][param]["std"] for s in xs], float)
        ls = "-" if i == 0 else "--"
        if yerr is None:
            ax.plot(x, y, marker="o", linestyle=ls, label=method)
        else:
            ax.errorbar(x, y, yerr=yerr, marker="o", linestyle=ls, capsize=4, label=method)

    if true_val is not None:
        ax.axhline(true_val, color=FARBY["referencia"], linestyle="--",
                   linewidth=1, label=f"{param} true")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{metric}({param})")
    ax.set_title(f"{dataset_label} | {', '.join(methods)} | {param} | {metric}")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend()

    if filename is None:
        filename = f"xy_{dataset_label}_{'-'.join(methods)}_{param}_{metric}"
    return _dokonci(fig, ax, filename, output_dir, show)


def xy_plot_over_sigma_compare_subplots(results_by_sigma: Mapping,
                                        methods: Sequence[str],
                                        param: str, metric: str,
                                        dataset_label: str = "medium",
                                        with_errorbar: bool = True,
                                        true_alpha: float | None = None,
                                        true_beta: float | None = None,
                                        output_dir: str | None = VYSTUP,
                                        xlabel: str = "σ (šum)",
                                        filename: str | None = None,
                                        show: bool = True):
    """Presne dve metódy pod sebou so zdieľanou osou x.

    Toto je verzia pre prípad, keď sa krivky v jednej osi úplne prekrývajú
    (typicky Analytický vs RK45) — vtedy je jedna os nečitateľná a dva
    podgrafy sú poctivejšie.
    """
    assert len(methods) == 2, "Táto verzia je pre presne dve metódy."
    xs = sorted(results_by_sigma.keys())
    x = np.array(xs, dtype=float)
    true_val = {"α": true_alpha, "β": true_beta}.get(param)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, method, farba in zip(axes, methods, [FARBY["metoda1"], FARBY["metoda2"]]):
        y = np.array([results_by_sigma[s][method][param][metric] for s in xs], float)
        yerr = None
        if with_errorbar and metric == "mean":
            yerr = np.array([results_by_sigma[s][method][param]["std"] for s in xs], float)
        if yerr is None:
            ax.plot(x, y, "o-", color=farba, label=method)
        else:
            ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-",
                        capsize=4, color=farba, label=method)
        if true_val is not None:
            ax.axhline(true_val, color=FARBY["referencia"], linestyle="--",
                       linewidth=1, label=f"{param} true")
        ax.set_ylabel(f"{metric}({param})")
        ax.grid(alpha=GRID_ALPHA)
        ax.legend()

    axes[0].set_title(f"{dataset_label} | {param} | {metric}")
    axes[1].set_xlabel(xlabel)

    if filename is None:
        filename = f"xysub_{dataset_label}_{'-'.join(methods)}_{param}_{metric}"
    return _dokonci(fig, axes, filename, output_dir, show)


# ======================================================================
# 4. Grafy špecifické pre Monte Carlo (vstup = mc_runs / mc_summary DataFrame)
# ======================================================================
def hist_odhadov(runs, param: str = "α", p_true: float | None = None,
                 metoda: str | None = None, scenar: str | None = None,
                 bins: int = 30, output_dir: str | None = VYSTUP,
                 filename: str | None = None, show: bool = True):
    """Rozdelenie MC odhadov jedného parametra + pravda + priemer.

    Sem sa pozeraj vždy, keď mean a median v tabuľke nesedia — histogram
    povie, či je to šikmosť, alebo pár utečených fitov na hranici boxu.
    """
    g = runs
    if metoda is not None:
        g = g[g["metoda"] == metoda]
    if scenar is not None:
        g = g[g["scenar"] == scenar]
    e = g[f"p_{param}"].to_numpy(dtype=float)
    e = e[np.isfinite(e)]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(e, bins=bins, color=FARBY["metoda2"], alpha=0.65,
            edgecolor="white", label=f"MC odhady ({len(e)} behov)")
    ax.axvline(np.mean(e), color=FARBY["analytic"], lw=2,
               label=f"priemer = {np.mean(e):.4f}")
    if p_true is not None:
        ax.axvline(p_true, color=FARBY["referencia"], ls="--", lw=1.5,
                   label=f"{param} true = {p_true}")
    ax.set_xlabel(f"{param}̂")
    ax.set_ylabel("počet behov")
    ax.set_title(f"Rozdelenie odhadov {param} | {metoda or 'všetky metódy'}"
                 + (f" | {scenar}" if scenar else ""))
    ax.grid(alpha=GRID_ALPHA)
    ax.legend()
    return _dokonci(fig, ax, filename or f"hist_{param}_{metoda}_{scenar}",
                    output_dir, show)


def oblak_parametrov(runs, p_names=("α", "β"), p_true=None,
                     metoda: str | None = None, scenar: str | None = None,
                     output_dir: str | None = VYSTUP,
                     filename: str | None = None, show: bool = True):
    """Oblak (α̂, β̂) — takto vyzerá korelácia parametrov na vlastné oči.

    Pretiahnutá elipsa = zle podmienený problém: dáta vedia určiť kombináciu
    parametrov, nie parametre samostatne. Číslo ρ v tabuľke je len jej sklon.
    """
    g = runs
    if metoda is not None:
        g = g[g["metoda"] == metoda]
    if scenar is not None:
        g = g[g["scenar"] == scenar]
    a = g[f"p_{p_names[0]}"].to_numpy(float)
    b = g[f"p_{p_names[1]}"].to_numpy(float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    r = np.corrcoef(a, b)[0, 1] if a.size > 1 else np.nan

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(a, b, s=18, alpha=0.5, color=FARBY["metoda2"],
               label=f"MC odhady (ρ = {r:.4f})")
    if p_true is not None:
        ax.scatter([p_true[0]], [p_true[1]], marker="*", s=220,
                   color=FARBY["analytic"], zorder=5,
                   label=f"pravda ({p_true[0]}, {p_true[1]})")
    ax.set_xlabel(f"{p_names[0]}̂")
    ax.set_ylabel(f"{p_names[1]}̂")
    ax.set_title(f"Korelačný oblak {p_names[0]}–{p_names[1]}"
                 + (f" | {metoda}" if metoda else "")
                 + (f" | {scenar}" if scenar else ""))
    ax.grid(alpha=GRID_ALPHA)
    ax.legend()
    return _dokonci(fig, ax, filename or f"oblak_{metoda}_{scenar}",
                    output_dir, show)


def graf_pokrytia(summary, x_stlpec: str = "T", metoda: str | None = None,
                  dataset: str | None = None, output_dir: str | None = VYSTUP,
                  filename: str | None = None, show: bool = True):
    """Empirické pokrytie 95 % intervalov cez faktor. Cieľová čiara = 0.95.

    Bod výrazne pod 0.95 znamená, že lokálna neistota klame — tam patrí
    profilová vierohodnosť, nie interval p̂ ± 1.96·SE.
    """
    g = summary
    if metoda is not None:
        g = g[g["metoda"] == metoda]
    if dataset is not None:
        g = g[g["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (param, gp) in enumerate(g.groupby("param")):
        gp = gp.sort_values(x_stlpec)
        ax.plot(gp[x_stlpec], gp["pokrytie"], marker="o",
                linestyle="-" if i == 0 else "--", label=f"pokrytie {param}")
    ax.axhline(0.95, color=FARBY["referencia"], ls="--", lw=1, label="nominál 0.95")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(x_stlpec)
    ax.set_ylabel("empirické pokrytie")
    ax.set_title(f"Pokrytie 95 % intervalov | {metoda or 'všetky metódy'}")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend()
    return _dokonci(fig, ax, filename or f"pokrytie_{metoda}_{dataset}",
                    output_dir, show)


def metrika_cez_faktor(summary, x_stlpec: str, metric: str = "rmse",
                       param: str = "α", metody: Sequence[str] | None = None,
                       dataset: str | None = None, logy: bool = True,
                       output_dir: str | None = VYSTUP,
                       filename: str | None = None, show: bool = True):
    """Priamo z mc_summary: metrika (rmse/std/bias/mean_se) cez T, σ alebo n.

    logy=True je predvolené, lebo RMSE cez T alebo n padá o rády — v lineárnej
    škále by celý priebeh splynul s nulou.
    """
    g = summary[summary["param"] == param]
    if dataset is not None:
        g = g[g["dataset"] == dataset]
    metody = metody if metody is not None else sorted(g["metoda"].unique())

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, m in enumerate(metody):
        gm = g[g["metoda"] == m].sort_values(x_stlpec)
        ax.plot(gm[x_stlpec], gm[metric], marker="o",
                linestyle="-" if i == 0 else "--", label=m)
    if logy:
        ax.set_yscale("log")
        ax.set_xscale("log" if x_stlpec == "n" else "linear")
    ax.set_xlabel({"T": "T (dĺžka intervalu)", "sigma": "σ (šum)",
                   "n": "n (počet meraní)"}.get(x_stlpec, x_stlpec))
    ax.set_ylabel(f"{metric}({param})")
    ax.set_title(f"{metric}({param}) cez {x_stlpec}"
                 + (f" | {dataset}" if dataset else ""))
    ax.grid(alpha=GRID_ALPHA, which="both")
    ax.legend()
    return _dokonci(fig, ax, filename or f"{metric}_{param}_cez_{x_stlpec}_{dataset}",
                    output_dir, show)


# ======================================================================
# 5. Pomôcka: mc_summary -> vnorený slovník pre xy_plot_* funkcie
# ======================================================================
def summary_na_slovnik(summary, x_stlpec: str = "sigma", dataset: str | None = None):
    """results[x][metoda][param][metrika] — formát, ktorý čakajú xy_plot_* funkcie."""
    g = summary if dataset is None else summary[summary["dataset"] == dataset]
    out: dict = {}
    metriky = ["mean", "std", "median", "iqr", "rmse", "bias", "mean_se",
               "se_pomer", "pokrytie"]
    for _, r in g.iterrows():
        x = r[x_stlpec]
        out.setdefault(x, {}).setdefault(r["metoda"], {})[r["param"]] = {
            m: r[m] for m in metriky if m in r
        }
    return out


# ======================================================================
# 6. Gauss-Newton: ako sa optimalizácia hýbe
#    Vstup = GNVysledok z gauss_newton.py (historia_p, historia_S, ...).
# ======================================================================
def _farba_gn(i):
    return [FARBY["gn"], FARBY["gn_tlmeny"], FARBY["metoda1"], FARBY["metoda2"]][i % 4]


def gn_cesta_na_konturach(A, B, S, drahy: Mapping, p_true=None,
                          p_names=("α", "β"), title: str = "Dráha Gauss-Newtona",
                          output_dir: str | None = VYSTUP,
                          filename: str = "gn_cesta_kontury", show: bool = True):
    """Vrstevnice log10 S(α, β) + dráha iterácií p_0 → p_K.

    A, B, S: mriežka (np.meshgrid) a hodnoty súčtu štvorcov rezíduí.
    drahy = {"Čistý GN": vysledok, "GN s polením kroku": vysledok2, ...}

    Pretiahnuté, šikmé údolie = korelované parametre; GN po ňom skáče
    tým lepšie, čím je model v okolí minima „lineárnejší".
    """
    fig, ax = plt.subplots(figsize=(6.5, 6))
    logS = np.log10(np.where(S > 0, S, np.nan))
    cf = ax.contourf(A, B, logS, levels=30, cmap="Greys_r", alpha=0.85)
    ax.contour(A, B, logS, levels=15, colors="white", linewidths=0.5, alpha=0.6)
    fig.colorbar(cf, ax=ax, label="log₁₀ S(α, β)")

    for i, (label, res) in enumerate(drahy.items()):
        P = res.P
        farba = _farba_gn(i)
        ax.plot(P[:, 0], P[:, 1], "-o", color=farba, lw=1.8, ms=5,
                label=f"{label} ({res.iteracie} it., "
                      f"{p_names[0]}={res.p[0]:.2f}, {p_names[1]}={res.p[1]:.2f})")
        ax.scatter(P[0, 0], P[0, 1], marker="s", s=70, color=farba,
                   edgecolor="k", zorder=6)
        for k in range(min(len(P), 6)):
            ax.annotate(str(k), P[k], textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color=farba)
    if p_true is not None:
        ax.scatter([p_true[0]], [p_true[1]], marker="*", s=220,
                   color=FARBY["true"], edgecolor="k", zorder=7,
                   label=f"pravda ({p_true[0]}, {p_true[1]})")

    ax.set_xlim(A.min(), A.max())
    ax.set_ylim(B.min(), B.max())
    ax.set_xlabel(p_names[0])
    ax.set_ylabel(p_names[1])
    ax.set_title(title)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(fontsize=8, loc="best")
    return _dokonci(fig, ax, filename, output_dir, show)


def gn_konvergencia(drahy: Mapping, title: str = "Konvergencia Gauss-Newtona",
                    output_dir: str | None = VYSTUP,
                    filename: str = "gn_konvergencia", show: bool = True):
    """Dva podgrafy so zdieľanou osou x: S(p_k) a ||Δp_k|| cez iteráciu k.

    Kvadratická (resp. rýchla lineárna pri malých rezíduách) konvergencia
    GN je v logaritmickej škále vidieť ako zrýchľujúci sa pád ||Δp||.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for i, (label, res) in enumerate(drahy.items()):
        farba = _farba_gn(i)
        k = np.arange(len(res.historia_S))
        axes[0].plot(k, res.historia_S, "o-", color=farba, label=label)
        kk = np.arange(1, len(res.historia_krok) + 1)
        axes[1].plot(kk, res.historia_krok, "o-", color=farba, label=label)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("S(p_k) = ‖r‖²")
    axes[1].set_ylabel("‖Δp_k‖")
    axes[1].set_xlabel("iterácia k")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(alpha=GRID_ALPHA, which="both")
        ax.legend()
    return _dokonci(fig, axes, filename, output_dir, show)


def gn_fit_iteracie(t, y_obs, y_true, res, predikcia, max_kriviek: int = 8,
                    p_names=("α", "β"), point_downsample: int | None = 20,
                    title: str = "Fit počas iterácií Gauss-Newtona",
                    output_dir: str | None = VYSTUP,
                    filename: str = "gn_fit_iteracie", show: bool = True):
    """Dáta + pravda + krivka y(t; p_k) pre prvých `max_kriviek` iterácií.

    Farba ide od svetlej (štart) po tmavú (posledná iterácia), takže je
    vidieť, ako sa model „priťahuje" k dátam.
    """
    t = np.asarray(t)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, y_true, color=FARBY["true"], lw=2.5, label="Skutočné riešenie")
    sl = slice(None, None, point_downsample if point_downsample and point_downsample > 1 else None)
    ax.scatter(t[sl], np.asarray(y_obs)[sl], color=FARBY["data"], s=15,
               alpha=0.4, label="Zašumené dáta")

    P = res.P
    idx = list(range(min(len(P), max_kriviek)))
    if len(P) - 1 not in idx:
        idx.append(len(P) - 1)
    cmap = plt.get_cmap("Oranges")
    for j, k in enumerate(idx):
        farba = cmap(0.3 + 0.7 * j / max(len(idx) - 1, 1))
        ax.plot(t, predikcia(t, P[k]), color=farba, lw=1.6,
                ls="-" if k == len(P) - 1 else "--",
                label=f"k={k} ({p_names[0]}={P[k][0]:.2f}, {p_names[1]}={P[k][1]:.2f})")

    # prvé iterácie môžu byť o rády mimo — os y držíme pri dátach
    lo, hi = np.nanmin(y_obs), np.nanmax(y_obs)
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.set_title(title)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(fontsize=8, ncol=2)
    return _dokonci(fig, ax, filename, output_dir, show)
