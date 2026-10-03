"""
Referenčné modely diplomovky: lineárna rovnica a Bernoulliho rovnica.

Každý model poskytuje DVE predikcie toho istého riešenia:
  - analytickú (uzavretý vzorec),
  - numerickú (RK45 cez solve_ivp).
Obe majú rovnaké rozhranie  predikcia(t, p) -> y,  aby ich Monte Carlo
engine mohol porovnávať ako dva "estimátory" na tých istých dátach.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ----------------------------------------------------------------------
# Spoločný kontajner
# ----------------------------------------------------------------------
@dataclass
class Model:
    """Model = pravda (p_true, y0) + jedna alebo viac predikčných funkcií."""
    nazov: str
    p_true: np.ndarray
    y0: float
    p_names: Sequence[str]
    predikcie: dict = field(default_factory=dict)   # {"Analytický": fn, "RK45": fn}
    t0: float = 0.0

    def __post_init__(self):
        self.p_true = np.asarray(self.p_true, dtype=float)


# ----------------------------------------------------------------------
# 1) Lineárna rovnica:  y' = a*y + b*t,  y(t0) = y0
# ----------------------------------------------------------------------
def linearna_analyticka(t, p, y0=1.0, t0=0.0, eps=1e-10):
    """y' = a*y + b*t. Pre a -> 0 prechádza na presné riešenie y0 + b(t^2-t0^2)/2."""
    a, b = float(p[0]), float(p[1])
    t = np.asarray(t, dtype=float)
    if abs(a) < eps:
        return y0 + 0.5 * b * (t**2 - t0**2)
    C = (y0 + (b / a) * t0 + b / a**2) * np.exp(-a * t0)
    return -(b / a) * t - b / a**2 + C * np.exp(a * t)


def linearna_rk45(t, p, y0=1.0, t0=0.0, rtol=1e-8, atol=1e-11):
    a, b = float(p[0]), float(p[1])
    return _integruj(lambda tt, y: [a * y[0] + b * tt], t, y0, t0, rtol, atol)


# ----------------------------------------------------------------------
# 2) Bernoulliho rovnica:  y' = a*y + b*t*y^2,  y(t0) = y0
#    Substitúcia u = 1/y dáva lineárnu u' = -a*u - b*t, teda
#    u(t) = C*exp(-a t) - (b/a) t + b/a^2,  C = 1/y0 - b/a^2,
#    y(t) = a^2 / ( b - a*b*t + C*a^2*exp(-a t) ).
#    Riešenie exploduje v čase t*, kde u(t*) = 0.
# ----------------------------------------------------------------------
def bernoulli_analyticka(t, p, y0=1.0, t0=0.0, eps=1e-10):
    a, b = float(p[0]), float(p[1])
    if abs(a) < eps:
        a = eps
    t = np.asarray(t, dtype=float)
    C = 1.0 / y0 - b / a**2
    menovatel = b - a * b * t + C * a**2 * np.exp(-a * t)
    # ochrana proti deleniu nulou tesne pri singularite
    menovatel = np.where(np.abs(menovatel) < 1e-10, 1e-10, menovatel)
    return a**2 / menovatel


def bernoulli_rk45(t, p, y0=1.0, t0=0.0, rtol=1e-8, atol=1e-11):
    a, b = float(p[0]), float(p[1])
    return _integruj(lambda tt, y: [a * y[0] + b * tt * y[0] ** 2], t, y0, t0, rtol, atol)


def bernoulli_singularita(p, y0=1.0, t_max_hladania=50.0):
    """Vráti t* (prvý kladný koreň u(t)=0), alebo np.inf ak žiadny nie je.

    T_max experimentu volíme bezpečne pod ním, v DP 0.8*t*.
    """
    a, b = float(p[0]), float(p[1])
    C = 1.0 / y0 - b / a**2
    u = lambda t: C * np.exp(-a * t) - (b / a) * t + b / a**2
    mriezka = np.linspace(1e-9, t_max_hladania, 20001)
    hod = u(mriezka)
    znamienko = np.where(np.diff(np.sign(hod)) != 0)[0]
    if len(znamienko) == 0:
        return np.inf
    i = znamienko[0]
    return float(brentq(u, mriezka[i], mriezka[i + 1], xtol=1e-14))


# ----------------------------------------------------------------------
# Spoločný integrátor: rieši na zoradených časoch, vracia v pôvodnom poradí
# ----------------------------------------------------------------------
def _integruj(prava_strana, t_vals, y0, t0, rtol, atol):
    t_vals = np.asarray(t_vals, dtype=float)
    idx = np.argsort(t_vals)
    t_sorted = t_vals[idx]
    if t_sorted[0] < t0:
        raise ValueError("t_vals obsahuje časy < t0, ale y0 je zadané pre čas t0.")

    prida_t0 = t_sorted[0] != t0
    t_eval = np.concatenate(([t0], t_sorted)) if prida_t0 else t_sorted

    sol = solve_ivp(prava_strana, (t0, t_eval[-1]), [y0], t_eval=t_eval,
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape[1] != len(t_eval):
        # Neúspech signalizujeme NaN — MC engine ten beh označí ako zlyhaný.
        return np.full_like(t_vals, np.nan, dtype=float)

    y_sorted = sol.y[0][1:] if prida_t0 else sol.y[0]
    y_out = np.empty_like(t_vals, dtype=float)
    y_out[idx] = y_sorted
    return y_out


# ----------------------------------------------------------------------
# Hotové inštancie s hodnotami z DP
# ----------------------------------------------------------------------
def model_linearny(p_true=(3.0, 9.0), y0=1.0):
    return Model(
        nazov="Lineárny",
        p_true=np.asarray(p_true, float),
        y0=y0,
        p_names=("α", "β"),
        predikcie={
            "Analytický": lambda t, p: linearna_analyticka(t, p, y0=y0),
            "RK45":       lambda t, p: linearna_rk45(t, p, y0=y0),
        },
    )


def model_bernoulli(p_true=(1.0, 2.0), y0=1.0):
    return Model(
        nazov="Bernoulliho",
        p_true=np.asarray(p_true, float),
        y0=y0,
        p_names=("α", "β"),
        predikcie={
            "Analytický": lambda t, p: bernoulli_analyticka(t, p, y0=y0),
            "RK45":       lambda t, p: bernoulli_rk45(t, p, y0=y0),
        },
    )
