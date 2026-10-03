"""
Gauss-Newton pre odhad parametrov ODR — jednoduchá streľba (single shooting).

Problém:  min_p  S(p) = ||r(p)||²,   r_i(p) = y(t_i; p) - y_obs_i,
kde y(t; p) je riešenie  y' = f(t, y, p),  y(t0) = y0.

Jakobián  J_ij = ∂y(t_i; p)/∂p_j  sa počíta cez rovnice citlivosti
(variačné rovnice), integrované spolu so stavom:

    s_j' = f_y · s_j + f_{p_j},    s_j(t0) = 0       (y0 nezávisí od p)

Krok GN:  Δp = argmin ||J Δp + r||  (lstsq — nikdy sa nezostavuje JᵀJ,
jej podmienenosť je druhá mocnina podmienenosti J).

Viacnásobná streľba (multiple shooting) sem zatiaľ nepatrí; rozhranie
`gauss_newton(rezidua_a_jakobian, p0, ...)` je však všeobecné, takže sa
neskôr dá podstrčiť aj rezíduum/Jakobián z multiple shooting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------
# Model pre GN: pravá strana + jej derivácie podľa y a p
# ----------------------------------------------------------------------
@dataclass
class OdrModel:
    nazov: str
    f: Callable          # f(t, y, p)    -> y'
    f_y: Callable        # f_y(t, y, p)  -> ∂f/∂y
    f_p: Callable        # f_p(t, y, p)  -> [∂f/∂p_1, ..., ∂f/∂p_m]
    y0: float = 1.0
    t0: float = 0.0
    p_names: tuple = ("α", "β")


def model_linearny_gn(y0=1.0):
    """y' = α y + β t."""
    return OdrModel(
        nazov="Lineárny",
        f=lambda t, y, p: p[0] * y + p[1] * t,
        f_y=lambda t, y, p: p[0],
        f_p=lambda t, y, p: np.array([y, t]),
        y0=y0,
    )


def model_bernoulli_gn(y0=1.0):
    """y' = α y + β t y²."""
    return OdrModel(
        nazov="Bernoulliho",
        f=lambda t, y, p: p[0] * y + p[1] * t * y**2,
        f_y=lambda t, y, p: p[0] + 2.0 * p[1] * t * y,
        f_p=lambda t, y, p: np.array([y, t * y**2]),
        y0=y0,
    )


# ----------------------------------------------------------------------
# Single shooting: y(t_i; p) a J = ∂y/∂p cez rovnice citlivosti
# ----------------------------------------------------------------------
def simuluj_s_citlivostami(model: OdrModel, t, p, rtol=1e-8, atol=1e-11):
    """Vráti (y, J) na časoch t; J má tvar (n, m). Pri zlyhaní (NaN, NaN)."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    m = p.size

    def rhs(tt, z):
        y, s = z[0], z[1:]
        return np.concatenate(([model.f(tt, y, p)],
                               model.f_y(tt, y, p) * s + model.f_p(tt, y, p)))

    z0 = np.concatenate(([model.y0], np.zeros(m)))
    # t musí byť zoradené a t[0] >= t0
    sol = solve_ivp(rhs, (model.t0, t[-1]), z0, t_eval=t,
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape[1] != t.size:
        return np.full(t.size, np.nan), np.full((t.size, m), np.nan)
    return sol.y[0], sol.y[1:].T


def single_shooting(model: OdrModel, t, y_obs):
    """Uzáver p -> (r, J) pre gauss_newton."""
    y_obs = np.asarray(y_obs, dtype=float)

    def rezidua_a_jakobian(p):
        y, J = simuluj_s_citlivostami(model, t, p)
        return y - y_obs, J

    return rezidua_a_jakobian


# ----------------------------------------------------------------------
# Gauss-Newton so záznamom histórie (na vizualizáciu)
# ----------------------------------------------------------------------
@dataclass
class GNVysledok:
    p: np.ndarray
    uspech: bool
    sprava: str
    historia_p: list = field(default_factory=list)     # p_0, p_1, ..., p_K
    historia_S: list = field(default_factory=list)     # S(p_k) = ||r||²
    historia_krok: list = field(default_factory=list)  # ||Δp_k|| (skutočne urobený)
    historia_lambda: list = field(default_factory=list)  # dĺžka kroku (1 = plný GN)
    historia_r: list = field(default_factory=list)     # rezíduá r(p_k)

    @property
    def iteracie(self):
        return len(self.historia_p) - 1

    @property
    def P(self):
        return np.array(self.historia_p)


def gauss_newton(rezidua_a_jakobian, p0, max_iter=50, tol_krok=1e-10,
                 tol_S=1e-14, tlmenie=False, min_lambda=1e-6, bounds=None):
    """Gauss-Newton.

    tlmenie=False: čistý GN, vždy plný krok λ = 1.
    tlmenie=True:  polenie kroku λ ∈ {1, 1/2, 1/4, ...}, kým S neklesne.
    bounds: voliteľne ((lo_1, ..., lo_m), (hi_1, ..., hi_m)) — krok sa orezáva
            do boxu, aby sa v zlom štarte neodišlo do pretečenia exp(α t).
    """
    p = np.asarray(p0, dtype=float).copy()
    r, J = rezidua_a_jakobian(p)
    S = float(r @ r)
    res = GNVysledok(p=p, uspech=False, sprava="")
    res.historia_p.append(p.copy())
    res.historia_S.append(S)
    res.historia_r.append(r.copy())

    if not np.isfinite(S):
        res.sprava = "Rezíduá v štartovacom bode nie sú konečné."
        return res

    for _ in range(max_iter):
        dp, *_ = np.linalg.lstsq(J, -r, rcond=None)

        lam = 1.0
        while True:
            p_new = p + lam * dp
            if bounds is not None:
                p_new = np.clip(p_new, bounds[0], bounds[1])
            r_new, J_new = rezidua_a_jakobian(p_new)
            S_new = float(r_new @ r_new)
            if not tlmenie and np.isfinite(S_new):
                break
            if np.isfinite(S_new) and S_new < S:
                break
            lam *= 0.5
            if lam < min_lambda:
                res.p = p
                res.sprava = "Polenie kroku zlyhalo (S neklesá ani pre malé λ)."
                return res

        krok = float(np.linalg.norm(p_new - p))
        p, r, J, S_old, S = p_new, r_new, J_new, S, S_new
        res.historia_p.append(p.copy())
        res.historia_S.append(S)
        res.historia_krok.append(krok)
        res.historia_lambda.append(lam)
        res.historia_r.append(r.copy())

        if krok <= tol_krok * (1.0 + np.linalg.norm(p)):
            res.uspech, res.sprava = True, "Konvergencia: ||Δp|| pod toleranciou."
            break
        if abs(S_old - S) <= tol_S * (1.0 + S):
            res.uspech, res.sprava = True, "Konvergencia: S sa už nemení."
            break
    else:
        res.sprava = f"Dosiahnutý max_iter = {max_iter}."

    res.p = p
    return res
