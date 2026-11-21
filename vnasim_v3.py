#!/usr/bin/env python3
"""
Quasi-optics bench: Gaussian-beam and S11 simulator for corrugated-waveguide THz setups.

Front end (default, as built): sender horn -> OAP1 -> beamsplitter (transmit) -> OAP2 -> beam path;
the return beam reflects off the beamsplitter -> OAP3 -> receiver horn, giving S21. Alternatively a
single corrugated waveguide sends and receives (S11). The beam path holds flat mirrors, OAPs,
samples and cryostats (window + gap + sample) on an x-y table; the last component reflects the
beam back.
Build the beam path in the left panel, choose a frequency band, press Simulate.

Outputs: setup diagram, beam profile (w along the path), S11 magnitude/phase, and a table
of numbers. File menu: open/save setups (JSON), export S11 (CSV), save figures (PNG).

Requirements: Python 3.8+, numpy, matplotlib (tkinter ships with Python).
Run:          python gbsim_gui.py

Physics (fundamental Gaussian mode, paraxial):
  * HE11 aperture field <-> Gaussian with w0 = 0.6435 a (98 % power coupling each way)
  * complex-q / ABCD propagation with Gouy phase; OAP = thin lens f = EFL; slab = [[1, d/n],[0,1]]
  * thin-film characteristic matrices for samples, backings (metals, ITO film on glass,
    dielectrics, or any eps(w) you type) and mirror losses
    (Drude metals, real angle of incidence and polarization)
  * every bounce path between in-line samples and the end reflector is traced with its own q
    and summed coherently; each returning beam is projected onto the waveguide mode
Conventions: user eps in physics convention (eps'' > 0 = loss); S11 in VNA convention exp(+j w t).
Not modelled: OAP aberrations / cross-polarization, higher-order modes, flange reflections.
"""
import copy
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# =============================================================================================
#  Physics engine
# =============================================================================================
C0 = 299792458.0
EPS0 = 8.8541878128e-12
MM = 1e-3
HE11_RATIO = 0.6435
X01 = 2.404825557695773

METALS = {"gold": (4.10e7, 2.5e-14), "copper": (5.96e7, 2.7e-14), "aluminum": (3.77e7, 0.8e-14)}
BANDS = {"B1": (70.0, 170.0), "B2": (170.0, 300.0), "B3": (300.0, 450.0),
         "B4": (450.0, 600.0), "B5": (600.0, 700.0)}
F_MIN, F_MAX, MAX_SPAN = 70.0, 700.0, 200.0


def eps_of(spec, f):
    """Dielectric function (physics convention) of a material spec dict at f [GHz] (array)."""
    f = np.asarray(f, float)
    kind = spec["kind"]
    if kind == "vacuum":
        return np.ones_like(f, dtype=complex)
    if kind == "const":
        return np.full(f.shape, complex(float(spec["re"]), float(spec["im"])))
    if kind == "lorentz":
        f0 = float(spec["f0"])
        return float(spec["epsInf"]) + float(spec["de"]) * f0**2 / (f0**2 - f**2 - 1j * float(spec["gamma"]) * f)
    if kind == "metal":
        sigma, tau = METALS[spec["metal"]] if spec["metal"] != "custom" else (float(spec["sigma"]), float(spec["tau"]))
        return drude_eps(sigma, tau, f)
    if kind == "ito":
        return drude_eps(float(spec["sigma"]), float(spec["tau_fs"]) * 1e-15, f)
    if kind == "expr":
        return eval_eps_expr(str(spec["expr"]), f)
    raise ValueError(f"unknown material kind {kind}")


def drude_eps(sigma, tau, f):
    """Drude conductor from DC conductivity sigma [S/m] and relaxation time tau [s]."""
    w = 2 * np.pi * np.asarray(f, float) * 1e9
    return 1 + 1j * (sigma / (1 - 1j * w * tau)) / (EPS0 * w)


# Names usable in a user-written dielectric function eps(w).
EXPR_NAMES = {
    "pi": np.pi, "j": 1j, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "arctan": np.arctan, "abs": np.abs, "real": np.real, "imag": np.imag, "where": np.where,
    "eps0": EPS0, "c": C0, "hbar": 1.054571817e-34, "e": 1.602176634e-19, "me": 9.1093837015e-31,
}


def eval_eps_expr(expr, f):
    """Evaluate a user expression for eps (physics convention, +i eps'' = loss).
    Variables: w = angular frequency [rad/s], f = frequency [GHz], f_hz = frequency [Hz],
    THz = frequency [THz]. Python syntax, 1j or j for the imaginary unit, numpy functions."""
    f = np.asarray(f, float)
    names = dict(EXPR_NAMES, w=2 * np.pi * f * 1e9, omega=2 * np.pi * f * 1e9, f=f, f_hz=f * 1e9, THz=f * 1e-3)
    try:
        val = eval(compile(expr, "<eps(w)>", "eval"), {"__builtins__": {}}, names)
    except Exception as err:
        raise ValueError(f"The dielectric function '{expr}' could not be evaluated: {err}")
    val = np.broadcast_to(np.asarray(val, dtype=complex), f.shape).copy()
    if not np.all(np.isfinite(val)):
        raise ValueError(f"The dielectric function '{expr}' is not finite at every frequency in the band.")
    return val


def backing_stack(b, f):
    """Extra layers (N list, d list in m) and the final semi-infinite index behind a sample."""
    if b["kind"] != "ito":
        return [], [], n_of(eps_of(b, f))
    Ns, ds = [], []
    t_nm = b.get("thickness_nm", "")
    film = n_of(eps_of(b, f))
    if t_nm in ("", None):                      # bulk ITO
        return Ns, ds, film
    Ns.append(film); ds.append(float(t_nm) * 1e-9)
    sub = n_of(eps_of({"kind": "const", "re": b["sub_re"], "im": b["sub_im"]}, f))
    t_sub = b.get("sub_thickness", "")
    if t_sub in ("", None):                     # semi-infinite substrate
        return Ns, ds, sub
    Ns.append(sub); ds.append(float(t_sub) * MM)
    return Ns, ds, np.ones_like(sub)


# Approximate THz dielectric constants for quick-fill presets (physics convention).
PRESETS = {
    "Quartz, crystalline (ordinary ray)": (4.44, 0.007),
    "Fused silica (amorphous quartz)": (3.81, 0.04),
    "Sapphire (ordinary ray)": (9.4, 0.01),
    "High-resistivity silicon": (11.67, 0.0005),
}
QUARTZ = {"kind": "const", "re": 4.44, "im": 0.007}
ITO_DEFAULT = {"kind": "ito", "sigma": 5e5, "tau_fs": 6, "thickness_nm": 150,
               "sub_re": 6.5, "sub_im": 0.5, "sub_thickness": 1.1}


def n_of(eps):
    """Engineering-convention refractive index N = n - j kappa (passive)."""
    N = np.sqrt(np.conj(eps))
    return np.abs(N.real) - 1j * np.abs(N.imag)


def tmm(layers_N, layers_d, N_sub, k0, theta=0.0, pol="s"):
    """Macleod characteristic-matrix r, t of a stack in vacuum. Arrays over frequency."""
    k0 = np.atleast_1d(np.asarray(k0, float))
    shp = k0.shape
    s2 = np.sin(theta) ** 2

    def adm(N):
        N = np.broadcast_to(np.asarray(N, complex), shp)
        Nc = np.sqrt(N**2 - s2)
        Nc = np.where(Nc.imag > 0, -Nc, Nc)
        return Nc, (Nc if pol == "s" else N**2 / Nc)

    _, eta0 = adm(1.0)
    _, etas = adm(N_sub)
    M11 = np.ones(shp, complex); M22 = np.ones(shp, complex)
    M12 = np.zeros(shp, complex); M21 = np.zeros(shp, complex)
    for N, d in zip(layers_N, layers_d):
        Nc, eta = adm(N)
        delta = k0 * Nc * d
        c, sn = np.cos(delta), np.sin(delta)
        m12, m21 = 1j * sn / eta, 1j * eta * sn
        M11, M12, M21, M22 = (M11 * c + M12 * m21, M11 * m12 + M12 * c,
                              M21 * c + M22 * m21, M21 * m12 + M22 * c)
    B = M11 + M12 * etas
    Cc = M21 + M22 * etas
    den = eta0 * B + Cc
    return (eta0 * B - Cc) / den, 2 * eta0 / den


def q_waist(w0, lam):
    return 1j * np.pi * w0**2 / lam


def w_of_q(q, lam):
    return np.sqrt(-lam / (np.pi * np.imag(1.0 / q)))


def abcd(q, A, B, C, D):
    """Return (q_out, phase factor of the normalised mode = Gouy term)."""
    return (A * q + B) / (C * q + D), np.exp(-1j * np.angle(A + B / q))


def clip_factor(q, lam, diameter_m):
    if not diameter_m:
        return 1.0
    w = w_of_q(q, lam)
    return np.sqrt(1.0 - np.exp(-2.0 * (diameter_m / 2.0) ** 2 / w**2))


def aperture_T(q, lam, diameter_m):
    """Fundamental-mode amplitude after a circular aperture: <G| circ |G> = 1 - exp(-2 a^2 / w^2).
    (The truncated part of the field is scattered into higher-order modes that do not couple.)"""
    if not diameter_m:
        return 1.0
    w = w_of_q(q, lam)
    return 1.0 - np.exp(-2.0 * (diameter_m / 2.0) ** 2 / w**2)


def gaussian_overlap(q, lam, w0):
    k = 2 * np.pi / lam
    w = w_of_q(q, lam)
    return 2.0 / (w0 * w * (1.0 / w0**2 + 1j * k / (2.0 * q)))


_GLX, _GLW = np.polynomial.legendre.leggauss(64)


def _j0(x):
    t = (_GLX + 1) * np.pi / 2
    return (np.cos(np.multiply.outer(x, np.sin(t))) @ _GLW) / 2


def he11_overlap(q, lam, a):
    """<HE11 aperture field | normalised Gaussian(q)>, arrays over frequency."""
    r = (_GLX + 1) * a / 2
    wr = _GLW * a / 2 * 2 * np.pi * r
    e = _j0(X01 * r / a)
    e = e / np.sqrt(np.sum(wr * e**2))
    q = np.atleast_1d(q)
    lam = np.broadcast_to(lam, q.shape)
    w = w_of_q(q, lam)
    k = 2 * np.pi / lam
    g = np.sqrt(2 / np.pi) / w[:, None] * np.exp(-1j * k[:, None] * r[None, :] ** 2 / (2 * q[:, None]))
    return g @ (wr * e)


def validate_band(f1, f2):
    if f2 <= f1:
        return "The stop frequency must be above the start."
    if f1 < F_MIN or f2 > F_MAX:
        return f"Bands must lie inside {F_MIN:g}-{F_MAX:g} GHz."
    if f2 - f1 > MAX_SPAN:
        return f"This band is {f2 - f1:g} GHz wide. Sweeps are limited to {MAX_SPAN:g} GHz; split it into two bands."
    return None


def _num(c, key, name, positive=False, optional=False):
    v = c.get(key, "")
    if v in ("", None):
        if optional:
            return None
        raise ValueError(f"{name}: enter a value for '{key}'.")
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: '{key}' must be a number (got {v!r}).")
    if positive and v <= 0:
        raise ValueError(f"{name}: '{key}' must be positive.")
    return v


FRONTENDS = ["Sender → beamsplitter → receiver (as built, S21)", "Single corrugated waveguide (S11)"]
BS_MODELS = ["Fixed R and T", "Dielectric slab (R, T vary with frequency)"]
RX_MODES = ["Direct: OAP3 → receiver", "Confocal: OAP3 → aperture → OAP4 → OAP5 → receiver"]

FRONTEND_DEFAULT = {
    # Lab layout (Nov 2026): 10 mm corrugated horns; OAP1 EFL 250 mm with the sender at its focus;
    # BS 125 mm after OAP1; OAP2 (EFL 125) 250 mm after the BS, beam path starts at its focus.
    # Return arm, confocal: the BS reflects toward -y onto OAP3 (EFL 125), which focuses onto a
    # spatial-filter aperture; OAP4 (EFL 125) recollimates and OAP5 (EFL 250) focuses into the
    # receiver. f5/f4 = 2 restores the 10 mm horn's waist (the OAP3 focus is half its size).
    # Angles are signed (deg, -90..90), measured counter-clockwise seen from above:
    #   bs_angle  = angle of incidence; + reflects the return beam toward +y, - toward -y
    #   oapN_angle = how far OAP N turns the beam (its off-axis angle); + = left (CCW), - = right (CW)
    "kind": "bs",
    "sender_D": 10.0, "receiver_D": 10.0, "model": "he11",
    "d_sender": 250.0,                                      # sender aperture -> OAP1 (= EFL1: collimated)
    "oap1_efl": 250.0, "oap1_D": 76.2, "oap1_mat": "gold", "oap1_angle": 90.0,
    "d1": 125.0,                                            # OAP1 -> beamsplitter
    "bs_angle": -45.0, "bs_D": 76.2, "bs_model": "fixed", "bs_R": 50.0, "bs_T": 50.0,
    "bs_thickness": 0.5, "bs_eps": {"kind": "const", "re": 11.67, "im": 0.0005},
    "d2": 250.0,                                            # beamsplitter -> OAP2
    "oap2_efl": 125.0, "oap2_D": 76.2, "oap2_mat": "gold", "oap2_angle": -90.0,
    "rx_mode": "confocal",                                  # "direct" (OAP3 -> receiver) or "confocal"
    "d3": 125.0,                                            # beamsplitter -> OAP3
    "oap3_efl": 125.0, "oap3_D": 76.2, "oap3_mat": "gold", "oap3_angle": -90.0,
    "d_ap": 125.0, "ap_D": 8.0,                             # OAP3 -> aperture (= EFL3: at the focus), aperture diameter
    "d4": 125.0,                                            # aperture -> OAP4 (= EFL4: recollimates)
    "oap4_efl": 125.0, "oap4_D": 76.2, "oap4_mat": "gold", "oap4_angle": 90.0,
    "d45": 375.0,                                           # OAP4 -> OAP5 (f4 + f5 for an exact 4f relay)
    "oap5_efl": 250.0, "oap5_D": 76.2, "oap5_mat": "gold", "oap5_angle": 90.0,
    "d_receiver": 250.0,                                    # last OAP -> receiver aperture (= its EFL)
}


def _rot(d, deg):
    """Rotate a direction counter-clockwise by deg (seen from above)."""
    c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]])


def normalize_frontend(fe):
    """Convert setups saved before signed angles (bs_refl / oapN_turn) to the signed convention."""
    fe = dict(fe)
    if "bs_refl" in fe:
        try:
            a = abs(float(fe.get("bs_angle", 45)))
            fe["bs_angle"] = -a if fe["bs_refl"] == "-y" else a
        except (TypeError, ValueError):
            pass
        fe.pop("bs_refl")
    for i in (3, 4, 5):
        t = fe.pop(f"oap{i}_turn", None)
        if t is not None and f"oap{i}_angle" not in fe:
            fe[f"oap{i}_angle"] = -90.0 if t == "right" else 90.0
    if "rx_mode" not in fe:                    # setups saved before the confocal option
        fe.setdefault("bs_angle", 45.0)
        fe.setdefault("oap3_angle", -90.0)
    fe.setdefault("oap1_angle", 90.0)
    fe.setdefault("oap2_angle", -90.0)
    return fe


def frontend_layout(fe):
    """Positions of the sender / beamsplitter / receiver front end.
    OAP1 sits at the origin and sends the beam +x through the beamsplitter to OAP2. The sender
    is placed so that OAP1 turns its beam by oap1_angle; OAP2 turns the beam by oap2_angle into
    the beam path. The return beam reflects off the beamsplitter (signed angle of incidence:
    + toward +y, - toward -y) into the return arm: OAP3 -> receiver ('direct'), or
    OAP3 -> aperture -> OAP4 -> OAP5 -> receiver ('confocal'). All angles are signed, + = CCW."""
    fe = normalize_frontend(fe)
    g = lambda k: float(fe[k])
    th = np.radians(g("bs_angle"))
    x = np.array([1.0, 0.0])
    oap1 = np.array([0.0, 0.0])
    sender_dir = _rot(x, -g("oap1_angle"))                     # sender beam direction before OAP1
    sender = oap1 - sender_dir * g("d_sender")
    bs = oap1 + x * g("d1")
    oap2 = bs + x * g("d2")
    out_dir = _rot(x, g("oap2_angle"))
    refl = np.array([np.cos(2 * th), np.sin(2 * th)])          # return beam after the BS
    chain, p, d = [], bs, refl
    lay = {"sender": sender, "oap1": oap1, "bs": bs, "oap2": oap2, "out_dir": out_dir, "refl_dir": refl}

    def space(L, label):
        nonlocal p
        chain.append({"type": "space", "L": L, "start": p.copy(), "dir": d.copy(), "label": label})
        p = p + d * L

    def oap(i):
        nonlocal d
        uin = d.copy()
        ang = g(f"oap{i}_angle")
        d = _rot(d, ang)
        chain.append({"type": "oap", "name": f"OAP{i}", "xy": p.copy(), "efl": g(f"oap{i}_efl"), "angle": ang,
                      "D": fe.get(f"oap{i}_D", ""), "mat": fe.get(f"oap{i}_mat", "gold"), "uin": uin, "uout": d.copy()})
        lay[f"oap{i}"] = p.copy()

    space(g("d3"), "Beamsplitter")
    oap(3)
    if fe.get("rx_mode", "direct") == "confocal":
        space(g("d_ap"), "OAP3")
        chain.append({"type": "aperture", "name": "Aperture", "xy": p.copy(), "D": fe.get("ap_D", ""), "u": d.copy()})
        lay["aperture"] = p.copy()
        space(g("d4"), "Aperture")
        oap(4)
        space(g("d45"), "OAP4")
        oap(5)
        space(g("d_receiver"), "OAP5")
    else:
        space(g("d_receiver"), "OAP3")
    lay.update(receiver=p.copy(), rx_dir=d.copy(), chain=chain)
    return lay


class Bench:
    """Optical system built from a setup dict (the same structure the GUI edits and saves)."""

    def __init__(self, setup):
        self.pol = setup.get("pol", "s")
        self.inline = setup.get("inline", True)
        self.warnings = []
        self.comps = []
        self.bs_idx = None
        self.rx = None
        fe = setup.get("frontend") or {"kind": "single"}
        if fe.get("kind") == "bs":
            fe = normalize_frontend(fe)
            self.meas = "S21"
            for k in ("sender_D", "receiver_D", "oap1_efl", "oap2_efl", "oap3_efl"):
                _num(fe, k, "Front end", positive=True)
            for k in ("d_sender", "d1", "d2", "d3", "d_receiver"):
                _num(fe, k, "Front end", positive=True)
            th = _num(fe, "bs_angle", "Beamsplitter")
            if not 0 < abs(th) < 90:
                raise ValueError("Beamsplitter: the angle of incidence must be between -90 and 90 deg, and not 0 "
                                 "(+ reflects the return beam toward +y, - toward -y).")
            if abs(th) < 10:
                self.warnings.append(f"Beamsplitter at {th:g} deg: the reflected return beam runs almost back along "
                                     "the incoming beam, so the return-arm optics would block it.")
            if abs(th) > 80:
                self.warnings.append(f"Beamsplitter at {th:g} deg is near grazing; its footprint is very long.")
            oap_keys = [1, 2, 3] + ([4, 5] if fe.get("rx_mode", "direct") == "confocal" else [])
            for i in oap_keys:
                a = _num(fe, f"oap{i}_angle", f"OAP{i} turn angle")
                if not 0 < abs(a) <= 90:
                    raise ValueError(f"OAP{i}: the turn (off-axis) angle must be between -90 and 90 deg and not 0 "
                                     "(+ turns the beam left / counter-clockwise, - right / clockwise).")
            lay = frontend_layout(fe)
            self.layout = lay
            self.wg_xy = lay["sender"]
            self.wg_D = float(fe["sender_D"])
            self.model = fe.get("model", "he11")
            opt = lambda k, nm: _num(fe, k, nm, positive=True, optional=True)
            self.comps.append(self._make({"type": "oap", "x": lay["oap1"][0], "y": lay["oap1"][1], "efl": fe["oap1_efl"],
                                          "offAxis": abs(float(fe["oap1_angle"])), "diameter": fe.get("oap1_D", ""),
                                          "material": fe.get("oap1_mat", "gold")}, "OAP1"))
            bs = {"type": "bs", "name": "Beamsplitter", "xy": lay["bs"], "diameter": opt("bs_D", "Beamsplitter"),
                  "angle": th, "model": fe.get("bs_model", "fixed")}
            if bs["model"] == "fixed":
                bs["R"] = _num(fe, "bs_R", "Beamsplitter reflectance (%)")
                bs["T"] = _num(fe, "bs_T", "Beamsplitter transmittance (%)")
                if not (0 <= bs["R"] <= 100 and 0 <= bs["T"] <= 100 and bs["R"] + bs["T"] <= 100 + 1e-9):
                    raise ValueError("Beamsplitter: R and T must each be 0-100 % and add up to at most 100 %.")
            else:
                bs["thickness"] = _num(fe, "bs_thickness", "Beamsplitter thickness", positive=True)
                bs["eps"] = fe["bs_eps"]
                eps_of(bs["eps"], np.array([F_MIN, 300.0, F_MAX]))
            self.bs_idx = 1
            self.comps.append(bs)
            self.comps.append(self._make({"type": "oap", "x": lay["oap2"][0], "y": lay["oap2"][1], "efl": fe["oap2_efl"],
                                          "offAxis": abs(float(fe["oap2_angle"])), "diameter": fe.get("oap2_D", ""),
                                          "material": fe.get("oap2_mat", "gold")}, "OAP2"))
            keys = ["d3", "oap3_efl", "d_receiver"]
            if fe.get("rx_mode", "direct") == "confocal":
                keys += ["d_ap", "d4", "oap4_efl", "d45", "oap5_efl"]
                _num(fe, "ap_D", "Aperture diameter", positive=True, optional=True)
            for k in keys:
                _num(fe, k, "Return arm", positive=True)
            chain = []
            for el in lay["chain"]:
                el = dict(el)
                if el["type"] == "oap":
                    el["D"] = _num(el, "D", el["name"], positive=True, optional=True)
                    el["mat"] = {"kind": "metal", "metal": el["mat"]}
                elif el["type"] == "aperture":
                    el["D"] = _num(el, "D", "Aperture", positive=True, optional=True)
                chain.append(el)
            self.rx = {"D": float(fe["receiver_D"]), "model": fe.get("model", "he11"), "chain": chain}
            prev_xy = lay["oap2"]
        else:
            self.meas = "S11"
            wg = setup["waveguide"]
            self.wg_xy = np.array([_num(wg, "x", "Waveguide"), _num(wg, "y", "Waveguide")])
            self.wg_D = _num(wg, "diameter", "Waveguide", positive=True)
            self.model = wg.get("model", "he11")
            prev_xy = self.wg_xy
        if self.bs_idx is not None and not setup["components"]:
            raise ValueError("Add the beam-path components after OAP2. The last one reflects the beam back.")
        for c in setup["components"]:
            nm = c.get("name") or c["type"]
            xy = np.array([_num(c, "x", nm), _num(c, "y", nm)])
            if c["type"] == "cryostat":
                # Expand into: window (in-line slab) -> gap -> sample. (x, y) is the sample front surface;
                # the window sits up-beam of it along the incoming beam direction.
                tw = _num(c, "window_thickness", f"{nm} window", positive=True)
                gap = _num(c, "gap", f"{nm} window-to-sample gap")
                if gap < 0:
                    raise ValueError(f"{nm}: the window-to-sample gap can't be negative.")
                v = xy - prev_xy
                L = float(np.hypot(*v))
                if L <= gap + tw:
                    raise ValueError(f"{nm}: the window ({tw:g} mm) plus the gap ({gap:g} mm) don't fit between "
                                     f"the previous element and the sample ({L:.1f} mm apart).")
                u = v / L
                win = {"type": "sample", "name": f"{nm} window", "x": xy[0] - u[0] * (gap + tw), "y": xy[1] - u[1] * (gap + tw),
                       "diameter": c.get("window_diameter", ""), "tilt": c.get("window_tilt", 0),
                       "layers": [{"thickness": tw, "eps": c["window_eps"]}], "backing": {"kind": "vacuum"}}
                smp = dict(c, type="sample", name=f"{nm} sample")
                self.comps.append(self._make(win, f"{nm} window"))
                self.comps.append(self._make(smp, f"{nm} sample"))
            else:
                self.comps.append(self._make(c, nm))
            prev_xy = xy
        self._geometry()

    def _make(self, c, nm):
        d = dict(type=c["type"], name=nm, xy=np.array([_num(c, "x", nm), _num(c, "y", nm)]),
                 diameter=_num(c, "diameter", nm, positive=True, optional=True))
        if c["type"] in ("mirror", "oap"):
            d["material"] = {"kind": "metal", "metal": c.get("material", "gold")}
        if c["type"] == "mirror":
            d["angle"] = _num(c, "angle", nm, optional=True)
        if c["type"] == "oap":
            d["efl"] = _num(c, "efl", nm, positive=True)
            d["offAxis"] = _num(c, "offAxis", nm, optional=True) or 90.0
        if c["type"] == "sample":
            if not c["layers"]:
                raise ValueError(f"{nm}: add at least one layer.")
            d["tilt"] = _num(c, "tilt", nm, optional=True) or 0.0
            d["layers"] = [(_num(L, "thickness", f"{nm} layer {i + 1}", positive=True), L["eps"])
                           for i, L in enumerate(c["layers"])]
            d["backing"] = c.get("backing", {"kind": "vacuum"})
            if d["backing"]["kind"] == "ito":
                b = d["backing"]
                for k, lab in (("sigma", "ITO conductivity"), ("tau_fs", "ITO scattering time")):
                    _num(b, k, f"{nm} backing: {lab}", positive=True)
                _num(b, "thickness_nm", f"{nm} backing: ITO thickness", positive=True, optional=True)
                _num(b, "sub_re", f"{nm} backing: substrate eps'")
                _num(b, "sub_im", f"{nm} backing: substrate eps''")
                _num(b, "sub_thickness", f"{nm} backing: substrate thickness", positive=True, optional=True)
            for e in [e for _, e in d["layers"]] + [d["backing"]]:
                if e["kind"] in ("expr", "const", "lorentz"):
                    eps_of(e, np.array([F_MIN, 300.0, F_MAX]))   # raises a clear error if invalid
            d["thick"] = sum(t for t, _ in d["layers"])
        return d

    # ------------------------------------------------------------------ geometry
    def _geometry(self):
        comps = self.comps
        if not comps:
            raise ValueError("Add at least one component. The last one reflects the beam back into the waveguide.")
        nodes = [self.wg_xy] + [c["xy"] for c in comps]
        self.nodes = np.array(nodes)
        self.u, self.seg_len = [], []
        for j in range(len(comps)):
            d = self.nodes[j + 1] - self.nodes[j]
            L = float(np.hypot(*d))
            if L == 0:
                raise ValueError(f"{comps[j]['name']} is at the same position as the element before it.")
            self.u.append(d / L)
            self.seg_len.append(L)
        n = len(comps)
        self.aoi = []
        for j, c in enumerate(comps):
            if j < n - 1:
                dfl = float(np.arccos(np.clip(np.dot(self.u[j], self.u[j + 1]), -1, 1)))
                dd = np.degrees(dfl)
                if c["type"] == "mirror":
                    if dd < 1e-6:
                        raise ValueError(f"{c['name']}: the beam goes straight through this mirror. Check the positions.")
                    a = (np.pi - dfl) / 2
                    if c["angle"] is not None and abs(np.degrees(a) - c["angle"]) > 0.5:
                        self.warnings.append(f"{c['name']}: the positions give an angle of incidence of "
                                             f"{np.degrees(a):.1f} deg, not the {c['angle']:g} deg entered. The position value is used.")
                elif c["type"] == "bs":
                    a = np.radians(abs(c["angle"]))
                    if dd > 0.5:
                        self.warnings.append("Beamsplitter: the beam should pass straight through it to OAP2.")
                elif c["type"] == "oap":
                    a = dfl / 2
                    if abs(dd - c["offAxis"]) > 0.5:
                        self.warnings.append(f"{c['name']}: the positions turn the beam by {dd:.1f} deg, "
                                             f"but this is a {c['offAxis']:g} deg OAP.")
                else:
                    a = np.radians(c["tilt"])
                    if dd > 0.5:
                        self.warnings.append(f"{c['name']}: samples in the middle of the path transmit, but the "
                                             f"next component is {dd:.1f} deg off the beam axis.")
                    if c["tilt"]:
                        self.warnings.append(f"{c['name']}: tilted sample; its lateral beam walk-off is ignored.")
            else:
                if c["type"] == "oap":
                    raise ValueError("The last component must send the beam straight back: "
                                     "use a flat mirror at normal incidence or a sample.")
                a = 0.0
            self.aoi.append(a)
        self.space_len, self.seg_start = [], []
        for j in range(n):
            tp = comps[j - 1]["thick"] if j > 0 and comps[j - 1]["type"] == "sample" else 0.0
            L = self.seg_len[j] - tp
            if L < 0:
                raise ValueError(f"{comps[j]['name']} is inside the sample {comps[j - 1]['name']}.")
            self.space_len.append(L)
            self.seg_start.append(self.nodes[j] + self.u[j] * tp)
        if self.bs_idx is None:
            self.round_trip = 2 * sum(self.seg_len)
        else:   # sender -> end reflector -> back to the beamsplitter -> receiver
            self.round_trip = (sum(self.seg_len) + sum(self.seg_len[self.bs_idx + 1:])
                               + sum(el["L"] for el in self.rx["chain"] if el["type"] == "space"))

    # ----------------------------------------------------------- coefficients
    def _coeffs(self, f, k0):
        out, n = [], len(self.comps)
        for j, c in enumerate(self.comps):
            last = j == n - 1
            if c["type"] in ("mirror", "oap"):
                r, _ = tmm([], [], n_of(eps_of(c["material"], f)), k0, 0.0 if last else self.aoi[j], self.pol)
                out.append({"r": r})
                continue
            if c["type"] == "bs":
                out.append(self._bs_coeffs(c, f, k0))
                continue
            Ns = [n_of(eps_of(e, f)) for _, e in c["layers"]]
            ds = [t * MM for t, _ in c["layers"]]
            if last:
                bN, bd, sub = backing_stack(c["backing"], f)
                r, _ = tmm(Ns + bN, ds + bd, sub, k0, 0.0, self.pol)
                out.append({"r": r})
                continue
            rf, t = tmm(Ns, ds, 1.0, k0, self.aoi[j], self.pol)
            rb, _ = tmm(Ns[::-1], ds[::-1], 1.0, k0, 0.0, self.pol)
            out.append({"rf": rf, "rb": rb, "t": t, "Lred": sum(d / np.real(N) for N, d in zip(Ns, ds))})
        return out

    def _bs_coeffs(self, c, f, k0):
        """Beamsplitter amplitude coefficients: t (sender side -> OAP2) and r (return -> OAP3)."""
        th = np.radians(abs(c["angle"]))
        if c["model"] == "fixed":
            one = np.ones_like(np.atleast_1d(k0), dtype=complex)
            return {"t": np.sqrt(c["T"] / 100) * one, "r": np.sqrt(c["R"] / 100) * one, "Lred": 0.0}
        N = n_of(eps_of(c["eps"], f))
        d = c["thickness"] * MM
        r, t = tmm([N], [d], 1.0, k0, th, self.pol)
        nr = np.real(N)
        cos_t = np.sqrt(1 - (np.sin(th) / nr) ** 2)
        # paraxial correction for the slab: path d/cos(theta_t) at index n instead of air
        Lred = d / cos_t * (1 / nr - 1)
        return {"t": t, "r": r, "Lred": Lred}

    def _rx_arm(self, q, amp, lam, k0, cf_rx, rec_space=None, rec_event=None):
        """Beamsplitter -> return arm -> receiver input. rec_space(q, amp, L, start, dir, label)
        records the beam; rec_event(name, q, transmission, aoi_deg) records each element."""
        sp = rec_space or (lambda q, amp, L, *a: self._space(q, amp, L, k0))
        k = 0
        for el in self.rx["chain"]:
            if el["type"] == "space":
                q, amp = sp(q, amp, el["L"], el["start"], el["dir"], el["label"])
            elif el["type"] == "oap":
                cl = clip_factor(q, lam, el["D"] * MM if el["D"] else None)
                if rec_event:
                    rec_event(el["name"], q, np.abs(np.atleast_1d(cl)[0]) ** 2, abs(el["angle"]) / 2)
                q, ph = abcd(q, 1, 0, -1 / (el["efl"] * MM), 1)
                amp = amp * cl * ph * cf_rx[k]
                k += 1
            else:   # spatial-filter aperture: projection back onto the fundamental mode
                T = aperture_T(q, lam, el["D"] * MM if el["D"] else None)
                if rec_event:
                    rec_event("Aperture", q, float(np.atleast_1d(T)[0]), 0.0)
                amp = amp * T
        return q, amp

    def _rx_r(self, f, k0):
        return [tmm([], [], n_of(eps_of(el["mat"], f)), k0, np.radians(abs(el["angle"]) / 2), self.pol)[0]
                for el in self.rx["chain"] if el["type"] == "oap"]

    def _clip(self, j, q, lam):
        D = self.comps[j]["diameter"]
        return clip_factor(q, lam, D * MM if D else None)

    @staticmethod
    def _space(q, amp, Lmm, k0):
        q2, ph = abcd(q, 1, Lmm * MM, 0, 1)
        return q2, amp * ph * np.exp(-1j * k0 * Lmm * MM)

    def _pass(self, j, q, amp, cf, lam):
        c = self.comps[j]
        amp = amp * self._clip(j, q, lam)
        if c["type"] == "mirror":
            return q, amp * cf[j]["r"]
        if c["type"] == "oap":
            q2, ph = abcd(q, 1, 0, -1 / (c["efl"] * MM), 1)
            return q2, amp * ph * cf[j]["r"]
        q2, ph = abcd(q, 1, cf[j]["Lred"], 0, 1)        # sample or beamsplitter (transmission)
        return q2, amp * cf[j]["t"] * ph

    def _receive(self, q, lam):
        if self.rx is not None:
            a = self.rx["D"] / 2 * MM
            if self.rx["model"] == "he11":
                return he11_overlap(q, lam, a)
            return gaussian_overlap(q, lam, HE11_RATIO * a)
        a = self.wg_D / 2 * MM
        if self.model == "he11":
            return he11_overlap(q, lam, a)
        return gaussian_overlap(q, lam, HE11_RATIO * a)

    def launch(self, lam):
        a = self.wg_D / 2 * MM
        q0 = q_waist(HE11_RATIO * a, lam)
        amp = he11_overlap(q0, lam, a) if self.model == "he11" else np.ones_like(np.atleast_1d(q0))
        return q0, amp

    # ------------------------------------------------------------ main-path profile
    def profile(self, f_ghz):
        """Main path (samples always transmit) plus, for every untilted in-line sample, the beam
        that reflects straight back off its front face. Each path is recorded with its own q."""
        f = np.array([float(f_ghz)])
        lam = C0 / (f * 1e9)
        k0 = 2 * np.pi / lam
        cf = self._coeffs(f, k0)
        cf_rx = self._rx_r(f, k0) if self.rx is not None else None
        n = len(self.comps)

        def new_rec(label, s0=0.0):
            return {"s": s0, "seg": [], "events": [], "waists": [], "pass": "forward", "label": label}

        def make_tools(rec):
            def space(q, amp, Lmm, start, direction, label):
                L = Lmm * MM
                if L > 0:
                    z = np.linspace(0, L, 200)
                    rec["seg"].append({"s": (rec["s"] + z) / MM, "w": w_of_q(q[0] + z, lam[0]) / MM,
                                       "xy": start[None, :] + direction[None, :] * z[:, None] / MM,
                                       "u": direction, "pass": rec["pass"]})
                    zw = -np.real(q[0])
                    tol = 0.01 * MM                      # waists within 10 um of a surface count as on it
                    if tol < zw <= L + tol:
                        zw = min(zw, L)
                        rec["waists"].append({"s": (rec["s"] + zw) / MM, "pass": rec["pass"],
                                              "w0": np.sqrt(lam[0] * np.imag(q[0]) / np.pi) / MM,
                                              "xy": start + direction * zw / MM, "after": label})
                    rec["s"] += L
                return self._space(q, amp, Lmm, k0)

            def event(name, q, T, idx=None, aoi=None):
                rec["events"].append({"s": rec["s"] / MM, "name": name, "idx": idx, "aoi": aoi, "pass": rec["pass"],
                                      "w": float(w_of_q(q[0], lam[0]) / MM), "clip": float(T)})

            def action(j, q, amp):
                event(self.comps[j]["name"], q, np.abs(np.atleast_1d(self._clip(j, q, lam))[0]) ** 2, idx=j)
                q2, a2 = self._pass(j, q, amp, cf, lam)
                if self.comps[j]["type"] == "sample":
                    rec["s"] += self.comps[j]["thick"] * MM
                return q2, a2

            def back_from(i_start, q, amp):
                """Travel back from the entrance face of component i_start to the receiver."""
                rec["pass"] = "return"
                for i in range(i_start, -1, -1):
                    if i < i_start and i == self.bs_idx:      # reflect into the receiver arm
                        cl = self._clip(i, q, lam)
                        event("Beamsplitter", q, np.abs(np.atleast_1d(cl)[0]) ** 2, idx=i)
                        amp = amp * cl * cf[i]["r"]
                        q, amp = self._rx_arm(q, amp, lam, k0, cf_rx, space,
                                              lambda nm, qq, T, aoi: event(nm, qq, T, aoi=aoi))
                        return q, amp
                    if i < i_start:
                        q, amp = action(i, q, amp)
                    q, amp = space(q, amp, self.space_len[i], self.nodes[i + 1], -self.u[i], self.comps[i]["name"])
                return q, amp
            return space, event, action, back_from

        rec = new_rec("main")
        space, event, action, back_from = make_tools(rec)
        q, amp = self.launch(lam)
        refl_starts = []                 # (component index, q, amp, s) at the entrance of in-line samples
        for j in range(n):
            q, amp = space(q, amp, self.space_len[j], self.seg_start[j], self.u[j],
                           "waveguide" if j == 0 else self.comps[j - 1]["name"])
            if j < n - 1:
                c = self.comps[j]
                if c["type"] == "sample" and self.inline and not c["tilt"]:
                    refl_starts.append((j, q, amp, rec["s"]))
                q, amp = action(j, q, amp)
            else:
                cl = self._clip(j, q, lam)
                event(self.comps[j]["name"], q, np.abs(np.atleast_1d(cl)[0]) ** 2, idx=j)
                q, amp = back_from(j, q, amp * cl * cf[j]["r"])
        main_amp = complex((amp * self._receive(q, lam))[0])

        refl = []
        for j, qj, aj, sj in refl_starts:
            r = new_rec(f"reflection off {self.comps[j]['name']}", sj)
            _, ev, _, back = make_tools(r)
            cl = self._clip(j, qj, lam)
            ev(self.comps[j]["name"] + " (front face)", qj, np.abs(np.atleast_1d(cl)[0]) ** 2, idx=j)
            qr, ar = back(j, qj, aj * cl * cf[j]["rf"])
            r["amp"] = complex((ar * self._receive(qr, lam))[0])
            # align with the main return pass so the same components sit at the same s in the plot
            main_back = [e for e in rec["events"] if e["pass"] == "return"]
            for e in r["events"][1:]:
                m = next((me for me in main_back if me["name"] == e["name"]), None)
                if m is not None:
                    off = m["s"] - e["s"]
                    for sg in r["seg"]:
                        sg["s"] = sg["s"] + off
                    for x in r["events"] + r["waists"]:
                        x["s"] += off
                    break
            r["w_end"] = float(w_of_q(qr[0], lam[0]) / MM)
            refl.append(r)
        launch_power = float(np.abs(self.launch(lam)[1][0]) ** 2)
        return {"rec": rec, "refl": refl, "main_amp": main_amp, "launch": launch_power, "lam_mm": lam[0] / MM,
                "w0_mm": HE11_RATIO * self.wg_D / 2}

    # ---------------------------------------------------------------- S11 sweep
    def s11(self, f_ghz, tol=1e-4, max_events=400):
        """Coherent sum over all bounce paths. Returns dict(f, total, groups{n_refl: S11}, deemb)."""
        f = np.atleast_1d(np.asarray(f_ghz, float))
        lam = C0 / (f * 1e9)
        k0 = 2 * np.pi / lam
        cf = self._coeffs(f, k0)
        n = len(self.comps)

        def split(j):
            c = self.comps[j]
            return j < n - 1 and c["type"] == "sample" and self.inline and not c["tilt"]

        q0, amp0 = self.launch(lam)
        cf_rx = self._rx_r(f, k0) if self.rx is not None else None
        groups = {}
        stack = [(0, +1, q0, amp0, 0, 0)]
        while stack:
            j, d, q, amp, nr, ev = stack.pop()
            if np.max(np.abs(amp)) < tol or ev > max_events:
                continue
            q, amp = self._space(q, amp, self.space_len[j], k0)
            if d > 0:
                if j == n - 1:
                    stack.append((j, -1, q, amp * self._clip(j, q, lam) * cf[j]["r"], nr, ev + 1))
                elif split(j):
                    A = amp * self._clip(j, q, lam)
                    q2, ph = abcd(q, 1, cf[j]["Lred"], 0, 1)
                    stack.append((j + 1, +1, q2, A * cf[j]["t"] * ph, nr, ev + 1))
                    stack.append((j, -1, q, A * cf[j]["rf"], nr + 1, ev + 1))
                else:
                    q2, a2 = self._pass(j, q, amp, cf, lam)
                    stack.append((j + 1, +1, q2, a2, nr, ev + 1))
            else:
                if j == 0:
                    groups[nr] = groups.get(nr, 0) + amp * self._receive(q, lam)
                    continue
                i = j - 1
                if i == self.bs_idx:      # reflect into the receiver arm; the transmitted part returns to the sender (lost)
                    A = amp * self._clip(i, q, lam) * cf[i]["r"]
                    q2, a2 = self._rx_arm(q, A, lam, k0, cf_rx)
                    groups[nr] = groups.get(nr, 0) + a2 * self._receive(q2, lam)
                    continue
                if split(i):
                    A = amp * self._clip(i, q, lam)
                    q2, ph = abcd(q, 1, cf[i]["Lred"], 0, 1)
                    stack.append((i, -1, q2, A * cf[i]["t"] * ph, nr, ev + 1))
                    stack.append((j, +1, q, A * cf[i]["rb"], nr + 1, ev + 1))
                else:
                    q2, a2 = self._pass(i, q, amp, cf, lam)
                    stack.append((i, -1, q2, a2, nr, ev + 1))
        total = sum(groups.values()) if groups else np.zeros_like(f, dtype=complex)
        return {"f": f, "total": total, "groups": dict(sorted(groups.items())),
                "deemb": total * np.exp(1j * k0 * self.round_trip * MM)}

    def auto_points(self, f1, f2):
        n = 10 * (f2 - f1) * 1e9 * self.round_trip * MM / C0
        return int(np.clip(np.ceil(n), 401, 20000))


# =============================================================================================
#  Plotting
# =============================================================================================
FWD, RET, GOLD, SAMPLE, COPPER, REFL = "#3b50c4", "#c27510", "#b48d2a", "#4b8660", "#b8703a", "#9c3fb5"


def _perp(u):
    return np.array([-u[1], u[0]])


def draw_layout(ax, bench, prof, f_ghz):
    rec = prof["rec"]
    ax.clear()
    ax.set_facecolor("#dde3e8")
    for sg in rec["seg"]:
        p = _perp(sg["u"])
        w = sg["w"][:, None]
        up, dn = sg["xy"] + p * w, sg["xy"] - p * w
        if sg["pass"] == "forward":
            ax.add_patch(Polygon(np.vstack([up, dn[::-1]]), closed=True, fc=FWD, alpha=0.22, ec="none", zorder=2))
            ax.plot(sg["xy"][:, 0], sg["xy"][:, 1], color=FWD, lw=0.9, zorder=3)
        else:
            ax.plot(up[:, 0], up[:, 1], color=RET, lw=0.9, ls="--", zorder=3)
            ax.plot(dn[:, 0], dn[:, 1], color=RET, lw=0.9, ls="--", zorder=3)
    def horn(P, u_beam, D, label, behind):
        """Horn body drawn on the side 'behind' (unit vector away from the beam)."""
        p, L, hw = _perp(u_beam), 40.0, D / 2 + 1.5
        body = np.array([P + behind * L + p * hw, P + p * hw, P - p * hw, P + behind * L - p * hw])
        ax.add_patch(Polygon(body, closed=True, fc="#5a6775", ec="k", lw=0.8, zorder=4))
        side = p if abs(p[0]) < 0.5 else p * 1.0
        off = hw + (16 if abs(u_beam[1]) > 0.5 else 7)   # vertical horns need room for the text width
        ax.annotate(label, P + behind * L / 2 + side * off, ha="center", va="center", fontsize=9, zorder=6)

    def oap_shape(P, uin, uout, D, efl, label):
        nr = (uout - uin) / np.linalg.norm(uout - uin)
        t = _perp(nr)
        s_ = np.linspace(-D / 2, D / 2, 40)
        curve = P + t * s_[:, None] + nr * (s_**2 / (4 * efl) * 1.5)[:, None]
        ax.add_patch(Polygon(np.vstack([curve, (curve - nr * 2.5)[::-1]]), closed=True, fc=GOLD, ec="k", lw=0.7, zorder=5))
        ax.annotate(label, P - nr * (0.35 * D + 9), ha="center", va="center", fontsize=9, zorder=6)

    u0 = bench.u[0]
    if bench.rx is None:
        horn(bench.wg_xy, u0, bench.wg_D, "Waveguide", -u0)
    else:
        lay = bench.layout
        horn(bench.wg_xy, u0, bench.wg_D, "Sender", -u0)
        horn(lay["receiver"], lay["rx_dir"], bench.rx["D"], "Receiver", lay["rx_dir"])
        for el in bench.rx["chain"]:
            if el["type"] == "oap":
                oap_shape(el["xy"], el["uin"], el["uout"], el["D"] or 25.4, el["efl"], f"{el['name']}\nEFL {el['efl']:g}")
            elif el["type"] == "aperture":
                p, D = _perp(el["u"]), el["D"] or 0.0
                for sgn in (1, -1):        # two jaws of the spatial filter, gap = aperture diameter
                    a0 = el["xy"] + p * sgn * D / 2
                    a1 = el["xy"] + p * sgn * (D / 2 + 14)
                    jaw = np.array([a0 - el["u"] * 1.2, a1 - el["u"] * 1.2, a1 + el["u"] * 1.2, a0 + el["u"] * 1.2])
                    ax.add_patch(Polygon(jaw, closed=True, fc="#2f3a44", ec="k", lw=0.6, zorder=6))
                ax.annotate(f"Aperture\n⌀ {el['D']:g} mm" if el["D"] else "Aperture (open)",
                            el["xy"] - p * (D / 2 + 26), ha="center", va="center", fontsize=9, zorder=6)
    # direct back-reflections off in-line samples (each with its own beam)
    for r in prof.get("refl", []):
        for sg in r["seg"]:
            pp = _perp(sg["u"])
            w = sg["w"][:, None]
            for side in (sg["xy"] + pp * w, sg["xy"] - pp * w):
                ax.plot(side[:, 0], side[:, 1], color=REFL, lw=0.9, ls=":", zorder=3)
    n = len(bench.comps)
    for j, c in enumerate(bench.comps):
        P = c["xy"]
        uin = bench.u[j]
        uout = bench.u[j + 1] if j < n - 1 else -uin
        if c["type"] == "bs":
            nr = bench.layout["refl_dir"] + uin
            nr = nr / np.linalg.norm(nr)
            t = _perp(nr)
            D = c["diameter"] or 25.4
            s_ = np.linspace(-D / 2, D / 2, 2)
            line = P + t * s_[:, None]
            ax.add_patch(Polygon(np.vstack([line + nr * 0.8, (line - nr * 0.8)[::-1]]), closed=True,
                                 fc="#a8c6dc", ec="#2f5f80", lw=0.8, alpha=0.9, zorder=5))
            lab = "Beamsplitter\n" + (f"R {c['R']:g}% / T {c['T']:g}%" if c["model"] == "fixed" else f"{c['thickness']:g} mm slab")
            ax.annotate(lab, P + nr * (0.3 * D + 12), ha="center", va="center", fontsize=9, zorder=6)
            continue
        if c["type"] in ("mirror", "oap"):
            nr = (uout - uin) / np.linalg.norm(uout - uin)
            t = _perp(nr)
            D = c["diameter"] or 25.4
            s = np.linspace(-D / 2, D / 2, 40)
            sag = s**2 / (4 * c["efl"]) * 1.5 if c["type"] == "oap" else 0 * s
            curve = P + t * s[:, None] + nr * sag[:, None]
            ax.add_patch(Polygon(np.vstack([curve, (curve - nr * 2.5)[::-1]]), closed=True, fc=GOLD, ec="k", lw=0.7, zorder=5))
            lab = c["name"] + (f"\nEFL {c['efl']:g}" if c["type"] == "oap" else "")
            off = -nr * (0.35 * D + 9)
        else:
            p = _perp(uin)
            D = c["diameter"] or 12.0
            T = max(c["thick"], 1.5)
            rect = np.array([P + p * D / 2, P + p * D / 2 + uin * T, P - p * D / 2 + uin * T, P - p * D / 2])
            ax.add_patch(Polygon(rect, closed=True, fc=SAMPLE, ec="k", lw=0.7, zorder=5))
            lab = c["name"]
            if j == n - 1 and c["backing"]["kind"] != "vacuum":
                b = np.array([P + uin * T + p * D / 2, P + uin * (T + 4) + p * D / 2,
                              P + uin * (T + 4) - p * D / 2, P + uin * T - p * D / 2])
                bk = c["backing"]
                col = {"copper": COPPER, "gold": GOLD, "aluminum": "#a9b2ba"}.get(bk.get("metal"), "#8fb6c9" if bk["kind"] == "ito" else "#8a96a3")
                ax.add_patch(Polygon(b, closed=True, fc=col, ec="k", lw=0.7, zorder=5))
                lab += "\non " + (bk.get("metal") or ("ITO" if bk["kind"] == "ito" else "dielectric"))
            off = uin * (T + 14) if j == n - 1 else p * (D / 2 + 16) + uin * T / 2
        ax.annotate(lab, P + off, ha="center", va="center", fontsize=9, zorder=6)
    for wv in rec["waists"]:
        if wv["pass"] == "forward":
            ax.plot(*wv["xy"], marker="x", color="k", ms=6, zorder=7)
    ax.set_aspect("equal")
    ax.autoscale_view()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pad = 0.08 * max(x1 - x0, y1 - y0) + 15
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    gx = np.arange(np.ceil((x0 - pad) / 25) * 25, x1 + pad, 25)
    gy = np.arange(np.ceil((y0 - pad) / 25) * 25, y1 + pad, 25)
    X, Y = np.meshgrid(gx, gy)
    ax.plot(X.ravel(), Y.ravel(), "o", ms=2.2, color="#bcc5cd", zorder=1)       # breadboard holes
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ttl = f"Setup at {f_ghz:g} GHz: beam ±w (outgoing filled, returning dashed), × = waist"
    if prof.get("refl"):
        ttl += "\npurple dotted: direct back-reflection off in-line samples (window)"
    ax.set_title(ttl, fontsize=10)


def draw_profile(ax, prof, f_ghz):
    rec = prof["rec"]
    ax.clear()
    for sg in rec["seg"]:
        col = FWD if sg["pass"] == "forward" else RET
        ax.fill_between(sg["s"], -sg["w"], sg["w"], color=col, alpha=0.18, lw=0)
        ax.plot(sg["s"], sg["w"], color=col, lw=1.2)
        ax.plot(sg["s"], -sg["w"], color=col, lw=1.2)
    for r in prof.get("refl", []):
        for k, sg in enumerate(r["seg"]):
            lab = r["label"] if k == 0 else None
            ax.plot(sg["s"], sg["w"], color=REFL, lw=1.1, ls=":", label=lab)
            ax.plot(sg["s"], -sg["w"], color=REFL, lw=1.1, ls=":")
    if prof.get("refl"):
        ax.legend(fontsize=8, loc="lower left")
    ymax = max(np.max(s["w"]) for s in rec["seg"]) * 1.25
    for r in prof.get("refl", []):
        ymax = max([ymax] + [np.max(sg["w"]) * 1.08 for sg in r["seg"]])
    for e in rec["events"]:
        ax.axvline(e["s"], color="0.4", lw=0.7, ls=":")
        ax.text(e["s"], ymax * 0.97, e["name"], rotation=90, va="top", ha="right", fontsize=8)
    for wv in rec["waists"]:
        ax.plot(wv["s"], 0, "kx")
        ax.annotate(f"{wv['w0']:.2f} mm", (wv["s"], 0), xytext=(4, -12), textcoords="offset points", fontsize=8)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("Path length along the beam (mm)   [outgoing: blue, returning: orange, back-reflections: purple dotted]")
    ax.set_ylabel("±w, 1/e² radius (mm)")
    ax.set_title(f"Beam profile at {f_ghz:g} GHz", fontsize=10)
    ax.grid(alpha=0.3)


def draw_s11(ax1, ax2, sw, f_mark, band_label, meas="S11"):
    ax1.clear()
    ax2.clear()
    f = sw["f"]
    lw = 0.7 if len(f) > 8000 else 1.1
    ax1.plot(f, 20 * np.log10(np.abs(sw["total"]) + 1e-15), color="k", lw=lw, label="Total", zorder=5)
    if len(sw["groups"]) > 1:
        for (g, v), col in zip(list(sw["groups"].items())[:3], [FWD, RET, SAMPLE]):
            lab = "Main path" if g == 0 else f"{g} sample reflection{'s' if g > 1 else ''}"
            ax1.plot(f, 20 * np.log10(np.abs(v) + 1e-15), color=col, lw=1.0, alpha=0.85, label=lab)
        ax1.legend(fontsize=8, loc="lower left")
    lo = max(ax1.get_ylim()[0], -80)
    ax1.set_ylim(lo, max(0.5, ax1.get_ylim()[1]))
    ax1.set_ylabel(f"|{meas}| (dB)")
    ax1.set_title((f"{meas}: sender → sample → receiver, " if meas == "S21" else "S11 into the waveguide, ") + band_label,
                  fontsize=10)
    ph = np.degrees(np.angle(sw["deemb"]))
    ph_plot = ph.copy()
    ph_plot[1:][np.abs(np.diff(ph)) > 180] = np.nan          # break the line at wraps
    ax2.plot(f, ph_plot, color="k", lw=lw)
    ax2.set_ylim(-185, 185)
    ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.set_ylabel("Phase (deg)\nround-trip delay removed")
    ax2.set_xlabel("Frequency (GHz)")
    for a in (ax1, ax2):
        a.grid(alpha=0.3)
        if f_mark is not None:
            a.axvline(f_mark, color="0.5", ls="--", lw=0.8)


# =============================================================================================
#  Example setups (same JSON structure as File > Save setup)
# =============================================================================================
EXAMPLES = {
    "As built, confocal return arm with spatial filter; cryostat at the OAP2 focus": {
        # beam path begins at the OAP2 focus (375, -125); the window pushes the focus ~1 mm deeper
        "frontend": dict(FRONTEND_DEFAULT),
        "waveguide": {"x": 0, "y": 0, "diameter": 4, "model": "he11"},
        "components": [
            {"type": "cryostat", "name": "Cryostat", "x": 375.0, "y": -126.05, "diameter": 10,
             "window_thickness": 2.0, "gap": 20.0, "window_diameter": 25.4, "window_tilt": 0,
             "window_eps": dict(QUARTZ), "tilt": 0,
             "layers": [{"thickness": 0.5, "eps": {"kind": "const", "re": 9.4, "im": 0.05}}],
             "backing": {"kind": "metal", "metal": "copper"}},
        ],
        "run": {"band": "B3", "f1": 300, "f2": 450, "freq": 440, "pol": "s", "inline": True},
    },
    "Focus onto a sample on copper (reflection)": {
        "waveguide": {"x": 0, "y": 0, "diameter": 4, "model": "he11"},
        "components": [
            {"type": "oap", "name": "OAP1", "x": 50.8, "y": 0, "efl": 50.8, "offAxis": 90, "diameter": 50.8, "material": "gold"},
            {"type": "mirror", "name": "M1", "x": 50.8, "y": 76.2, "angle": 45, "diameter": 50.8, "material": "gold"},
            {"type": "oap", "name": "OAP2", "x": 127, "y": 76.2, "efl": 101.6, "offAxis": 90, "diameter": 50.8, "material": "gold"},
            {"type": "sample", "name": "Crystal", "x": 127, "y": -25.4, "tilt": 0, "diameter": 10,
             "layers": [{"thickness": 0.5, "eps": {"kind": "const", "re": 9.4, "im": 0.05}}],
             "backing": {"kind": "metal", "metal": "copper"}},
        ],
        "run": {"band": "B3", "f1": 300, "f2": 450, "freq": 440, "pol": "s", "inline": True},
    },
    "Sample in the collimated beam (transmission)": {
        "waveguide": {"x": 0, "y": 0, "diameter": 4, "model": "he11"},
        "components": [
            {"type": "oap", "name": "OAP1", "x": 50.8, "y": 0, "efl": 50.8, "offAxis": 90, "diameter": 50.8, "material": "gold"},
            {"type": "sample", "name": "Composite", "x": 50.8, "y": 70, "tilt": 0, "diameter": 25.4,
             "layers": [{"thickness": 0.05, "eps": {"kind": "lorentz", "epsInf": 12, "f0": 520, "de": 1.5, "gamma": 15}},
                        {"thickness": 1.0, "eps": {"kind": "const", "re": 9.4, "im": 0.01}}],
             "backing": {"kind": "vacuum"}},
            {"type": "oap", "name": "OAP2", "x": 50.8, "y": 152.4, "efl": 101.6, "offAxis": 90, "diameter": 50.8, "material": "gold"},
            {"type": "mirror", "name": "Retro", "x": 152.4, "y": 152.4, "angle": 0, "diameter": "", "material": "gold"},
        ],
        "run": {"band": "B4", "f1": 450, "f2": 600, "freq": "", "pol": "s", "inline": True},
    },
    "Cryostat: quartz window, then sample on copper": {
        "waveguide": {"x": 0, "y": 0, "diameter": 4, "model": "he11"},
        "components": [
            {"type": "oap", "name": "OAP1", "x": 50.8, "y": 0, "efl": 50.8, "offAxis": 90, "diameter": 50.8, "material": "gold"},
            {"type": "mirror", "name": "M1", "x": 50.8, "y": 76.2, "angle": 45, "diameter": 50.8, "material": "gold"},
            {"type": "oap", "name": "OAP2", "x": 127, "y": 76.2, "efl": 101.6, "offAxis": 90, "diameter": 50.8, "material": "gold"},
            {"type": "cryostat", "name": "Cryostat", "x": 127, "y": -26.45, "diameter": 10,
             "window_thickness": 2.0, "gap": 20.0, "window_diameter": 25.4, "window_tilt": 0,
             "window_eps": dict(QUARTZ), "tilt": 0,
             "layers": [{"thickness": 0.5, "eps": {"kind": "const", "re": 9.4, "im": 0.05}}],
             "backing": {"kind": "metal", "metal": "copper"}},
        ],
        "run": {"band": "B3", "f1": 300, "f2": 450, "freq": 440, "pol": "s", "inline": True},
    },
    "Empty path": {
        "waveguide": {"x": 0, "y": 0, "diameter": 4, "model": "he11"},
        "components": [],
        "run": {"band": "B3", "f1": 300, "f2": 450, "freq": "", "pol": "s", "inline": True},
    },
}

# =============================================================================================
#  GUI
# =============================================================================================
KIND_LABEL = {"mirror": "Flat mirror", "oap": "Off-axis parabola", "sample": "Sample",
              "cryostat": "Cryostat: window, gap, sample"}
BAND_CHOICES = [f"{k}: {a:g}-{b:g} GHz" for k, (a, b) in BANDS.items()] + ["Custom band"]
BACKINGS = ["Nothing (free standing)", "Gold", "Copper", "Aluminum", "ITO film on glass",
            "Dielectric (constant eps)", "Dielectric eps(w) expression"]
EPS_KINDS = ["Constant eps", "Lorentz oscillator", "eps(w) expression"] + [f"Preset: {k}" for k in PRESETS]
EXPR_HINT = ("Python syntax. w = angular frequency (rad/s), f = GHz, THz = THz. j = sqrt(-1); "
             "sqrt, exp, pi, eps0, c... available.\nPhysics convention: +j*eps'' is loss.")
EXPR_DEFAULT = "12 + 1.5*(2*pi*520e9)**2 / ((2*pi*520e9)**2 - w**2 - j*w*2*pi*15e9)"


class ScrollFrame(ttk.Frame):
    """Vertically scrollable frame (mouse wheel works on Windows, macOS and Linux)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.bind_all("<MouseWheel>", self._wheel, add="+")
        self.bind_all("<Button-4>", lambda e: self._scroll(e, -1), add="+")
        self.bind_all("<Button-5>", lambda e: self._scroll(e, 1), add="+")

    def _inside(self, e):
        w = e.widget
        while w is not None:
            if w is self:
                return True
            w = getattr(w, "master", None)
        return False

    def _wheel(self, e):
        if self._inside(e):
            step = -1 if e.delta > 0 else 1
            self.canvas.yview_scroll(step * (1 if abs(e.delta) < 120 else abs(e.delta) // 120), "units")

    def _scroll(self, e, step):
        if self._inside(e):
            self.canvas.yview_scroll(step, "units")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quasi-optics bench: Gaussian beam and S11 simulator")
        self.geometry("1400x880")
        self.minsize(1000, 650)
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Card.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        style.configure("Run.TButton", font=("TkDefaultFont", 11, "bold"), padding=8)
        style.configure("End.TLabel", foreground=FWD)
        style.configure("Hint.TLabel", foreground="#5a6775")

        self.state = copy.deepcopy(EXAMPLES["As built, confocal return arm with spatial filter; cryostat at the OAP2 focus"])
        self.vars = {}            # path tuple -> tk.StringVar for every editable scalar
        self.result = None
        self.q = queue.Queue()
        self._build_menu()
        self._build_layout()
        self.rebuild_path()
        self.load_run_settings()

    # ------------------------------------------------------------------ layout
    def _build_menu(self):
        m = tk.Menu(self)
        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label="Open setup…", command=self.open_setup)
        fm.add_command(label="Save setup…", command=self.save_setup)
        fm.add_separator()
        fm.add_command(label="Export S11 / S21 as CSV…", command=self.export_csv)
        fm.add_command(label="Save figures as PNG…", command=self.save_figures)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.destroy)
        m.add_cascade(label="File", menu=fm)
        em = tk.Menu(m, tearoff=0)
        for name in EXAMPLES:
            em.add_command(label=name, command=lambda n=name: self.load_setup(copy.deepcopy(EXAMPLES[n])))
        m.add_cascade(label="Examples", menu=em)
        self.config(menu=m)

    def _build_layout(self):
        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, padding=(10, 8))
        right = ttk.Frame(pw, padding=(6, 6))
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        ttk.Label(left, text="Total beam path", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(left, text="From the waveguide outward. The last component sends the beam back.\n"
                             "Positions (mm) are where the beam axis meets each part.\n"
                             "Leave a diameter blank for no aperture clipping.",
                  style="Hint.TLabel").pack(anchor="w", pady=(0, 6))
        self.scroll = ScrollFrame(left)
        self.scroll.pack(fill="both", expand=True)
        self.scroll.configure(width=430)
        add = ttk.Frame(left)
        add.pack(fill="x", pady=6)
        for t, lab in (("mirror", "Add flat mirror"), ("oap", "Add OAP"), ("sample", "Add sample")):
            ttk.Button(add, text=lab, command=lambda t=t: self.add_component(t)).pack(side="left", padx=(0, 6))
        ttk.Button(left, text="Add cryostat (window → gap → sample on copper)",
                   command=self.add_cryostat).pack(anchor="w")

        run = ttk.LabelFrame(left, text="Simulation", padding=8)
        run.pack(fill="x")
        self.band_var = tk.StringVar()
        self.freq_var = tk.StringVar()
        self.pol_var = tk.StringVar()
        self.f1_var, self.f2_var = tk.StringVar(), tk.StringVar()
        self.inline_var = tk.BooleanVar(value=True)
        ttk.Label(run, text="Band").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(run, textvariable=self.band_var, values=BAND_CHOICES, state="readonly", width=20)
        cb.grid(row=0, column=1, sticky="ew", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._toggle_custom())
        ttk.Label(run, text="Profile at (GHz)").grid(row=0, column=2, sticky="w")
        ttk.Entry(run, textvariable=self.freq_var, width=8).grid(row=0, column=3, sticky="w")
        self.custom = ttk.Frame(run)
        self.custom.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(self.custom, text="Start (GHz)").pack(side="left")
        ttk.Entry(self.custom, textvariable=self.f1_var, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(self.custom, text="Stop (GHz)").pack(side="left")
        ttk.Entry(self.custom, textvariable=self.f2_var, width=8).pack(side="left", padx=4)
        ttk.Label(run, text="Polarization").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(run, textvariable=self.pol_var, values=["E vertical (s)", "E horizontal (p)"],
                     state="readonly", width=20).grid(row=2, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Label(run, text="(blank = band centre)", style="Hint.TLabel").grid(row=2, column=2, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(run, text="Include sample back-reflections (standing waves)",
                        variable=self.inline_var).grid(row=3, column=0, columnspan=4, sticky="w", pady=6)
        self.run_btn = ttk.Button(run, text="Simulate", style="Run.TButton", command=self.simulate)
        self.run_btn.grid(row=4, column=0, columnspan=4, sticky="ew")
        self.status = ttk.Label(run, text="", style="Hint.TLabel")
        self.status.grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))
        run.columnconfigure(1, weight=1)

        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        self.figs = {}
        for key, title in (("layout", "Setup diagram"), ("profile", "Beam profile"), ("s11", "S11 / S21")):
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=title)
            fig = Figure(figsize=(9, 6), dpi=100, constrained_layout=True)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            NavigationToolbar2Tk(canvas, frame).update()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figs[key] = (fig, canvas)
            fig.text(0.5, 0.5, "Build the beam path, then press Simulate.", ha="center", va="center", color="0.4")
            canvas.draw()
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text="Numbers")
        self.text = tk.Text(frame, wrap="none", font=("Courier", 10), padx=10, pady=8)
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)

    # ---------------------------------------------------------- path editor
    def _var(self, path, value):
        v = tk.StringVar(value="" if value is None else str(value))
        self.vars[path] = v
        return v

    def _field(self, parent, label, path, value, row, col, width=9):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 2))
        ttk.Entry(parent, textvariable=self._var(path, value), width=width).grid(row=row + 1, column=col, sticky="ew", padx=(0, 8), pady=(0, 4))

    def _combo(self, parent, label, path, value, values, row, col, colspan=1, callback=None, width=14):
        ttk.Label(parent, text=label).grid(row=row, column=col, columnspan=colspan, sticky="w")
        cb = ttk.Combobox(parent, textvariable=self._var(path, value), values=values, state="readonly", width=width)
        cb.grid(row=row + 1, column=col, columnspan=colspan, sticky="ew", padx=(0, 8), pady=(0, 4))
        if callback:
            cb.bind("<<ComboboxSelected>>", lambda e: callback())
        return cb

    def sync(self):
        """Copy every entry back into self.state (keeps strings for blank fields)."""
        for path, var in self.vars.items():
            obj = self.state
            for k in path[:-1]:
                obj = obj[k]
            val = var.get().strip()
            key = path[-1]
            if key in ("name", "expr", "_choice"):
                pass
            elif key == "material" or key.endswith("_mat"):
                val = val.lower()
            elif key == "bs_model":
                val = "slab" if val.startswith("Dielectric") else "fixed"
            elif key == "bs_refl":
                val = "-y" if "−y" in val or "-y" in val else "+y"
            elif key == "rx_mode":
                val = "confocal" if val.startswith("Confocal") else "direct"
            elif key.endswith("_turn"):
                val = "right" if val.startswith("Right") else "left"
            elif key == "model":
                val = "he11" if val.startswith("HE11") else "gaussian"
            else:
                try:
                    val = float(val) if val != "" else ""
                    if isinstance(val, float) and val.is_integer() and key not in ("x", "y"):
                        val = val
                except ValueError:
                    pass
            obj[key] = val
        self.save_run_settings()

    def rebuild_path(self):
        for w in self.scroll.inner.winfo_children():
            w.destroy()
        self.vars = {}
        self._ensure_frontend()
        fe = self.state["frontend"]
        top = ttk.Frame(self.scroll.inner)
        top.pack(fill="x", padx=2, pady=(2, 0))
        ttk.Label(top, text="Front end").pack(side="left")
        self.fe_var = tk.StringVar(value=FRONTENDS[0] if fe["kind"] == "bs" else FRONTENDS[1])
        cb = ttk.Combobox(top, textvariable=self.fe_var, values=FRONTENDS, state="readonly", width=40)
        cb.pack(side="left", padx=6, fill="x", expand=True)
        cb.bind("<<ComboboxSelected>>", lambda e: self.change_frontend())
        if fe["kind"] == "bs":
            self._frontend_card(fe)
        else:
            wg = self.state["waveguide"]
            card = ttk.LabelFrame(self.scroll.inner, text="Corrugated waveguide (source and receiver)",
                                  style="Card.TLabelframe", padding=8)
            card.pack(fill="x", pady=4, padx=2)
            self._field(card, "x (mm)", ("waveguide", "x"), wg["x"], 0, 0)
            self._field(card, "y (mm)", ("waveguide", "y"), wg["y"], 0, 1)
            self._field(card, "Diameter (mm)", ("waveguide", "diameter"), wg["diameter"], 0, 2)
            self._combo(card, "Mode model", ("waveguide", "model"),
                        "HE11 aperture field" if wg.get("model", "he11") == "he11" else "Ideal Gaussian",
                        ["HE11 aperture field", "Ideal Gaussian"], 2, 0, colspan=3)
        n = len(self.state["components"])
        for i, c in enumerate(self.state["components"]):
            self._component_card(i, c, i == n - 1)
        if n == 0:
            ttk.Label(self.scroll.inner, text="No components yet. Add the first one below.",
                      style="Hint.TLabel").pack(anchor="w", padx=6, pady=8)

    def _ensure_frontend(self):
        self.state.setdefault("frontend", dict(copy.deepcopy(FRONTEND_DEFAULT), kind="single"))
        legacy = "rx_mode" not in self.state["frontend"]
        self.state["frontend"] = normalize_frontend(self.state["frontend"])
        if legacy:                                         # setups saved before the confocal option
            self.state["frontend"]["rx_mode"] = "direct"
        for k, v in FRONTEND_DEFAULT.items():
            if k != "kind":
                self.state["frontend"].setdefault(k, copy.deepcopy(v))
        self.state.setdefault("waveguide", {"x": 0, "y": 0, "diameter": 4, "model": "he11"})

    def _frontend_card(self, fe):
        F = ("frontend",)
        metals = ["Gold", "Copper", "Aluminum"]
        card = ttk.LabelFrame(self.scroll.inner, text="Front end (as built)", style="Card.TLabelframe", padding=8)
        ttk.Label(card, text="Sender → OAP1 → beamsplitter (transmit) → OAP2 → beam path.\n"
                             "Return: OAP2 → beamsplitter (reflect) → return arm → receiver.\n"
                             "Angles are signed, −90 to 90 deg, seen from above. Beamsplitter: + reflects the\n"
                             "return beam toward +y, − toward −y. OAP turn: + turns the beam left (counter-\n"
                             "clockwise), − right (clockwise); its size is the OAP's off-axis angle.",
                  style="Hint.TLabel", justify="left").grid(row=10, column=0, columnspan=3, sticky="w")
        card.pack(fill="x", pady=4, padx=2)

        def box(title, row):
            f = ttk.LabelFrame(card, text=title, padding=6)
            f.grid(row=row, column=0, columnspan=3, sticky="ew", pady=3)
            for col in range(3):
                f.columnconfigure(col, weight=1)
            return f

        h = box("Horns", 0)
        self._field(h, "Sender diameter (mm)", F + ("sender_D",), fe["sender_D"], 0, 0)
        self._field(h, "Receiver diameter (mm)", F + ("receiver_D",), fe["receiver_D"], 0, 1)
        self._combo(h, "Mode model", F + ("model",),
                    "HE11 aperture field" if fe.get("model", "he11") == "he11" else "Ideal Gaussian",
                    ["HE11 aperture field", "Ideal Gaussian"], 0, 2, width=16)
        o1 = box("OAP1 (collimates the sender)", 1)
        self._field(o1, "Sender → OAP1 (mm)", F + ("d_sender",), fe["d_sender"], 0, 0)
        self._field(o1, "EFL (mm)", F + ("oap1_efl",), fe["oap1_efl"], 0, 1)
        self._field(o1, "Diameter (mm)", F + ("oap1_D",), fe["oap1_D"], 0, 2)
        self._combo(o1, "Coating", F + ("oap1_mat",), fe["oap1_mat"].capitalize(), metals, 2, 0, width=9)
        self._field(o1, "OAP1 turn (deg, ±90)", F + ("oap1_angle",), fe["oap1_angle"], 2, 1)
        b = box("Beamsplitter", 2)
        self._field(b, "OAP1 → BS (mm)", F + ("d1",), fe["d1"], 0, 0)
        self._field(b, "Angle of incidence (deg, ±90)", F + ("bs_angle",), fe["bs_angle"], 0, 1)
        self._field(b, "Diameter (mm)", F + ("bs_D",), fe["bs_D"], 0, 2)
        self._combo(b, "Type", F + ("bs_model",),
                    BS_MODELS[0] if fe["bs_model"] == "fixed" else BS_MODELS[1], BS_MODELS, 2, 0, colspan=2,
                    callback=lambda: self._edit(lambda: None), width=26)
        if fe["bs_model"] == "fixed":
            self._field(b, "Reflectance R (%)", F + ("bs_R",), fe["bs_R"], 4, 0)
            self._field(b, "Transmittance T (%)", F + ("bs_T",), fe["bs_T"], 4, 1)
            ttk.Label(b, text="Power fractions. S21 goes as T × R (forward pass transmits, return reflects); "
                              "R + T < 100 % means absorption.", style="Hint.TLabel", wraplength=370,
                      justify="left").grid(row=6, column=0, columnspan=3, sticky="w")
        else:
            self._field(b, "Thickness (mm)", F + ("bs_thickness",), fe["bs_thickness"], 4, 0)
            r = self._eps_ui(b, F + ("bs_eps",), fe["bs_eps"], 4, col=1)
            ttk.Label(b, text="R and T computed for this slab at the angle and polarization chosen, so they vary "
                              "with frequency (etalon fringes). Lateral beam offset is ignored.",
                      style="Hint.TLabel", wraplength=370, justify="left").grid(row=r, column=0, columnspan=3, sticky="w")
        o2 = box("OAP2 (sends the beam down into the beam path)", 3)
        self._field(o2, "BS → OAP2 (mm)", F + ("d2",), fe["d2"], 0, 0)
        self._field(o2, "EFL (mm)", F + ("oap2_efl",), fe["oap2_efl"], 0, 1)
        self._field(o2, "Diameter (mm)", F + ("oap2_D",), fe["oap2_D"], 0, 2)
        self._combo(o2, "Coating", F + ("oap2_mat",), fe["oap2_mat"].capitalize(), metals, 2, 0, width=9)
        self._field(o2, "OAP2 turn (deg, ±90)", F + ("oap2_angle",), fe["oap2_angle"], 2, 1)
        o3 = box("Return arm (after the beamsplitter reflection)", 4)
        self._combo(o3, "Return arm layout", F + ("rx_mode",),
                    RX_MODES[1] if fe.get("rx_mode", "direct") == "confocal" else RX_MODES[0], RX_MODES, 0, 0,
                    colspan=3, callback=lambda: self._edit(lambda: None), width=44)

        def oap_row(parent, i, row, dist_key, dist_label):
            self._field(parent, dist_label, F + (dist_key,), fe[dist_key], row, 0)
            self._field(parent, f"OAP{i} EFL (mm)", F + (f"oap{i}_efl",), fe[f"oap{i}_efl"], row, 1)
            self._field(parent, f"OAP{i} diameter (mm)", F + (f"oap{i}_D",), fe[f"oap{i}_D"], row, 2)
            self._combo(parent, "Coating", F + (f"oap{i}_mat",), fe[f"oap{i}_mat"].capitalize(), metals, row + 2, 0, width=9)
            self._field(parent, f"OAP{i} turn (deg, ±90)", F + (f"oap{i}_angle",), fe[f"oap{i}_angle"], row + 2, 1)

        oap_row(o3, 3, 2, "d3", "BS → OAP3 (mm)")
        if fe.get("rx_mode", "direct") == "confocal":
            self._field(o3, "OAP3 → aperture (mm)", F + ("d_ap",), fe["d_ap"], 6, 0)
            self._field(o3, "Aperture ⌀ (mm, blank = open)", F + ("ap_D",), fe["ap_D"], 6, 1)
            oap_row(o3, 4, 8, "d4", "Aperture → OAP4 (mm)")
            oap_row(o3, 5, 12, "d45", "OAP4 → OAP5 (mm)")
            self._field(o3, "OAP5 → receiver (mm)", F + ("d_receiver",), fe["d_receiver"], 16, 0)
            hint = ("Confocal: OAP3 focuses onto the aperture (put it one EFL away), OAP4 recollimates "
                    "(one EFL after the aperture), OAP5 focuses into the receiver (one EFL away). For a matched "
                    "receiver, f5/f4 = receiver waist / waist at the aperture. The aperture removes the part of "
                    "each beam outside the main focus: defocused back-reflections lose much more than the main beam.")
            hr = 18
        else:
            self._field(o3, "OAP3 → receiver (mm)", F + ("d_receiver",), fe["d_receiver"], 6, 0)
            hint = "Direct: OAP3 focuses straight into the receiver."
            hr = 8
        ttk.Label(o3, text=hint, style="Hint.TLabel", wraplength=370, justify="left").grid(
            row=hr, column=0, columnspan=3, sticky="w")
        try:
            lay = frontend_layout(fe)
            x, y = lay["oap2"]
            od = lay["out_dir"]
            fx, fy = lay["oap2"] + od * float(fe["oap2_efl"])
            msg = (f"Positions follow from these distances and angles, with OAP1 at (0, 0). OAP2 is at "
                   f"({x:.1f}, {y:.1f}) and sends the beam along ({od[0]:.2f}, {od[1]:.2f}); its focus is at "
                   f"({fx:.1f}, {fy:.1f}). Place the beam-path components along that line.")
        except (TypeError, ValueError, KeyError):
            msg = "Positions follow from these distances, with OAP1 at (0, 0)."
        ttk.Label(card, text=msg, style="Hint.TLabel", wraplength=390, justify="left").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(self.scroll.inner, text="Beam path after OAP2", font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w", padx=4, pady=(8, 0))

    def change_frontend(self):
        def do():
            self.state["frontend"]["kind"] = "bs" if self.fe_var.get() == FRONTENDS[0] else "single"
        self._edit(do)

    def _path_tail(self):
        """(last point, beam direction after it, EFL if the last element is an OAP whose turn is unknown)."""
        comps = self.state["components"]
        fe = self.state.get("frontend", {})
        if fe.get("kind") == "bs":
            try:
                lay = frontend_layout(fe)
            except (TypeError, ValueError, KeyError):
                lay = None
            if lay is not None and not comps:
                return lay["oap2"], lay["out_dir"], None, float(fe.get("oap2_efl") or 100)
            pts = [lay["oap2"] if lay is not None else np.zeros(2)]
        else:
            pts = [np.array([float(self.state["waveguide"]["x"] or 0), float(self.state["waveguide"]["y"] or 0)])]
        for c in comps:
            try:
                pts.append(np.array([float(c["x"]), float(c["y"])]))
            except (TypeError, ValueError):
                pass
        a = pts[-1]
        if len(pts) > 1:
            b = pts[-2]
        elif fe.get("kind") == "bs":
            b = a - np.array([0.0, -1.0])
        else:
            b = a - np.array([1.0, 0.0])
        d = (a - b) / (np.linalg.norm(a - b) or 1)
        oap_efl = float(comps[-1].get("efl") or 100) if comps and comps[-1]["type"] == "oap" else None
        return a, d, oap_efl, None

    def _component_card(self, i, c, last):
        P = ("components", i)
        card = ttk.LabelFrame(self.scroll.inner, text=f"{i + 1}. {KIND_LABEL[c['type']]}", style="Card.TLabelframe", padding=8)
        card.pack(fill="x", pady=4, padx=2)
        head = ttk.Frame(card)
        head.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Label(head, text="Name").pack(side="left")
        ttk.Entry(head, textvariable=self._var(P + ("name",), c["name"]), width=14).pack(side="left", padx=4)
        if last:
            ttk.Label(head, text="End reflector", style="End.TLabel").pack(side="left", padx=4)
        ttk.Button(head, text="✕", width=3, command=lambda: self.remove(i)).pack(side="right")
        ttk.Button(head, text="↓", width=3, command=lambda: self.move(i, 1),
                   state="disabled" if last else "normal").pack(side="right")
        ttk.Button(head, text="↑", width=3, command=lambda: self.move(i, -1),
                   state="disabled" if i == 0 else "normal").pack(side="right")
        self._field(card, "x (mm)", P + ("x",), c["x"], 1, 0)
        self._field(card, "y (mm)", P + ("y",), c["y"], 1, 1)
        self._field(card, "Diameter (mm)", P + ("diameter",), c.get("diameter", ""), 1, 2)
        metals = ["Gold", "Copper", "Aluminum"]
        if c["type"] == "mirror":
            self._field(card, "Incidence (deg)" if not last else "Angle (0 = normal)", P + ("angle",), c.get("angle", ""), 3, 0)
            self._combo(card, "Coating", P + ("material",), c.get("material", "gold").capitalize(), metals, 3, 1, width=9)
            ttk.Label(card, text="Retro-reflects at normal incidence." if last else "Angle optional: checked against the positions.",
                      style="Hint.TLabel").grid(row=5, column=0, columnspan=3, sticky="w")
        elif c["type"] == "oap":
            self._field(card, "EFL (mm)", P + ("efl",), c["efl"], 3, 0)
            self._field(card, "Off-axis angle (deg)", P + ("offAxis",), c.get("offAxis", 90), 3, 1)
            self._combo(card, "Coating", P + ("material",), c.get("material", "gold").capitalize(), metals, 3, 2, width=9)
            if last:
                ttk.Label(card, text="An OAP can't end the path. Add a mirror or sample after it.",
                          foreground="#b3261e").grid(row=5, column=0, columnspan=3, sticky="w")
        else:
            row = 3
            if c["type"] == "cryostat":
                wf = ttk.LabelFrame(card, text="Window (up-beam of the sample)", padding=6)
                wf.grid(row=row, column=0, columnspan=3, sticky="ew", pady=3)
                self._field(wf, "Thickness (mm)", P + ("window_thickness",), c.get("window_thickness", 2.0), 0, 0)
                self._field(wf, "Gap to sample (mm)", P + ("gap",), c.get("gap", 20.0), 0, 1)
                self._field(wf, "Diameter (mm)", P + ("window_diameter",), c.get("window_diameter", ""), 0, 2)
                self._field(wf, "Tilt (deg)", P + ("window_tilt",), c.get("window_tilt", 0), 2, 0)
                r2 = self._eps_ui(wf, P + ("window_eps",), c["window_eps"], 2, col=1)
                e = c["window_eps"]
                try:
                    t = float(c.get("window_thickness"))
                    n = np.sqrt(float(e["re"])) if e["kind"] == "const" else None
                    shift = f"This window pushes the focus back by t(1 - 1/n) = {t * (1 - 1 / n):.2f} mm. " if n else ""
                except (TypeError, ValueError, KeyError):
                    shift = ""
                ttk.Label(wf, text=shift + "Gap = window back face to sample front face. Tilt the window to "
                                           "stop its reflection reaching the waveguide.",
                          style="Hint.TLabel", wraplength=370, justify="left").grid(row=r2, column=0, columnspan=4, sticky="w")
                for col in range(3):
                    wf.columnconfigure(col, weight=1)
                row += 1
                ttk.Label(card, text="Sample: x, y above are its front surface, Diameter is the sample's.",
                          style="Hint.TLabel").grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))
                row += 1
            if not last:
                self._field(card, "Sample tilt (deg)" if c["type"] == "cryostat" else "Tilt (deg)",
                            P + ("tilt",), c.get("tilt", 0), row, 0)
                ttk.Label(card, text="Transmits. Tilt it to stop its back-reflection.", style="Hint.TLabel").grid(
                    row=row + 1, column=1, columnspan=2, sticky="w")
                row += 2
            else:
                row = self._backing_ui(card, P, c, i, row)
            for j, L in enumerate(c["layers"]):
                lf = ttk.LabelFrame(card, text=("Sample layer " if c["type"] == "cryostat" else "Layer ") + f"{j + 1}"
                                    + (" (beam side)" if j == 0 else ""), padding=6)
                lf.grid(row=row, column=0, columnspan=3, sticky="ew", pady=3)
                row += 1
                LP = P + ("layers", j)
                self._field(lf, "Thickness (mm)", LP + ("thickness",), L["thickness"], 0, 0)
                self._eps_ui(lf, LP + ("eps",), L["eps"], 0, col=1)
                ttk.Button(lf, text="Remove layer", command=lambda i=i, j=j: self.remove_layer(i, j),
                           state="normal" if len(c["layers"]) > 1 else "disabled").grid(row=0, column=3, rowspan=2, padx=4)
            ttk.Button(card, text="Add layer", command=lambda: self.add_layer(i)).grid(row=row, column=0, sticky="w", pady=(4, 0))
        for col in range(3):
            card.columnconfigure(col, weight=1)

    def _eps_ui(self, parent, epath, e, row, col=1):
        """Dielectric-function chooser plus its parameter fields. Returns the next free row."""
        self._combo(parent, "Dielectric function", epath + ("_choice",),
                    {"const": EPS_KINDS[0], "lorentz": EPS_KINDS[1], "expr": EPS_KINDS[2]}[e["kind"]],
                    EPS_KINDS, row, col, colspan=2, callback=lambda: self.change_eps_at(epath), width=22)
        r = row + 2
        if e["kind"] == "const":
            self._field(parent, "eps'", epath + ("re",), e["re"], r, 0)
            self._field(parent, "eps'' (loss)", epath + ("im",), e["im"], r, 1)
            return r + 2
        if e["kind"] == "expr":
            return self._expr_field(parent, epath, e, r)
        self._field(parent, "eps_inf", epath + ("epsInf",), e["epsInf"], r, 0)
        self._field(parent, "f0 (GHz)", epath + ("f0",), e["f0"], r, 1)
        self._field(parent, "delta eps", epath + ("de",), e["de"], r, 2)
        self._field(parent, "gamma (GHz)", epath + ("gamma",), e["gamma"], r + 2, 0)
        return r + 4

    def _backing_ui(self, card, P, c, i, row):
        b = c.get("backing", {"kind": "vacuum"})
        cur = {"vacuum": BACKINGS[0], "ito": BACKINGS[4], "const": BACKINGS[5], "expr": BACKINGS[6]}.get(
            b["kind"], b.get("metal", "gold").capitalize())
        self._combo(card, "Backing (behind the last layer)", P + ("backing", "_choice"), cur, BACKINGS, row, 0,
                    colspan=2, callback=lambda: self.change_backing(i), width=26)
        row += 2
        BP = P + ("backing",)
        if b["kind"] == "const":
            self._field(card, "eps' backing", BP + ("re",), b["re"], row, 0)
            self._field(card, "eps'' backing", BP + ("im",), b["im"], row, 1)
            row += 2
        elif b["kind"] == "expr":
            row = self._expr_field(card, BP, b, row)
        elif b["kind"] == "ito":
            bf = ttk.LabelFrame(card, text="ITO film (Drude) on a glass substrate", padding=6)
            bf.grid(row=row, column=0, columnspan=3, sticky="ew", pady=3)
            self._field(bf, "Conductivity (S/m)", BP + ("sigma",), b["sigma"], 0, 0, width=10)
            self._field(bf, "Scattering time (fs)", BP + ("tau_fs",), b["tau_fs"], 0, 1, width=10)
            self._field(bf, "Film thickness (nm)", BP + ("thickness_nm",), b.get("thickness_nm", ""), 0, 2, width=10)
            self._field(bf, "Substrate eps'", BP + ("sub_re",), b["sub_re"], 2, 0, width=10)
            self._field(bf, "Substrate eps''", BP + ("sub_im",), b["sub_im"], 2, 1, width=10)
            self._field(bf, "Substrate (mm)", BP + ("sub_thickness",), b.get("sub_thickness", ""), 2, 2, width=10)
            try:
                rs = f"Sheet resistance ≈ {1 / (float(b['sigma']) * float(b['thickness_nm']) * 1e-9):.1f} Ω/sq. "
            except (TypeError, ValueError, ZeroDivisionError):
                rs = ""
            ttk.Label(bf, text=rs + "Blank film thickness = bulk ITO; blank substrate = semi-infinite.\n"
                                    "Defaults: ρ = 2×10⁻⁴ Ω·cm ITO on soda-lime glass (approximate THz values).",
                      style="Hint.TLabel", wraplength=370, justify="left").grid(row=4, column=0, columnspan=3, sticky="w")
            for col in range(3):
                bf.columnconfigure(col, weight=1)
            row += 1
        return row

    def _expr_field(self, parent, path, spec, row):
        ttk.Label(parent, text="eps(w) =").grid(row=row, column=0, columnspan=3, sticky="w")
        ttk.Entry(parent, textvariable=self._var(path + ("expr",), spec.get("expr", EXPR_DEFAULT)),
                  font=("Courier", 10)).grid(row=row + 1, column=0, columnspan=4, sticky="ew", pady=(0, 2))
        ttk.Label(parent, text=EXPR_HINT, style="Hint.TLabel", wraplength=380, justify="left").grid(
            row=row + 2, column=0, columnspan=4, sticky="w")
        return row + 3

    # ----------------------------------------------------------- structure edits
    def _clean_choices(self):
        """Drop helper '_choice' keys created by comboboxes."""
        self.state.get("frontend", {}).get("bs_eps", {}).pop("_choice", None)
        for c in self.state["components"]:
            if c["type"] in ("sample", "cryostat"):
                c.get("backing", {}).pop("_choice", None)
                c.get("window_eps", {}).pop("_choice", None)
                for L in c["layers"]:
                    L["eps"].pop("_choice", None)

    def _edit(self, fn):
        self.sync()
        fn()
        self._clean_choices()
        self.rebuild_path()

    def add_component(self, t):
        def do():
            comps = self.state["components"]
            a, d, oap_efl, focus = self._path_tail()
            if oap_efl is not None:
                d = np.array([d[1], -d[0]])      # assume the OAP turns the beam 90 deg clockwise
            pos = np.round(a + d * 100, 1)
            k = sum(1 for c in comps if c["type"] == t) + 1
            c = {"type": t, "name": {"mirror": "M", "oap": "OAP", "sample": "Sample "}[t] + str(k),
                 "x": float(pos[0]), "y": float(pos[1]), "diameter": 10 if t == "sample" else 50.8}
            if t == "mirror":
                c.update(angle="", material="gold")
            elif t == "oap":
                c.update(efl=101.6, offAxis=90, material="gold")
            else:
                c.update(tilt=0, layers=[{"thickness": 0.5, "eps": {"kind": "const", "re": 10, "im": 0.01}}],
                         backing={"kind": "vacuum"})
            comps.append(c)
        self._edit(do)
        self.after(50, lambda: self.scroll.canvas.yview_moveto(1.0))

    def add_cryostat(self):
        """Append a cryostat component: 2 mm crystalline-quartz window, 20 mm gap, sample on copper.
        Thickness and gap are editable on its card. If the path ends on an OAP the sample goes at
        its focus: one EFL away, plus the ~1 mm focal shift the window introduces."""
        def do():
            comps = self.state["components"]
            a, d, oap_efl, focus = self._path_tail()
            dist = 100.0
            if oap_efl is not None:
                # the OAP turns the beam: assume a clockwise 90 deg turn (edit positions if yours turns the other way)
                d = np.array([d[1], -d[0]])
                dist = oap_efl
            elif focus is not None:             # straight after the front end's OAP2: go to its focus
                dist = focus
            # a slab of thickness t and index n pushes the focus back by t (1 - 1/n): about 1 mm here
            dist += 2.0 * (1 - 1 / np.sqrt(QUARTZ["re"]))
            xy = np.round(a + d * dist, 2)
            k = sum(1 for c in comps if c["type"] == "cryostat") + 1
            comps.append({"type": "cryostat", "name": "Cryostat" + (f" {k}" if k > 1 else ""),
                          "x": float(xy[0]), "y": float(xy[1]), "diameter": 10,
                          "window_thickness": 2.0, "gap": 20.0, "window_diameter": 25.4, "window_tilt": 0,
                          "window_eps": dict(QUARTZ), "tilt": 0,
                          "layers": [{"thickness": 0.5, "eps": {"kind": "const", "re": 10, "im": 0.01}}],
                          "backing": {"kind": "metal", "metal": "copper"}})
        self._edit(do)
        self.after(50, lambda: self.scroll.canvas.yview_moveto(1.0))

    def remove(self, i):
        self._edit(lambda: self.state["components"].pop(i))

    def move(self, i, d):
        def do():
            c = self.state["components"]
            c[i], c[i + d] = c[i + d], c[i]
        self._edit(do)

    def add_layer(self, i):
        self._edit(lambda: self.state["components"][i]["layers"].append(
            {"thickness": 0.1, "eps": {"kind": "const", "re": 10, "im": 0.01}}))

    def remove_layer(self, i, j):
        self._edit(lambda: self.state["components"][i]["layers"].pop(j))

    def change_eps(self, i, j):
        self.change_eps_at(("components", i, "layers", j, "eps"))

    def change_eps_at(self, epath):
        def do():
            holder = self.state
            for k in epath[:-1]:
                holder = holder[k]
            key = epath[-1]
            cur = holder[key]
            choice = cur.get("_choice")
            if choice == EPS_KINDS[0] and cur["kind"] != "const":
                holder[key] = {"kind": "const", "re": 10, "im": 0.01}
            elif choice == EPS_KINDS[1] and cur["kind"] != "lorentz":
                holder[key] = {"kind": "lorentz", "epsInf": 10, "f0": 500, "de": 1, "gamma": 10}
            elif choice == EPS_KINDS[2] and cur["kind"] != "expr":
                holder[key] = {"kind": "expr", "expr": EXPR_DEFAULT}
            elif choice and choice.startswith("Preset: "):
                re_, im_ = PRESETS[choice[len("Preset: "):]]
                holder[key] = {"kind": "const", "re": re_, "im": im_}
        self._edit(do)

    def change_backing(self, i):
        def do():
            c = self.state["components"][i]
            ch = c["backing"].get("_choice")
            if ch == BACKINGS[0]:
                c["backing"] = {"kind": "vacuum"}
            elif ch == BACKINGS[4]:
                c["backing"] = dict(ITO_DEFAULT)
            elif ch == BACKINGS[5]:
                c["backing"] = {"kind": "const", "re": 10, "im": 0.01}
            elif ch == BACKINGS[6]:
                c["backing"] = {"kind": "expr", "expr": "1 + j*5e5/(eps0*w)"}
            else:
                c["backing"] = {"kind": "metal", "metal": ch.lower()}
        self._edit(do)

    # ---------------------------------------------------------------- settings
    def load_run_settings(self):
        r = self.state.get("run", EXAMPLES["Empty path"]["run"])
        self.band_var.set(BAND_CHOICES[list(BANDS).index(r["band"])] if r["band"] in BANDS else "Custom band")
        self.freq_var.set("" if r.get("freq") in ("", None) else str(r["freq"]))
        self.f1_var.set(str(r.get("f1", "")))
        self.f2_var.set(str(r.get("f2", "")))
        self.pol_var.set("E vertical (s)" if r.get("pol", "s") == "s" else "E horizontal (p)")
        self.inline_var.set(bool(r.get("inline", True)))
        self._toggle_custom()

    def save_run_settings(self):
        b = self.band_var.get()
        band = "custom" if b.startswith("Custom") else b.split(":")[0]
        num = lambda s: float(s) if s.strip() not in ("",) else ""
        try:
            f1, f2, fr = num(self.f1_var.get()), num(self.f2_var.get()), num(self.freq_var.get())
        except ValueError:
            f1, f2, fr = self.f1_var.get(), self.f2_var.get(), self.freq_var.get()
        self.state["run"] = {"band": band, "f1": f1, "f2": f2, "freq": fr,
                             "pol": "s" if self.pol_var.get().startswith("E vertical") else "p",
                             "inline": bool(self.inline_var.get())}

    def _toggle_custom(self):
        if self.band_var.get().startswith("Custom"):
            self.custom.grid()
        else:
            self.custom.grid_remove()

    # ---------------------------------------------------------------- files
    def load_setup(self, s):
        if ("waveguide" not in s and "frontend" not in s) or "components" not in s:
            messagebox.showerror("Open setup", "That file isn't a setup saved from this program.")
            return
        s.setdefault("run", copy.deepcopy(EXAMPLES["Empty path"]["run"]))
        self.state = s
        self.rebuild_path()
        self.load_run_settings()

    def open_setup(self):
        p = filedialog.askopenfilename(filetypes=[("Setup JSON", "*.json"), ("All files", "*.*")])
        if p:
            try:
                with open(p) as fh:
                    self.load_setup(json.load(fh))
            except (OSError, json.JSONDecodeError) as err:
                messagebox.showerror("Open setup", f"Couldn't read {p}:\n{err}")

    def save_setup(self):
        self.sync()
        self._clean_choices()
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Setup JSON", "*.json")])
        if p:
            with open(p, "w") as fh:
                json.dump(self.state, fh, indent=2)

    def export_csv(self):
        if not self.result:
            messagebox.showinfo("Export S11", "Run a simulation first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            sw = self.result["sweep"]
            np.savetxt(p, np.column_stack([sw["f"], sw["total"].real, sw["total"].imag]), delimiter=",",
                       header=f"f_GHz,Re_{self.result['bench'].meas},Im_{self.result['bench'].meas}  (VNA convention exp(+j w t))", comments="")

    def save_figures(self):
        if not self.result:
            messagebox.showinfo("Save figures", "Run a simulation first.")
            return
        d = filedialog.askdirectory(title="Folder for layout.png, profile.png, s11.png")
        if d:
            for key, (fig, _) in self.figs.items():
                fig.savefig(f"{d}/{key}.png", dpi=150)

    # ---------------------------------------------------------------- simulate
    def simulate(self):
        self.sync()
        self._clean_choices()
        r = self.state["run"]
        try:
            if r["band"] == "custom":
                if r["f1"] == "" or r["f2"] == "":
                    raise ValueError("Enter a start and stop frequency for the custom band.")
                f1, f2 = float(r["f1"]), float(r["f2"])
                err = validate_band(f1, f2)
                if err:
                    raise ValueError(err)
                label = f"{f1:g}-{f2:g} GHz"
            else:
                f1, f2 = BANDS[r["band"]]
                label = f"{r['band']} ({f1:g}-{f2:g} GHz)"
            setup = {"waveguide": self.state["waveguide"], "components": self.state["components"],
                     "frontend": self.state.get("frontend"), "pol": r["pol"], "inline": r["inline"]}
            bench = Bench(setup)
        except (ValueError, KeyError) as err:
            messagebox.showerror("Can't simulate yet", str(err))
            return
        warnings = list(bench.warnings)
        fp = (f1 + f2) / 2 if r["freq"] in ("", None) else float(r["freq"])
        if not f1 <= fp <= f2:
            warnings.append(f"The beam-profile frequency {fp:g} GHz is outside the band, so the band centre is used.")
            fp = (f1 + f2) / 2
        if bench.rx is not None and bench.rx["D"] < 2 * 299.792458 / f1:
            warnings.append(f"The {bench.rx['D']:g} mm receiver horn is under two wavelengths across at {f1:g} GHz, "
                            "where the Gaussian approximation becomes inaccurate.")
        if bench.wg_D < 2 * 299.792458 / f1:
            warnings.append(f"The {bench.wg_D:g} mm {'sender horn' if bench.rx is not None else 'waveguide'} is under two wavelengths across at {f1:g} GHz, "
                            "where the Gaussian launch becomes inaccurate. A larger guide suits this band better.")
        self.run_btn.configure(state="disabled", text="Simulating…")
        self.status.configure(text="Computing…")

        def work():
            try:
                prof = bench.profile(fp)
                npts = bench.auto_points(f1, f2)
                sw = bench.s11(np.linspace(f1, f2, npts))
                at = bench.s11([fp])
                self.q.put(("ok", dict(bench=bench, prof=prof, sweep=sw, at=at, fp=fp, label=label,
                                       warnings=warnings, npts=npts)))
            except Exception as err:          # report any numerical problem in the GUI
                self.q.put(("err", err))

        threading.Thread(target=work, daemon=True).start()
        self.after(100, self._poll)

    def _poll(self):
        try:
            kind, payload = self.q.get_nowait()
        except queue.Empty:
            self.after(100, self._poll)
            return
        self.run_btn.configure(state="normal", text="Simulate")
        if kind == "err":
            self.status.configure(text="")
            messagebox.showerror("Simulation failed", str(payload))
            return
        self.result = payload
        self.status.configure(text=f"{payload['npts']:,} frequency points, {payload['label']}"
                                   + (f"   ({len(payload['warnings'])} warning(s): see Numbers)" if payload["warnings"] else ""))
        self.render()

    def render(self):
        R = self.result
        bench, prof, sw, fp = R["bench"], R["prof"], R["sweep"], R["fp"]
        fig, cv = self.figs["layout"]
        fig.clf()
        draw_layout(fig.add_subplot(111), bench, prof, fp)
        cv.draw()
        fig, cv = self.figs["profile"]
        fig.clf()
        draw_profile(fig.add_subplot(111), prof, fp)
        cv.draw()
        fig, cv = self.figs["s11"]
        fig.clf()
        a1 = fig.add_subplot(211)
        a2 = fig.add_subplot(212, sharex=a1)
        draw_s11(a1, a2, sw, fp, R["label"], bench.meas)
        cv.draw()
        self.text.delete("1.0", "end")
        self.text.insert("end", self.summary())

    def summary(self):
        R = self.result
        bench, prof, sw, at, fp = R["bench"], R["prof"], R["sweep"], R["at"], R["fp"]
        rec = prof["rec"]
        s = complex(at["total"][0])
        db = 20 * np.log10(np.abs(sw["total"]) + 1e-15)
        L = [f"Band {R['label']}, beam profile at {fp:g} GHz (lambda = {prof['lam_mm']:.4f} mm), "
             f"polarization {'E vertical (s)' if bench.pol == 's' else 'E horizontal (p)'}", ""]
        if R["warnings"]:
            L += ["WARNINGS"] + [f"  - {w}" for w in R["warnings"]] + [""]
        M = bench.meas
        if bench.rx is not None:
            fe = self.state["frontend"]
            L += [f"Front end: sender D = {bench.wg_D:g} mm, receiver D = {bench.rx['D']:g} mm, launch waist w0 = {prof['w0_mm']:.3f} mm, "
                  f"HE11 -> Gaussian power coupling = {prof['launch']:.4f} (each horn)",
                  "Beamsplitter: " + (f"fixed R = {float(fe['bs_R']):g} %, T = {float(fe['bs_T']):g} %"
                                      if fe["bs_model"] == "fixed" else f"{float(fe['bs_thickness']):g} mm dielectric slab")
                  + f" at {float(fe['bs_angle']):g} deg (return reflected toward {'+y' if float(fe['bs_angle']) > 0 else '-y'})",
                  f"Total path sender -> sample -> receiver = {bench.round_trip:.1f} mm", ""]
        else:
            L += [f"Waveguide D = {bench.wg_D:g} mm, launch waist w0 = {prof['w0_mm']:.3f} mm, "
                  f"HE11 -> Gaussian power coupling = {prof['launch']:.4f}",
                  f"Round-trip path = {bench.round_trip:.1f} mm", ""]
        L += [f"|{M}| at {fp:g} GHz = {abs(s):.4f} ({20 * np.log10(abs(s) + 1e-15):.2f} dB), "
              f"phase (delay removed) = {np.degrees(np.angle(at['deemb'][0])):.1f} deg",
              f"Deepest |{M}| in band = {db.min():.2f} dB at {sw['f'][db.argmin()]:.2f} GHz", "",
              f"{'Pass':6s} {'Component':16s} {'s (mm)':>9s} {'w (mm)':>8s} {'AOI (deg)':>10s} {'Transmission':>21s}",
              "  (mirrors: power inside their diameter; spatial-filter aperture: fundamental-mode amplitude)"]
        for e in rec["events"]:
            L.append(f"{'Out' if e['pass'] == 'forward' else 'Back':6s} {e['name'][:16]:16s} {e['s']:9.1f} {e['w']:8.3f} "
                     f"{(np.degrees(bench.aoi[e['idx']]) if e['idx'] is not None else e.get('aoi') or 0.0):10.1f} {100 * e['clip']:20.2f}%")
        L += ["", "Beam waists", f"{'Pass':6s} {'After':16s} {'s (mm)':>9s} {'w0 (mm)':>8s} {'x (mm)':>9s} {'y (mm)':>9s}"]
        for w in rec["waists"]:
            L.append(f"{'Out' if w['pass'] == 'forward' else 'Back':6s} {w['after'][:16]:16s} {w['s']:9.1f} "
                     f"{w['w0']:8.3f} {w['xy'][0]:9.1f} {w['xy'][1]:9.1f}")
        if prof.get("refl"):
            L += ["", "Direct back-reflections off in-line samples (beam traced separately, purple dotted in the plots)",
                  f"  main path for comparison: |{M}| contribution = {abs(prof['main_amp']):.4f}"]
            for r in prof["refl"]:
                L.append(f"  {r['label']}: |{M}| contribution = {abs(r['amp']):.4f} "
                         f"({20 * np.log10(abs(r['amp']) / max(abs(prof['main_amp']), 1e-30)):+.1f} dB vs main path)")
                for e in r["events"]:
                    if e["name"] == "Aperture":
                        mw = next((m["w"] for m in rec["events"] if m["name"] == "Aperture"), None)
                        L.append(f"    at the aperture: w = {e['w']:.2f} mm (main beam {mw:.2f} mm); "
                                 f"fundamental-mode amplitude passed {100 * e['clip']:.1f} %")
                for w in r["waists"]:
                    L.append(f"    waist {w['w0']:.2f} mm at ({w['xy'][0]:.1f}, {w['xy'][1]:.1f}) after {w['after']}")
        if len(at["groups"]) > 1:
            L += ["", f"{M} contributions at {fp:g} GHz (grouped by number of sample reflections)"]
            for g, v in at["groups"].items():
                v = complex(v[0])
                L.append(f"  {'main path' if g == 0 else f'{g} reflection(s)':18s} |S| = {abs(v):.4f}  "
                         f"phase = {np.degrees(np.angle(v)):7.1f} deg")
        return "\n".join(L) + "\n"


if __name__ == "__main__":
    App().mainloop()