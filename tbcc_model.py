"""
tbcc_model.py  -  Physics core for the TBCC Engine Lab
=======================================================
Turbine-Based Combined Cycle = a turbojet flowpath (take-off .. ~Mach 3) plus a
ramjet / dual-mode flowpath (~Mach 2.5 .. 6) that share the inlet envelope and are
blended through a mode-transition schedule.

Division of labour
  * NumPy    : standard atmosphere, ram-recovery, schedules, blending, sweeps and a
               complete built-in cycle model (used as warm-start + automatic fallback).
  * OpenMDAO : model / solver framework (Newton + BalanceComp) for the real cycles.
  * pyCycle  : real-gas (CEA / JP-7) thermodynamics for inlet, compressor, burner,
               turbine and C-D nozzle elements.

Scramjet (supersonic-combustion) operation is NOT modelled - pyCycle's combustor
needs a subsonic entry state - so the upper Mach limit is 6.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Optional heavy dependencies
# --------------------------------------------------------------------------- #
try:
    import openmdao.api as om
    import pycycle.api as pyc

    PYCYCLE_OK, PYCYCLE_ERR = True, ""
except Exception as exc:  # noqa: BLE001
    PYCYCLE_OK, PYCYCLE_ERR = False, f"{type(exc).__name__}: {exc}"

# --------------------------------------------------------------------------- #
# Constants (SI)
# --------------------------------------------------------------------------- #
G0 = 9.80665
R_AIR = 287.05
GAMMA_AIR = 1.4
CP_C = 1004.5          # cold-section cp  [J/kg/K]
CP_H = 1156.7          # hot-section  cp  (gamma = 1.33)
GAM_H = 1.33
LHV = 43.2e6           # fuel lower heating value [J/kg]
ETA_C, ETA_T, ETA_B, ETA_N, ETA_M = 0.85, 0.88, 0.98, 0.98, 0.99
DP_B_TJ, DP_B_RJ = 0.05, 0.07
PHI_RJ = 0.95          # ramjet capture / mass-flow ratio

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EngineParams:
    # turbojet
    w_corr: float = 45.0      # corrected airflow at sea-level static  [kg/s]
    a_tj: float = 0.45        # turbojet inlet capture area             [m2]
    pr_des: float = 10.0      # max compressor pressure ratio
    t4_tj: float = 1650.0     # max turbine-inlet temperature           [K]
    t3_max: float = 900.0     # compressor-exit temperature limit       [K]
    # ramjet / dual-mode
    a_rj: float = 0.60        # ramjet inlet capture area               [m2]
    t4_rj: float = 2200.0     # max combustor-exit temperature          [K]
    # scheduling
    m_t0: float = 2.5         # mode transition start (Mach)
    m_t1: float = 3.5         # mode transition end   (Mach)
    throttle: float = 1.0     # 0.5 .. 1.0  (scales T4 between Tt3 and T4 max)


# --------------------------------------------------------------------------- #
# Atmosphere / flight condition  (NumPy, vectorised)
# --------------------------------------------------------------------------- #
def isa(h_m):
    """ISA temperature [K] and pressure [Pa], 0-32 km."""
    h = np.clip(np.asarray(h_m, dtype=float), 0.0, 32000.0)
    T = np.where(h <= 11000, 288.15 - 0.0065 * h,
                 np.where(h <= 20000, 216.65, 216.65 + 0.001 * (h - 20000)))
    P = np.where(h <= 11000, 101325.0 * (T / 288.15) ** 5.25588,
                 np.where(h <= 20000,
                          22632.06 * np.exp(-G0 * (h - 11000) / (R_AIR * 216.65)),
                          5474.889 * (T / 216.65) ** -34.1632))
    return T, P


_H_GRID = np.linspace(0.0, 32000.0, 1281)
_P_GRID = isa(_H_GRID)[1]


def alt_for_q(mach: float, q_pa: float) -> float:
    """Altitude [m] that gives a constant dynamic-pressure flight corridor."""
    m = max(float(mach), 0.3)
    p_target = 2.0 * q_pa / (GAMMA_AIR * m * m)
    return float(np.interp(-p_target, -_P_GRID, _H_GRID))


def inlet_recovery(mach):
    """MIL-E-5007D total-pressure recovery (x0.99 duct loss)."""
    m = np.asarray(mach, dtype=float)
    r = np.where(m <= 1.0, 1.0,
                 np.where(m < 5.0, 1.0 - 0.075 * np.maximum(m - 1.0, 0.0) ** 1.35,
                          800.0 / (m ** 4 + 935.0)))
    r = 0.99 * r
    return float(r) if r.ndim == 0 else r


@dataclass
class Flight:
    M: float
    alt: float      # m
    T0: float
    P0: float
    a: float
    V: float
    rho: float
    q: float
    Tt0: float
    Pt0: float


def flight(mach: float, alt_m: float) -> Flight:
    T, P = isa(alt_m)
    T, P = float(T), float(P)
    a = float(np.sqrt(GAMMA_AIR * R_AIR * T))
    V = mach * a
    rho = P / (R_AIR * T)
    ram = 1.0 + 0.2 * mach ** 2
    return Flight(mach, float(alt_m), T, P, a, V, rho, 0.5 * rho * V ** 2,
                  T * ram, P * ram ** 3.5)


def mode_weights(mach: float, p: EngineParams) -> tuple[float, float]:
    """(turbojet, ramjet) airflow weights: smoothstep across the transition band."""
    x = float(np.clip((mach - p.m_t0) / max(p.m_t1 - p.m_t0, 1e-6), 0.0, 1.0))
    s = x * x * (3.0 - 2.0 * x)
    return 1.0 - s, s


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class ModeResult:
    F: float                 # net thrust                [N]
    mdot_f: float            # fuel flow                 [kg/s]
    W: float                 # air flow                  [kg/s]
    Ve: float                # exhaust velocity          [m/s]
    stations: list           # [(label, Tt[K], Pt[kPa]), ...]
    backend: str = "NumPy"


@dataclass
class Point:
    fl: Flight
    w_tj: float
    w_rj: float
    tj: ModeResult | None = None
    rj: ModeResult | None = None
    notes: list = field(default_factory=list)

    def _w(self, attr: str, kind: str) -> float:
        r = getattr(self, kind)
        return (self.w_tj if kind == "tj" else self.w_rj) * getattr(r, attr) if r else 0.0

    @property
    def F_tj(self): return self._w("F", "tj")
    @property
    def F_rj(self): return self._w("F", "rj")
    @property
    def F(self): return self.F_tj + self.F_rj
    @property
    def mdot_f(self): return self._w("mdot_f", "tj") + self._w("mdot_f", "rj")
    @property
    def W(self): return self._w("W", "tj") + self._w("W", "rj")
    @property
    def isp(self): return self.F / (self.mdot_f * G0) if self.mdot_f > 1e-9 else 0.0
    @property
    def tsfc(self): return self.mdot_f / self.F * 1e6 if self.F > 1.0 else 0.0  # g/(kN s)

    @property
    def mode(self) -> str:
        if self.w_tj > 0.995: return "TURBOJET"
        if self.w_rj > 0.995: return "RAMJET"
        return "MODE TRANSITION"

    @property
    def backend(self) -> str:
        used = {r.backend for r in (self.tj, self.rj) if r}
        return used.pop() if len(used) == 1 else ("Mixed" if used else "-")


# --------------------------------------------------------------------------- #
# Boundary conditions (NumPy scheduling, shared by both back-ends)
# --------------------------------------------------------------------------- #
def _fuel_air(Tt3, Tt4):
    return (CP_H * Tt4 - CP_C * Tt3) / (ETA_B * LHV - CP_H * Tt4)


def tj_boundary(fl: Flight, p: EngineParams) -> dict:
    Tt2 = fl.Tt0
    rec = inlet_recovery(fl.M)
    Pt2 = fl.Pt0 * rec
    W_eng = p.w_corr * (Pt2 / 101325.0) / np.sqrt(Tt2 / 288.15)
    W_cap = fl.rho * max(fl.V, 0.5 * fl.a) * p.a_tj
    W = float(min(W_eng, W_cap))

    base = 1.0 + ETA_C * (p.t3_max / Tt2 - 1.0)
    PR = float(np.clip(base ** 3.5, 1.05, p.pr_des)) if base > 1.0 else 1.05
    for _ in range(60):
        Tt3 = Tt2 * (1.0 + (PR ** 0.285714 - 1.0) / ETA_C)
        T4 = max(Tt3 + 100.0, Tt3 + p.throttle * (p.t4_tj - Tt3))
        f = _fuel_air(Tt3, T4)
        dT = CP_C * (Tt3 - Tt2) / ((1.0 + f) * CP_H * ETA_M)
        if dT <= 0.45 * T4:
            break
        PR = max(1.05, PR * 0.92)
    return dict(W=W, rec=rec, Tt2=Tt2, Pt2=Pt2, PR=PR, T4=T4, f=f, Tt3=Tt3)


def rj_boundary(fl: Flight, p: EngineParams) -> dict | None:
    W = fl.rho * fl.V * p.a_rj * PHI_RJ
    if W < 1e-3:
        return None
    Tt2 = fl.Tt0
    rec = inlet_recovery(fl.M)
    Pt2 = fl.Pt0 * rec
    T4 = max(Tt2 + 100.0, Tt2 + p.throttle * (p.t4_rj - Tt2))
    f = _fuel_air(Tt2, T4)
    return dict(W=float(W), rec=rec, Tt2=Tt2, Pt2=Pt2, T4=T4, f=f)


# --------------------------------------------------------------------------- #
# NumPy cycle models
# --------------------------------------------------------------------------- #
def _nozzle_ve(Tt5, Pt5, P0):
    npr = Pt5 / P0
    if npr <= 1.0:
        return 0.0
    return float(np.sqrt(2.0 * CP_H * Tt5 * ETA_N * (1.0 - npr ** (-(GAM_H - 1.0) / GAM_H))))


def tj_numpy(bc: dict, fl: Flight) -> ModeResult:
    Tt2, Pt2, PR, T4, f, Tt3, W = (bc[k] for k in ("Tt2", "Pt2", "PR", "T4", "f", "Tt3", "W"))
    Pt3 = Pt2 * PR
    Pt4 = Pt3 * (1.0 - DP_B_TJ)
    dT = CP_C * (Tt3 - Tt2) / ((1.0 + f) * CP_H * ETA_M)
    Tt5 = T4 - dT
    er = (1.0 - dT / (ETA_T * T4)) ** (GAM_H / (GAM_H - 1.0))
    Pt5 = Pt4 * er
    Ve = _nozzle_ve(Tt5, Pt5, fl.P0)
    F = W * ((1.0 + f) * Ve - fl.V)
    st = [("0  Freestream", fl.Tt0, fl.Pt0 / 1e3), ("2  Inlet exit", Tt2, Pt2 / 1e3),
          ("3  Compressor exit", Tt3, Pt3 / 1e3), ("4  Burner exit", T4, Pt4 / 1e3),
          ("5  Turbine exit", Tt5, Pt5 / 1e3)]
    return ModeResult(F, f * W, W, Ve, st, "NumPy")


def rj_numpy(bc: dict, fl: Flight) -> ModeResult:
    Tt2, Pt2, T4, f, W = (bc[k] for k in ("Tt2", "Pt2", "T4", "f", "W"))
    Pt4 = Pt2 * (1.0 - DP_B_RJ)
    Ve = _nozzle_ve(T4, Pt4, fl.P0)
    F = W * ((1.0 + f) * Ve - fl.V)
    st = [("0  Freestream", fl.Tt0, fl.Pt0 / 1e3), ("2  Diffuser exit", Tt2, Pt2 / 1e3),
          ("4  Burner exit", T4, Pt4 / 1e3)]
    return ModeResult(F, f * W, W, Ve, st, "NumPy")


# --------------------------------------------------------------------------- #
# OpenMDAO + pyCycle models  (design-point cycles solved at every flight point)
# --------------------------------------------------------------------------- #
_LOCK = threading.RLock()
_PROBS: dict = {}

if PYCYCLE_OK:

    def _attach_newton(model):
        nl = model.nonlinear_solver = om.NewtonSolver()
        nl.options["atol"] = 1e-6
        nl.options["rtol"] = 1e-6
        nl.options["iprint"] = -1
        nl.options["maxiter"] = 30
        nl.options["solve_subsystems"] = True
        nl.options["max_sub_solves"] = 100
        nl.options["reraise_child_analysiserror"] = False
        nl.linesearch = om.BoundsEnforceLS()
        nl.linesearch.options["bound_enforcement"] = "scalar"
        nl.linesearch.options["iprint"] = -1
        model.linear_solver = om.DirectSolver(assemble_jac=True)

    class TBCCTurbojet(pyc.Cycle):
        """fc -> inlet -> compressor -> burner -> turbine -> C-D nozzle (single spool)."""

        def setup(self):
            self.options["thermo_method"] = "CEA"
            self.options["thermo_data"] = pyc.species_data.janaf

            self.add_subsystem("fc", pyc.FlightConditions())
            self.add_subsystem("inlet", pyc.Inlet())
            self.add_subsystem("comp", pyc.Compressor(map_data=pyc.AXI5, map_extrap=True),
                               promotes_inputs=["Nmech"])
            self.add_subsystem("burner", pyc.Combustor(fuel_type="JP-7"))
            self.add_subsystem("turb", pyc.Turbine(map_data=pyc.LPT2269),
                               promotes_inputs=["Nmech"])
            self.add_subsystem("nozz", pyc.Nozzle(nozzType="CD", lossCoef="Cv"))
            self.add_subsystem("shaft", pyc.Shaft(num_ports=2), promotes_inputs=["Nmech"])
            self.add_subsystem("perf", pyc.Performance(num_nozzles=1, num_burners=1))

            self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I", connect_w=False)
            self.pyc_connect_flow("inlet.Fl_O", "comp.Fl_I")
            self.pyc_connect_flow("comp.Fl_O", "burner.Fl_I")
            self.pyc_connect_flow("burner.Fl_O", "turb.Fl_I")
            self.pyc_connect_flow("turb.Fl_O", "nozz.Fl_I")
            self.connect("comp.trq", "shaft.trq_0")
            self.connect("turb.trq", "shaft.trq_1")
            self.connect("fc.Fl_O:stat:P", "nozz.Ps_exhaust")
            self.connect("inlet.Fl_O:tot:P", "perf.Pt2")
            self.connect("comp.Fl_O:tot:P", "perf.Pt3")
            self.connect("burner.Wfuel", "perf.Wfuel_0")
            self.connect("inlet.F_ram", "perf.ram_drag")
            self.connect("nozz.Fg", "perf.Fg_0")

            bal = self.add_subsystem("balance", om.BalanceComp())
            bal.add_balance("FAR", eq_units="degR", lower=1e-4, val=0.025)
            self.connect("balance.FAR", "burner.Fl_I:FAR")
            self.connect("burner.Fl_O:tot:T", "balance.lhs:FAR")
            bal.add_balance("turb_PR", val=2.0, lower=1.001, upper=10, eq_units="hp", rhs_val=0.0)
            self.connect("balance.turb_PR", "turb.PR")
            self.connect("shaft.pwr_net", "balance.lhs:turb_PR")

            self.set_order(["balance", "fc", "inlet", "comp", "burner", "turb",
                            "nozz", "shaft", "perf"])
            _attach_newton(self)

    class TBCCRamjet(pyc.Cycle):
        """fc -> inlet -> burner -> C-D nozzle (no rotating machinery)."""

        def setup(self):
            self.options["thermo_method"] = "CEA"
            self.options["thermo_data"] = pyc.species_data.janaf

            self.add_subsystem("fc", pyc.FlightConditions())
            self.add_subsystem("inlet", pyc.Inlet())
            self.add_subsystem("burner", pyc.Combustor(fuel_type="JP-7"))
            self.add_subsystem("nozz", pyc.Nozzle(nozzType="CD", lossCoef="Cv"))
            self.add_subsystem("perf", pyc.Performance(num_nozzles=1, num_burners=1))

            self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I", connect_w=False)
            self.pyc_connect_flow("inlet.Fl_O", "burner.Fl_I")
            self.pyc_connect_flow("burner.Fl_O", "nozz.Fl_I")
            self.connect("fc.Fl_O:stat:P", "nozz.Ps_exhaust")
            self.connect("inlet.Fl_O:tot:P", "perf.Pt2")
            self.connect("inlet.Fl_O:tot:P", "perf.Pt3")
            self.connect("burner.Wfuel", "perf.Wfuel_0")
            self.connect("inlet.F_ram", "perf.ram_drag")
            self.connect("nozz.Fg", "perf.Fg_0")

            bal = self.add_subsystem("balance", om.BalanceComp())
            bal.add_balance("FAR", eq_units="degR", lower=1e-4, val=0.04)
            self.connect("balance.FAR", "burner.Fl_I:FAR")
            self.connect("burner.Fl_O:tot:T", "balance.lhs:FAR")

            self.set_order(["balance", "fc", "inlet", "burner", "nozz", "perf"])
            _attach_newton(self)

    def _get_prob(kind: str):
        with _LOCK:
            if kind not in _PROBS:
                try:
                    prob = om.Problem(reports=False)
                except TypeError:                       # older OpenMDAO
                    prob = om.Problem()
                prob.model = TBCCTurbojet() if kind == "tj" else TBCCRamjet()
                prob.setup(check=False)
                prob.set_solver_print(level=-1)
                _PROBS[kind] = prob
            return _PROBS[kind]

    def _val(prob, name, units=None) -> float:
        return float(np.squeeze(prob.get_val(name, units=units)))

    def _stations(prob, items):
        return [(lbl, _val(prob, f"{path}:tot:T", "K"), _val(prob, f"{path}:tot:P", "kPa"))
                for lbl, path in items]

    def tj_pycycle(bc: dict, fl: Flight) -> ModeResult:
        with _LOCK:
            prob = _get_prob("tj")
            prob.set_val("fc.alt", fl.alt, units="m")
            prob.set_val("fc.MN", max(fl.M, 1e-4))
            prob.set_val("inlet.ram_recovery", bc["rec"])
            prob.set_val("inlet.MN", 0.60)
            prob.set_val("inlet.Fl_I:stat:W", bc["W"], units="kg/s")
            prob.set_val("comp.PR", bc["PR"])
            prob.set_val("comp.eff", ETA_C)
            prob.set_val("comp.MN", 0.30)
            prob.set_val("burner.MN", 0.10)
            prob.set_val("burner.dPqP", DP_B_TJ)
            prob.set_val("turb.eff", ETA_T)
            prob.set_val("turb.MN", 0.40)
            prob.set_val("nozz.Cv", ETA_N)
            prob.set_val("Nmech", 8000.0, units="rpm")
            prob.set_val("balance.rhs:FAR", bc["T4"], units="K")
            prob.set_val("balance.FAR", bc["f"])                       # warm start (NumPy)
            prob.run_model()

            T4 = _val(prob, "burner.Fl_O:tot:T", "K")
            net = _val(prob, "shaft.pwr_net", "kW")
            if abs(T4 - bc["T4"]) > 5.0 or abs(net) > 25.0:
                raise RuntimeError(f"pyCycle turbojet did not converge (T4 err {T4 - bc['T4']:.1f} K, "
                                   f"shaft residual {net:.1f} kW)")
            F = _val(prob, "perf.Fn", "N")
            mf = _val(prob, "burner.Wfuel", "kg/s")
            Ve = _val(prob, "nozz.Fl_O:stat:V", "m/s")
            st = _stations(prob, [("0  Freestream", "fc.Fl_O"), ("2  Inlet exit", "inlet.Fl_O"),
                                  ("3  Compressor exit", "comp.Fl_O"),
                                  ("4  Burner exit", "burner.Fl_O"),
                                  ("5  Turbine exit", "turb.Fl_O")])
        return ModeResult(F, mf, bc["W"], Ve, st, "pyCycle")

    def rj_pycycle(bc: dict, fl: Flight) -> ModeResult:
        with _LOCK:
            prob = _get_prob("rj")
            prob.set_val("fc.alt", fl.alt, units="m")
            prob.set_val("fc.MN", max(fl.M, 1e-4))
            prob.set_val("inlet.ram_recovery", bc["rec"])
            prob.set_val("inlet.MN", 0.40)
            prob.set_val("inlet.Fl_I:stat:W", bc["W"], units="kg/s")
            prob.set_val("burner.MN", 0.30)
            prob.set_val("burner.dPqP", DP_B_RJ)
            prob.set_val("nozz.Cv", ETA_N)
            prob.set_val("balance.rhs:FAR", bc["T4"], units="K")
            prob.set_val("balance.FAR", bc["f"])
            prob.run_model()

            T4 = _val(prob, "burner.Fl_O:tot:T", "K")
            if abs(T4 - bc["T4"]) > 5.0:
                raise RuntimeError(f"pyCycle ramjet did not converge (T4 err {T4 - bc['T4']:.1f} K)")
            F = _val(prob, "perf.Fn", "N")
            mf = _val(prob, "burner.Wfuel", "kg/s")
            Ve = _val(prob, "nozz.Fl_O:stat:V", "m/s")
            st = _stations(prob, [("0  Freestream", "fc.Fl_O"), ("2  Diffuser exit", "inlet.Fl_O"),
                                  ("4  Burner exit", "burner.Fl_O")])
        return ModeResult(F, mf, bc["W"], Ve, st, "pyCycle")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def evaluate(mach: float, alt_m: float, p: EngineParams, backend: str = "numpy") -> Point:
    """Evaluate the full TBCC at one flight point."""
    fl = flight(mach, alt_m)
    w_tj, w_rj = mode_weights(mach, p)
    pt = Point(fl, w_tj, w_rj)
    use_pc = backend == "pycycle" and PYCYCLE_OK

    def run(kind, bc, np_fn, pc_fn):
        if use_pc:
            try:
                return pc_fn(bc, fl)
            except Exception as exc:  # noqa: BLE001
                pt.notes.append(f"{kind}: pyCycle failed ({type(exc).__name__}: {str(exc)[:160]}) "
                                f"- NumPy fallback used")
        return np_fn(bc, fl)

    if w_tj > 1e-4:
        pt.tj = run("turbojet", tj_boundary(fl, p), tj_numpy, tj_pycycle if PYCYCLE_OK else None)
    if w_rj > 1e-4:
        bc = rj_boundary(fl, p)
        if bc is not None:
            pt.rj = run("ramjet", bc, rj_numpy, rj_pycycle if PYCYCLE_OK else None)
    return pt


def sweep(machs, alt_fn, p: EngineParams, backend: str = "numpy") -> dict:
    """Mach sweep. alt_fn(M) -> altitude [m]."""
    machs = np.asarray(machs, dtype=float)
    keys = ("F", "F_tj", "F_rj", "isp", "mdot_f", "alt")
    out = {k: np.zeros_like(machs) for k in keys}
    notes: list = []
    for i, m in enumerate(machs):
        h = alt_fn(m)
        pt = evaluate(float(m), h, p, backend)
        out["F"][i], out["F_tj"][i], out["F_rj"][i] = pt.F, pt.F_tj, pt.F_rj
        out["isp"][i], out["mdot_f"][i], out["alt"][i] = pt.isp, pt.mdot_f, h
        notes += pt.notes
    out["M"] = machs
    out["notes"] = sorted(set(notes))
    return out
