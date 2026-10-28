"""
Interactive viewer for the denominator of the gradient amplifier's current
loop. Component names follow Diagramas_Gradientes/circuito_gradientes.py.

==============================================================================
1) Circuit model (ideal op-amps)
==============================================================================
  Vin   voltage into the controller: output of U2, a unity-gain difference
        amplifier (R17 = R18 = R19 = R20).
  U1    inverting stage with its summing node S at virtual ground. Inputs:
          - Vin through R5
          - the shunt voltage, V_sh = I*R4, through R3
        U1 feedback:  Zf = (RF + 1/(s*CF)) || 1/(s*C1)
  Output stage: unity-gain follower  ->  Vout = Vc  (Vc = U1 output)
  Load: the coil (L, R_L) in series with the shunt R4 and with the equivalent
        emitter resistance of the parallel power branches:
            Re_eq   = 1 / (1/Re_1 + 1/Re_2 + ... + 1/Re_N)
            R_total = R_L + R4 + Re_eq
        (both halves of the output stage are assumed to have the same branches)
        =>  I = Vc / (R_total + s*L)

==============================================================================
2) Closed loop I/Vin and gain
==============================================================================
  Node S:   Vin/R5 + I*R4/R3 + Vc/Zf = 0   =>   Vc = -Zf*(Vin/R5 + I*R4/R3)
  Load:     Vc = I*(R_total + s*L)
  Equating, solving for I/Vin and multiplying top and bottom by R3/(R4*Zf):

      I/Vin = -G / D(s)        D(s) = 1 + (R3/R4)*(R_total + s*L)/Zf

  Transconductance gain: in steady state CF blocks DC (Zf -> infinity), so at
  node S only the input current (through R5) and the feedback current
  (through R3) cancel:

      Vin/R5 = -I*R4/R3      =>      I = -G*Vin,      G = R3/(R4*R5)  [A/V]

      G = 1 A/V   <=>   R3/R4 = R5        (with R4 = 0.1 Ohm: R5 = 10*R3)
      for a chosen G:    R5 = R3/(R4*G)

  R5 only appears in the numerator: G scales the amplitude of I but does NOT
  move the poles.

==============================================================================
3) Pole-zero cancellation
==============================================================================
  1/Zf          = s*(C1 + CF + s*C1*CF*RF) / (1 + s*CF*RF)
  R_total + s*L = R_total*(1 + s*L/R_total)
  Imposing CF*RF = L/R_total, the factors (1 + s*CF*RF) and (1 + s*L/R_total)
  cancel and D(s) becomes second order:

      K = R3*R_total/R4          [Ohm]
      D(s) = a*s^2 + b*s + c,    a = K*C1*CF*RF,   b = K*(C1 + CF),   c = 1

  The roots of D(s) are the closed-loop poles. If they are complex:
      wn = sqrt(c/a) [rad/s]        zeta = b / (2*sqrt(a*c))
  The polynomial is only valid while CF*RF = L/R_total holds; the panel warns
  when it does not (for example, when a slider hits its limit).

==============================================================================
4) Controls
==============================================================================
  Sliders:   R3, RF, C1, CF. RF and CF are coupled: moving one recomputes the
             other so that CF*RF = L/R_total keeps holding.
  Boxes:     L, R_L, Re of each branch (comma-separated list) and G.
             Changing L, R_L or the Re values keeps CF and recomputes RF.
             R5 follows from G: moving R3 keeps G and recomputes R5.
  Selector:  x-axis range.
  Table:     value of every component being assigned and what sets it.
  The plot shows y = D(x) with x = s real, on a symlog scale (logarithmic on
  both sides of zero and linear near it).

Usage:
    pip install numpy matplotlib
    python polinomio.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox

# ===== fixed constant =====
R4 = 0.1                # current-sense shunt [Ohm]

# ===== initial values =====
R3_0 = 270              # feedback resistor from the shunt [Ohm]
RF_0 = 10e3             # compensation resistor [Ohm]
C1_0 = 10e-12           # parallel capacitor of the network [F]
CF_0 = 10e-9            # capacitor in series with RF [F]
L_0 = 25e-6             # coil inductance [H]
R_L_0 = 0.4             # coil resistance [Ohm]
RE_0 = [0.1, 0.1]       # emitter resistor of each parallel branch [Ohm]
G_0 = 1.0               # transconductance gain I/Vin [A/V]

# slider limits
R3_MIN, R3_MAX = 1, 1000
RF_MIN, RF_MAX = 1e3, 100e3
C1_MIN, C1_MAX = 10e-12, 1000e-12
CF_MIN, CF_MAX = 100e-12, 100e-9

# colors
COL_TRACE = "#0a9c8e"
COL_ROOT = "#dd4636"
COL_WN = "#c07a00"

# values typed into the text boxes (the sliders hold the rest)
state = {
    "L": L_0,                  # [H]
    "R_L": R_L_0,              # [Ohm]
    "Re": list(RE_0),          # [Ohm], one per branch
    "G": G_0,                  # [A/V]
    "mode": "Auto (decades)",  # x-axis range
    "lock": False,             # keeps on_rf and on_cf from calling each other
}


# ===== model calculations (sections 1 to 3 of the header) =====
def equivalent_emitter_resistance(Re):
    """Parallel branches: Re_eq = 1 / (1/Re_1 + ... + 1/Re_N)."""
    total_conductance = 0.0
    for r in Re:
        total_conductance += 1.0 / r
    return 1.0 / total_conductance


def total_resistance():
    """Everything in series with the coil: R_L + R4 + Re_eq."""
    return state["R_L"] + R4 + equivalent_emitter_resistance(state["Re"])


def load_tau():
    """Load time constant L/R_total; the cancellation requires CF*RF = tau."""
    return state["L"] / total_resistance()


def compute():
    """Computes every model quantity from the current values."""
    R3, RF, C1, CF = sR3.val, sRF.val, sC1.val, sCF.val
    G = state["G"]

    Re_eq = equivalent_emitter_resistance(state["Re"])
    R_total = total_resistance()
    tau = load_tau()
    R5 = R3 / (R4 * G)              # from G = R3/(R4*R5)

    K = R3 * R_total / R4           # D(s) = a*s^2 + b*s + c
    a = K * C1 * CF * RF
    b = K * (C1 + CF)
    c = 1.0
    disc = b * b - 4 * a * c        # discriminant: < 0 => complex poles

    return {"R3": R3, "RF": RF, "C1": C1, "CF": CF, "G": G, "R5": R5,
            "Re_eq": Re_eq, "R_total": R_total, "tau": tau, "K": K,
            "a": a, "b": b, "c": c, "disc": disc}


def real_roots(a, b, c):
    """Real roots of a*x^2 + b*x + c (quadratic formula).
    Returns an empty list if they are complex."""
    if a == 0:
        return [] if b == 0 else [-c / b]
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    r = np.sqrt(disc)
    return sorted([(-b + r) / (2 * a), (-b - r) / (2 * a)])


def clamp(v, lo, hi):
    """Limits v to a slider's range [lo, hi]."""
    return max(lo, min(hi, v))


def si(value, unit):
    """Value with an SI prefix, e.g. si(4.7e-9, "F") -> '4.7 nF'."""
    for factor, prefix in ((1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"),
                           (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")):
        if abs(value) >= factor:
            return f"{value / factor:.3g} {prefix}{unit}"
    return f"{value:.3g} {unit}"


# ===== x axis on a symlog scale =====
def x_window(rts, mode, a, c):
    """Returns (xmin, xmax, linthresh) for the x axis.
    In Auto mode the range goes from -6*|largest root| to +0.6*|largest root|
    (stable poles are negative) and the linear band is |smallest root|/60.
    Without real roots wn is used as the scale."""
    if mode == "+-1e7":
        return -1e7, 1e7, 1e3
    if mode == "+-2e6":
        return -2e6, 2e6, 1e2
    mags = [abs(r) for r in rts if r != 0]
    if not mags and a > 0 and c > 0:
        mags = [np.sqrt(c / a)]
    if not mags:
        return -1e7, 1e7, 1e3
    mn, mx = min(mags), max(mags)
    lt = max(mn / 60.0, 1.0)
    return -mx * 6.0, mx * 0.6, lt


def samples(xmin, xmax, lt):
    """Points where the curve is evaluated: log-spaced on each side and
    linearly spaced in the band [-lt, lt], so it looks smooth in symlog."""
    pts = {0.0, xmin, xmax}
    pts.update(np.linspace(-lt, lt, 80))
    if xmin < -lt:
        pts.update(-np.logspace(np.log10(lt), np.log10(-xmin), 700))
    if xmax > lt:
        pts.update(np.logspace(np.log10(lt), np.log10(xmax), 700))
    x = np.array(sorted(pts))
    return x[(x >= xmin) & (x <= xmax)]


# ===== figure and layout =====
# three columns: plot and sliders | component table | inputs and readings
fig = plt.figure(figsize=(15, 8.5))
fig.suptitle("Denominator of I/Vin = -G / D(s)   (symlog x axis)",
             fontsize=13, fontweight="bold")

ax = fig.add_axes([0.05, 0.44, 0.50, 0.46])
ax.set_xlabel("x  (variable s)   [symlog scale]")
ax.set_ylabel("y = D(x)")
ax.grid(True, which="both", alpha=0.3)
ax.axhline(0, color="0.55", lw=0.8)

(trace,) = ax.plot([], [], color=COL_TRACE, lw=2, label="trace y(x)")
(roots_plot,) = ax.plot([], [], "o", color=COL_ROOT, ms=9,
                        mec="white", mew=1.2, label="roots (poles)")
wn_line = ax.axvline(0, color=COL_WN, lw=1.2, ls="--",
                     label="-wn (complex poles)")
wn_line.set_visible(False)
ax.legend(loc="upper center", fontsize=9, ncol=3, framealpha=0.9)

# crosshair (hover)
cursor_line = ax.axvline(0, color="0.5", lw=1, ls=":")
cursor_line.set_visible(False)
annot = ax.annotate("", xy=(0, 0), xytext=(12, 12),
                    textcoords="offset points", fontsize=9, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white",
                              ec="0.6", alpha=0.92))
annot.set_visible(False)

# ---- component table (redrawn on every update) ----
ax_table = fig.add_axes([0.59, 0.30, 0.19, 0.60])

# ---- text boxes: load, branches and gain ----
fig.text(0.80, 0.925, "Load, branches and gain", fontsize=10,
         fontweight="bold")
ax_L = fig.add_axes([0.90, 0.875, 0.09, 0.034])
ax_R_L = fig.add_axes([0.90, 0.832, 0.09, 0.034])
ax_Re = fig.add_axes([0.90, 0.789, 0.09, 0.034])
ax_G = fig.add_axes([0.90, 0.746, 0.09, 0.034])
tb_L = TextBox(ax_L, "L [H]  ", initial=f"{L_0:.3e}")
tb_R_L = TextBox(ax_R_L, "R_L [Ohm]  ", initial=f"{R_L_0:g}")
tb_Re = TextBox(ax_Re, "Re branches [Ohm]  ",
                initial=", ".join(f"{r:g}" for r in RE_0))
tb_G = TextBox(ax_G, "G [A/V]  ", initial=f"{G_0:g}")
fig.text(0.80, 0.722, "Re: one per branch, comma-separated (Enter)",
         fontsize=8, color="0.35")

# readings panel
info = fig.text(0.80, 0.695, "", va="top", ha="left", fontsize=9,
                family="monospace")

# x-range selector
ax_radio = fig.add_axes([0.59, 0.04, 0.13, 0.14])
ax_radio.set_title("X-axis range", fontsize=10)
radio = RadioButtons(ax_radio, ("Auto (decades)", "+-2e6", "+-1e7"), active=0)

# ---- sliders ----
ax_R3 = fig.add_axes([0.11, 0.31, 0.37, 0.03])
ax_RF = fig.add_axes([0.11, 0.26, 0.37, 0.03])
ax_C1 = fig.add_axes([0.11, 0.21, 0.37, 0.03])
ax_CF = fig.add_axes([0.11, 0.16, 0.37, 0.03])

sR3 = Slider(ax_R3, "R3 [Ohm]", R3_MIN, R3_MAX, valinit=R3_0, valfmt="%.0f",
             color=COL_TRACE)
sRF = Slider(ax_RF, "RF [Ohm]", RF_MIN, RF_MAX, valinit=RF_0, valfmt="%.0f",
             color=COL_TRACE)
sC1 = Slider(ax_C1, "C1 [F]", C1_MIN, C1_MAX, valinit=C1_0, valfmt="%.2e",
             color=COL_TRACE)
sCF = Slider(ax_CF, "CF [F]", CF_MIN, CF_MAX, valinit=CF_0, valfmt="%.2e",
             color=COL_TRACE)
fig.text(0.05, 0.09, "RF and CF coupled: CF*RF = L/R_total  "
         "(R_total = R_L + R4 + Re_eq)", fontsize=9, color="0.35")


# ===== component table =====
def component_rows(m):
    """Rows of the table: (reference, value, what sets it)."""
    rows = [
        ("R3", si(m["R3"], "Ω"), "slider"),
        ("R4", si(R4, "Ω"), "fixed (shunt)"),
        ("R5", si(m["R5"], "Ω"), "from G"),
        ("RF", si(m["RF"], "Ω"), "slider (coupled)"),
        ("CF", si(m["CF"], "F"), "slider (coupled)"),
        ("C1", si(m["C1"], "F"), "slider"),
        ("L", si(state["L"], "H"), "text box (coil)"),
        ("R_L", si(state["R_L"], "Ω"), "text box (coil)"),
    ]
    for i, r in enumerate(state["Re"], start=1):
        rows.append((f"Re_{i}", si(r, "Ω"), f"text box (branch {i})"))
    return rows


def draw_table(m):
    ax_table.clear()
    ax_table.axis("off")
    ax_table.set_title("Component values", fontsize=10, fontweight="bold",
                       loc="left")
    table = ax_table.table(cellText=[list(r) for r in component_rows(m)],
                           colLabels=["Ref", "Value", "Set by"],
                           colWidths=[0.2, 0.3, 0.5],
                           cellLoc="left", loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)


# ===== plot / readings =====
def update(_=None):
    m = compute()
    a, b, c, disc = m["a"], m["b"], m["c"], m["disc"]
    rts = real_roots(a, b, c)
    xmin, xmax, lt = x_window(rts, state["mode"], a, c)

    ax.set_xscale("symlog", linthresh=lt, linscale=0.9)

    x = samples(xmin, xmax, lt)
    y = a * x**2 + b * x + c
    trace.set_data(x, y)

    rx = [r for r in rts if xmin <= r <= xmax]
    roots_plot.set_data(rx, [0] * len(rx))

    if disc < 0 and a > 0 and c > 0:
        wn = np.sqrt(c / a)
        wn_line.set_xdata([-wn, -wn])
        wn_line.set_visible(-wn >= xmin)
    else:
        wn_line.set_visible(False)

    ax.set_xlim(xmin, xmax)
    # y axis limited to the region of the roots (parabola vertex and y=0)
    yv = c - b * b / (4 * a) if a != 0 else float(y.min())
    lo, hi = min(0.0, yv), max(0.0, yv)
    if hi == lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.6
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_title(f"D(s) = {a:.3e} s^2  +  {b:.3e} s  +  {c:.3e}", fontsize=11)

    # ---- readings ----
    if len(rts) == 2:
        f1, f2 = abs(rts[0]) / (2 * np.pi), abs(rts[1]) / (2 * np.pi)
        rtxt = (f"  s1 = {rts[0]:.3e}  ({f1:.3e} Hz)\n"
                f"  s2 = {rts[1]:.3e}  ({f2:.3e} Hz)")
    elif len(rts) == 1:
        rtxt = f"  s (linear) = {rts[0]:.3e}"
    elif a > 0 and c > 0:
        wn = np.sqrt(c / a)
        zeta = b / (2 * np.sqrt(a * c))
        rtxt = (f"  no real roots\n"
                f"  wn   = {wn:.3e} rad/s ({wn / (2 * np.pi):.3e} Hz)\n"
                f"  zeta = {zeta:.3f}")
    else:
        rtxt = "  no real roots"
    diag = ("2 real negative poles\n(stable system)" if disc >= 0
            else "complex conjugate poles\n(underdamped)")

    cancel_error = abs(m["CF"] * m["RF"] / m["tau"] - 1)
    if cancel_error < 1e-3:
        txt_cancel = "  holds"
    else:
        txt_cancel = (f"  does NOT hold (error {100 * cancel_error:.1f} %):\n"
                      "  the polynomial is not valid")

    info.set_text(
        "Gain I/Vin\n"
        "  G = R3/(R4*R5)\n"
        f"    = {m['R3']:.4g}/({R4:.4g}*{m['R5']:.4g}) = {m['G']:.4g} A/V\n"
        "  G = 1 A/V  <=>  R3/R4 = R5\n"
        "Series resistance with L\n"
        "  R_total = R_L + R4 + Re_eq\n"
        f"    = {state['R_L']:.3g} + {R4:.3g} + {m['Re_eq']:.3g}"
        f" = {m['R_total']:.3g} Ohm\n"
        f"  ({len(state['Re'])} branches in parallel)\n"
        "Cancellation CF*RF = L/R_total\n"
        f"  CF*RF    = {m['CF'] * m['RF']:.3e} s\n"
        f"  L/R_tot  = {m['tau']:.3e} s\n"
        f"{txt_cancel}\n"
        f"{'-' * 32}\n"
        "D(s), independent of R5\n"
        f"  K = R3*R_total/R4 = {m['K']:.4g} Ohm\n"
        f"Poles [rad/s]:\n{rtxt}\n"
        f"Disc b^2-4ac = {disc:.3e}\n{diag}"
    )

    draw_table(m)
    fig.canvas.draw_idle()


# ===== CF <-> RF coupling  (CF*RF = L/R_total) =====
def on_cf(_=None):
    if state["lock"]:
        return
    state["lock"] = True
    sRF.set_val(clamp(load_tau() / sCF.val, RF_MIN, RF_MAX))
    state["lock"] = False
    update()


def on_rf(_=None):
    if state["lock"]:
        return
    state["lock"] = True
    sCF.set_val(clamp(load_tau() / sRF.val, CF_MIN, CF_MAX))
    state["lock"] = False
    update()


def reconcile():
    """Keeps CF and recomputes RF after L, R_L or the Re values change."""
    state["lock"] = True
    sRF.set_val(clamp(load_tau() / sCF.val, RF_MIN, RF_MAX))
    state["lock"] = False
    update()


# ===== text boxes =====
def read_positive(text):
    """Converts the text to a number > 0; returns None if it cannot."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def on_L(text):
    value = read_positive(text)
    if value is not None:
        state["L"] = value
        reconcile()


def on_R_L(text):
    value = read_positive(text)
    if value is not None:
        state["R_L"] = value
        reconcile()


def on_Re(text):
    """List 'Re_1, Re_2, ...': one emitter resistor per branch."""
    values = [read_positive(t) for t in text.split(",")]
    if values and None not in values:
        state["Re"] = values
        reconcile()


def on_G(text):
    value = read_positive(text)
    if value is not None:
        state["G"] = value
        update()                # G only changes R5, not the poles


def on_radio(label):
    state["mode"] = label
    update()


def on_move(event):
    if event.inaxes is not ax or event.xdata is None:
        if annot.get_visible():
            annot.set_visible(False)
            cursor_line.set_visible(False)
            fig.canvas.draw_idle()
        return
    m = compute()
    xh = event.xdata
    yh = m["a"] * xh * xh + m["b"] * xh + m["c"]
    cursor_line.set_xdata([xh, xh])
    cursor_line.set_visible(True)
    annot.xy = (xh, yh)
    annot.set_text(f"x={xh:.2e}\ny={yh:.2e}")
    annot.set_visible(True)
    fig.canvas.draw_idle()


# events
sR3.on_changed(update)
sC1.on_changed(update)
sRF.on_changed(on_rf)
sCF.on_changed(on_cf)
tb_L.on_submit(on_L)
tb_R_L.on_submit(on_R_L)
tb_Re.on_submit(on_Re)
tb_G.on_submit(on_G)
radio.on_clicked(on_radio)
fig.canvas.mpl_connect("motion_notify_event", on_move)

# initial render: RF is adjusted so the cancellation holds from the start
reconcile()


if __name__ == "__main__":
    plt.show()
