"""
133Cs MOT Simulation GUI
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="¹³³Cs MOT Simulator",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️  ¹³³Cs MOT Force-Profile Simulator")
st.caption("Rate-equation model · D2 line (6S₁/₂ → 6P₃/₂)")

# ── sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Laser Parameters")

    cooler_ratio = st.slider(
        "Cooler saturation  (s_cool / s)",
        min_value=0.01, max_value=20.0,
        value=1.0, step=0.01,
        help="Relative saturation of the F=4→5 cooling beams",
    )

    repumper_ratio = st.slider(
        "Repumper saturation  (s_rep / s)",
        min_value=0.001, max_value=10.0,
        value=0.01, step=0.001,
        format="%.3f",
        help="Relative saturation of the F=3→4 repump beams",
    )

    s = st.slider(
        "Total saturation parameter  s",
        min_value=0.1, max_value=10.0,
        value=1.0, step=0.1,
        help="Peak saturation parameter (both beam sets scaled by their ratio)",
    )

    det = st.slider(
        "Cooler detuning  δ / Γ",
        min_value=-5.0, max_value=-0.1,
        value=-2.0, step=0.1,
        help="Cooling-beam detuning in units of the natural linewidth",
    )

    st.header("🧲 Magnetic Field")
    alpha = st.slider(
        "Gradient  α  (dimensionless)",
        min_value=0.1, max_value=5.0,
        value=1.0, step=0.1,
        help="Quadrupole field gradient in pylcp units",
    )

    st.header("🗺️ Grid")
    nx = st.slider("Position points", 21, 81, 41, step=10)
    nv = st.slider("Velocity points", 21, 81, 41, step=10)
    x_max = st.slider("Position range  |x|_max", 2.0, 10.0, 5.0, step=0.5)
    v_max = st.slider("Velocity range  |v|_max", 2.0, 10.0, 5.0, step=0.5)

    run_btn = st.button("▶  Run simulation", use_container_width=True, type="primary")

# ── helper: run simulation ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_simulation(cooler_ratio, repumper_ratio, s, det, alpha, nx, nv, x_max, v_max):
    import pylcp

    # grid
    x = np.linspace(-x_max, x_max, nx)
    v = np.linspace(-v_max, v_max, nv)
    dx = x[1] - x[0]
    dv = v[1] - v[0]
    X, V = np.meshgrid(x, v)

    # atom & hamiltonians
    atom = pylcp.atom("133Cs")

    H_g_D2, mu_q_g_D2 = pylcp.hamiltonians.hyperfine_coupled(
        atom.state[0].J, atom.I, atom.state[0].gJ, atom.gI,
        atom.state[0].Ahfs / atom.state[2].gammaHz,
        Bhfs=0, Chfs=0, muB=1,
    )
    H_e_D2, mu_q_e_D2 = pylcp.hamiltonians.hyperfine_coupled(
        atom.state[2].J, atom.I, atom.state[2].gJ, atom.gI,
        Ahfs=atom.state[2].Ahfs / atom.state[2].gammaHz,
        Bhfs=atom.state[2].Bhfs / atom.state[2].gammaHz,
        Chfs=0, muB=1,
    )
    dijq_D2 = pylcp.hamiltonians.dqij_two_hyperfine_manifolds(
        atom.state[0].J, atom.state[2].J, atom.I
    )

    E_g_D2 = np.unique(np.diagonal(H_g_D2))   # [F=3, F=4]
    E_e_D2 = np.unique(np.diagonal(H_e_D2))   # [F=2, F=3, F=4, F=5]

    hamiltonian_D2 = pylcp.hamiltonian(
        H_g_D2, H_e_D2, mu_q_g_D2, mu_q_e_D2, dijq_D2
    )

    # magnetic field
    magField = pylcp.quadrupoleMagneticField(alpha)

    # laser beams – full hyperfine D2
    laserBeams_cooling = pylcp.conventional3DMOTBeams(
        s=cooler_ratio * s,
        delta=(E_e_D2[-1] - E_g_D2[-1]) + det,   # F=4→5
    )
    laserBeams_repump = pylcp.conventional3DMOTBeams(
        s=repumper_ratio * s,
        delta=(E_e_D2[-2] - E_g_D2[-2]),           # F=3→4
    )
    laserBeams_D2 = laserBeams_cooling + laserBeams_repump

    trap_D2 = pylcp.rateeq(
        laserBeams_D2, magField, hamiltonian_D2, include_mag_forces=False
    )
    trap_D2.generate_force_profile(
        [np.zeros(X.shape), np.zeros(X.shape), X],
        [np.zeros(V.shape), np.zeros(V.shape), V],
        name="Fz",
    )
    FzCs_D2 = trap_D2.profile["Fz"].F[2]

    # bare F=4→5 (reference)
    H_g_45, mu_q_g_45 = pylcp.hamiltonians.singleF(F=4, gF=1/4, muB=1)
    H_e_45, mu_q_e_45 = pylcp.hamiltonians.singleF(F=5, gF=2/5, muB=1)
    dijq_45 = pylcp.hamiltonians.dqij_two_bare_hyperfine(4, 5)
    hamiltonian_45 = pylcp.hamiltonian(
        H_g_45, H_e_45, mu_q_g_45, mu_q_e_45, dijq_45
    )
    laserBeams_45 = pylcp.conventional3DMOTBeams(s=s, delta=det)
    trap_45 = pylcp.rateeq(
        laserBeams_45, magField, hamiltonian_45, include_mag_forces=False
    )
    trap_45.generate_force_profile(
        [np.zeros(X.shape), np.zeros(X.shape), X],
        [np.zeros(V.shape), np.zeros(V.shape), V],
        name="Fz",
    )
    Fz4to5 = trap_45.profile["Fz"].F[2]

    return x, v, dx, dv, X, V, FzCs_D2, Fz4to5


def make_figure(x, v, dx, dv, X, V, FzCs_D2, Fz4to5):
    fig = plt.figure(figsize=(9, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.12)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    extent = (
        np.amin(x) - dx/2, np.amax(x) + dx/2,
        np.amin(v) - dv/2, np.amax(v) + dv/2,
    )
    kw_im = dict(origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")

    im0 = ax00.imshow(FzCs_D2, **kw_im)
    im1 = ax01.imshow(Fz4to5, **kw_im)
    fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)

    mid_v = int(X.shape[0] / 2)
    mid_x = int(X.shape[1] / 2) + 1

    ax10.plot(X[mid_v, :], FzCs_D2[mid_v, :],        label="¹³³Cs D2")
    ax10.plot(X[mid_v, :], Fz4to5[mid_v, :],  "--",  label="F=4→5", lw=0.9)
    ax10.axhline(0, color="k", lw=0.5, ls=":")
    ax10.legend(fontsize=8)

    ax11.plot(V[:, mid_x], FzCs_D2[:, mid_x],        label="¹³³Cs D2")
    ax11.plot(V[:, mid_x], Fz4to5[:, mid_x],  "--",  label="F=4→5", lw=0.9)
    ax11.axhline(0, color="k", lw=0.5, ls=":")
    ax11.legend(fontsize=8)

    # titles
    ax00.set_title("¹³³Cs D2  (full hyperfine)", fontsize=10)
    ax01.set_title("Bare F=4→F'=5", fontsize=10)

    # axis labels
    ax00.set_ylabel(r"$v\,/\,(\Gamma/k)$")
    ax10.set_ylabel(r"$F_z\,/\,(\hbar k\Gamma)$")
    ax10.set_xlabel(r"$x\,/\,(\mu_B B'/\hbar\Gamma)$")
    ax11.set_xlabel(r"$v\,/\,(\Gamma/k)$")
    ax10.set_title("Force vs position  (v = 0)", fontsize=9)
    ax11.set_title("Force vs velocity  (x = 0)", fontsize=9)

    for ax in (ax00, ax01):
        ax.set_xlabel(r"$x\,/\,(\mu_B B'/\hbar\Gamma)$")
        ax.set_ylabel(r"$v\,/\,(\Gamma/k)$")

    plt.tight_layout()
    return fig


# ── main panel ─────────────────────────────────────────────────────────────────
params = dict(
    cooler_ratio=cooler_ratio,
    repumper_ratio=repumper_ratio,
    s=s, det=det, alpha=alpha,
    nx=nx, nv=nv,
    x_max=x_max, v_max=v_max,
)

# Summary row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cooler ratio",   f"{cooler_ratio:.2f}")
c2.metric("Repumper ratio", f"{repumper_ratio:.3f}")
c3.metric("Saturation s",   f"{s:.1f}")
c4.metric("Detuning δ/Γ",  f"{det:.1f}")
c5.metric("B-field grad α", f"{alpha:.1f}")

st.divider()

if run_btn or "last_result" in st.session_state:
    if run_btn:
        with st.spinner("Running rate-equation solver…"):
            result = run_simulation(**params)
        st.session_state["last_result"] = result
        st.session_state["last_params"] = params
    else:
        result = st.session_state["last_result"]

    x, v, dx, dv, X, V, FzCs_D2, Fz4to5 = result

    col_plot, col_info = st.columns([3, 1])
    with col_plot:
        fig = make_figure(x, v, dx, dv, X, V, FzCs_D2, Fz4to5)
        st.pyplot(fig, use_container_width=True)

    with col_info:
        st.subheader("📊 Quick stats")
        st.write("**¹³³Cs D2  (full)**")
        st.metric("Max force",  f"{FzCs_D2.max():.4f}")
        st.metric("Min force",  f"{FzCs_D2.min():.4f}")

        st.write("**Bare F=4→5**")
        st.metric("Max force",  f"{Fz4to5.max():.4f}")
        st.metric("Min force",  f"{Fz4to5.min():.4f}")

        # capture velocity (zero-crossing slope near x=0)
        mid_v = int(X.shape[0] / 2)
        mid_x = int(X.shape[1] / 2) + 1
        Fz_v = FzCs_D2[:, mid_x]
        v_arr = V[:, mid_x]
        dFdv = np.gradient(Fz_v, v_arr)
        # damping coefficient near v=0
        idx0 = np.argmin(np.abs(v_arr))
        st.write("**Damping  β  (at x=v=0)**")
        st.metric("D2 β", f"{dFdv[idx0]:.4f}")

else:
    st.info(
        "Adjust the sliders in the sidebar, then press **▶ Run simulation**.",
        icon="ℹ️",
    )

st.divider()
st.caption(
    "Parameters are cached — only changed combinations re-run the solver.  "
    "Grid coarseness trades speed vs. accuracy."
)
