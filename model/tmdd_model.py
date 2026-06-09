"""
Ocular TMDD Model - Python reference implementation
====================================================
Mirrors the rxode2 model in 01_model_definition.R exactly. Shared by all
Python analyses in this repo.

Imports:
    from model.tmdd_model import (
        rhs, simulate, params_for_format,
        TARGET_BIO, MW_KDA, PHYS_VOLUMES,
    )

See docs/methods.md for the full ODE system and parameter rationale.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ==========================================================================
# Physiological constants
# ==========================================================================
PHYS_VOLUMES = {
    'rabbit': dict(Vv=1.5,  Va=0.30, Vr=0.25, Vp=200),
    'human':  dict(Vv=4.0,  Va=0.25, Vr=0.40, Vp=3000),
}

K_AO_BY_SPECIES = {'rabbit': 0.40, 'human': 0.60}

TARGET_BIO = dict(R0_nM=5.0, k_deg=0.05, k_int=0.10)
TARGET_BIO['k_syn'] = TARGET_BIO['k_deg'] * TARGET_BIO['R0_nM']

MW_KDA = dict(naked=30, fab=80, fc=115)

# Allometric scaling exponent (Betts 2018)
ALLOMETRIC_EXPONENT = 0.85


# ==========================================================================
# Format-specific parameter factory
# ==========================================================================
def params_for_format(format_name, species='rabbit'):
    """Return parameter dict for the given format and species.

    Parameters
    ----------
    format_name : str
        One of 'naked', 'fab', 'fc'
    species : str
        'rabbit' (default) or 'human'

    Returns
    -------
    dict with all rate constants, volumes, target biology, and CL_p.
    """
    V = PHYS_VOLUMES[species]
    bw_scale = (70 / 3) ** ALLOMETRIC_EXPONENT if species == 'human' else 1.0

    base = dict(**V, k_ao=K_AO_BY_SPECIES[species],
                k_int=TARGET_BIO['k_int'],
                k_syn=TARGET_BIO['k_syn'], k_deg=TARGET_BIO['k_deg'])

    if format_name == 'naked':
        return dict(**base, k_va=0.0140, k_av=0.05,
                    k_vr=0.0025, k_rv=0.005,
                    k_on=0.05, k_off=0.50,
                    phi_FcRn=0, f_rec=0,
                    CL_p=18.0 * bw_scale)
    elif format_name == 'fab':
        return dict(**base, k_va=0.0102, k_av=0.03,
                    k_vr=0.0015, k_rv=0.003,
                    k_on=0.05, k_off=0.50,
                    phi_FcRn=0, f_rec=0,
                    CL_p=4.5 * bw_scale)
    elif format_name == 'fc':
        return dict(**base, k_va=0.0090, k_av=0.02,
                    k_vr=0.0013, k_rv=0.0025,
                    k_on=0.10, k_off=0.25,
                    phi_FcRn=1, f_rec=0.50,
                    CL_p=0.57 * bw_scale)
    else:
        raise ValueError(f"Unknown format: {format_name}. "
                         f"Choose from 'naked', 'fab', 'fc'.")


# ==========================================================================
# Right-hand side of the ODE system
# ==========================================================================
def rhs(t, y, p):
    """RHS of the 6-state TMDD model.

    State order: A_Cv, A_Ca, A_Cr, A_R, A_DR, A_Cp (amounts, nmol).
    """
    A_Cv, A_Ca, A_Cr, A_R, A_DR, A_Cp = y
    Vv, Va, Vr, Vp = p['Vv'], p['Va'], p['Vr'], p['Vp']

    # Concentrations (nM)
    Cr = A_Cr / Vr
    R  = A_R  / Vr
    DR = A_DR / Vr

    # Binding fluxes (nM/h)
    bind   = p['k_on']  * Cr * R
    unbind = p['k_off'] * DR
    recycle = p['phi_FcRn'] * p['f_rec'] * p['k_int'] * DR

    return [
        -p['k_va']*A_Cv + p['k_av']*A_Ca - p['k_vr']*A_Cv + p['k_rv']*A_Cr,
         p['k_va']*A_Cv - p['k_av']*A_Ca - p['k_ao']*A_Ca,
         p['k_vr']*A_Cv - p['k_rv']*A_Cr - (bind - unbind)*Vr + recycle*Vr,
        (p['k_syn'] - p['k_deg']*R)*Vr - (bind - unbind)*Vr,
        (bind - unbind)*Vr - p['k_int']*A_DR,
         p['k_ao']*A_Ca - (p['CL_p']/p['Vp'])*A_Cp,
    ]


# ==========================================================================
# Simulation driver
# ==========================================================================
def simulate(params, dose_nmol, t_max_d=60, n_pts=1500, R0_override=None,
             rtol=1e-8, atol=1e-11):
    """Run a forward simulation and return trajectory as a dict.

    Parameters
    ----------
    params : dict
        Parameter dict from params_for_format() or a sample.
    dose_nmol : float
        IVT bolus dose in nmol.
    t_max_d : float
        Simulation duration in days.
    n_pts : int
        Number of output time points.
    R0_override : float or None
        If given, overrides the steady-state target abundance. Useful for
        sensitivity sweeps on R0.

    Returns
    -------
    dict with keys: t_h, t_d, Cv, Cr, R, DR, Cp, TO (all numpy arrays)
    or None if integration failed.
    """
    R0 = R0_override if R0_override is not None else TARGET_BIO['R0_nM']
    Vr = params['Vr']
    y0 = [dose_nmol, 0.0, 0.0, R0 * Vr, 0.0, 0.0]
    t_eval = np.linspace(0, t_max_d * 24, n_pts)

    sol = solve_ivp(rhs, (0, t_max_d * 24), y0, args=(params,),
                    t_eval=t_eval, method='LSODA', rtol=rtol, atol=atol)
    if not sol.success:
        return None

    return dict(
        t_h=sol.t,
        t_d=sol.t / 24,
        Cv=sol.y[0] / params['Vv'],
        Ca=sol.y[1] / params['Va'],
        Cr=sol.y[2] / params['Vr'],
        R= sol.y[3] / params['Vr'],
        DR=sol.y[4] / params['Vr'],
        Cp=sol.y[5] / params['Vp'],
        TO=sol.y[4] / params['Vr'] /
           (sol.y[3]/params['Vr'] + sol.y[4]/params['Vr'] + 1e-30),
    )


# ==========================================================================
# Convenience: standard dose calculation
# ==========================================================================
def dose_nmol_from_mg(dose_mg, format_name):
    """Convert mg dose to nmol given format molecular weight."""
    return dose_mg * 1e6 / (MW_KDA[format_name] * 1000)
