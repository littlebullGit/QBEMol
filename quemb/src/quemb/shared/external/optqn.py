# Author(s): Hong-Zhou Ye
#            Oinam Romesh Meitei
#            Minsik Cho
# NOTICE: The following code is mostly written by Hong-Zhou Ye
#   (except for the trust region routine)
#         The code has been slightly modified.
#
import logging
from collections.abc import Sequence

import numpy as np
from numpy import array, empty, float64, outer, zeros
from numpy.linalg import inv, norm, pinv

from quemb.kbe.pfrag import Frags as pFrags
from quemb.molbe.helper import get_eri, get_scfObj
from quemb.molbe.pfrag import Frags
from quemb.shared.external.cphf_utils import cphf_kernel_batch, get_rhf_dP_from_u
from quemb.shared.external.cpmp2_utils import get_dPmp2_batch_r
from quemb.shared.external.jac_utils import get_dPccsdurlx_batch_u
from quemb.shared.typing import GlobalAOIdx, Matrix, RelAOIdx, SeqOverEdge

logger = logging.getLogger(__name__)


def line_search_LF(func, xold, fold, dx, iter_):
    """Adapted from D.-H. Li and M. Fukushima, Optimization Metheods and Software,
    13, 181 (2000)

    Enhanced with chemical potential step limiting for Bootstrap Embedding stability.
    """
    beta = 0.1
    rho = 0.9
    sigma1 = 1e-3
    sigma2 = 1e-3
    eta = (iter_ + 1) ** -2.0

    # Enhanced stability: limit initial step size for large chemical potential changes
    max_chem_pot_step = 2.0  # Maximum chemical potential change in Hartree
    max_step_magnitude = norm(dx)

    if max_step_magnitude > max_chem_pot_step:
        print(f"  Line search: Large step detected ({max_step_magnitude:.6f} Ha)", flush=True)
        print(f"  Line search: Pre-scaling step to {max_chem_pot_step:.6f} Ha for stability", flush=True)
        initial_scale = max_chem_pot_step / max_step_magnitude
        dx = dx * initial_scale
        print(f"  Line search: Applied pre-scaling factor: {initial_scale:.6f}", flush=True)

    xk = xold + dx
    lcout = 0

    fk = func(xk)
    lcout += 1

    norm_dx = norm(dx)
    norm_fk = norm(fk)
    norm_fold = norm(fold)
    alp = 1.0

    if norm_fk > rho * norm_fold - sigma2 * norm_dx**2.0:
        while norm_fk > (1.0 + eta) * norm_fold - sigma1 * alp**2.0 * norm_dx**2.0:
            alp *= beta
            xk = xold + alp * dx

            fk = func(xk)

            lcout += 1
            norm_fk = norm(fk)

            # Additional safety: stop line search if step becomes too small
            if alp < 1e-8:
                print(f"  Line search: Step size too small (α = {alp:.2e}), stopping", flush=True)
                break
            if lcout == 20:
                break

    print(" No. of line search steps in QN opt :", lcout, flush=True)
    print(f" Final step size (α): {alp:.6f}", flush=True)
    print(flush=True)
    return alp, xk, fk


def trustRegion(func, xold, fold, Binv, c=0.5):
    r"""Perform Trust Region Optimization.

    See "A Broyden Trust Region Quasi-Newton Method
    for Nonlinear Equations" (https://www.iaeng.org/IJCS/issues_v46/issue_3/IJCS_46_3_09.pdf)"
    Algorithm 1 for more information

    Parameters
    ----------
    func : typing.Callable
        Cost function
    xold : list or numpy.ndarray
        Current x_p (potentials in BE optimization)
    fold : list or numpy.ndarray
        Current f(x_p) (error vector)
    Binv : numpy.ndarray
        Inverse of Jacobian approximate (B^{-1}); This is updated in Broyden's Method
        through Sherman-Morrison formula
    c : float, optional
        Initial value of trust radius :math:`\in (0, 1)`, by default 0.5

    Returns
    -------
    xnew, fnew: tuple
        x_{p+1} and f_{p+1}. These values are used to proceed with Broyden's Method.
    """
    # c := initial trust radius (trust_radius = c^p)
    microiter = 0  # p
    rho = 0.001  # Threshold for trust region subproblem
    ratio = 0  # Initial r
    B = inv(Binv)  # approx Jacobian
    # dx_gn = - Binv@fold
    dx_gn = -(Binv @ Binv.T) @ B.T @ fold
    dx_sd = -B.T @ fold  # Steepest Descent step
    t = norm(dx_sd) ** 2 / norm(B @ dx_sd) ** 2
    prevdx = None
    ared = 0.0  # Reduction in the objective function; Initialize value to 0
    while ratio < rho or ared < 0.0:
        # Trust Region subproblem
        # minimize (1/2) ||F_k + B_k d||^2 w.r.t. d, s.t. d w/i trust radius
        # to pick the optimal direction using dog leg method
        if norm(dx_gn) < max(1.0, norm(xold)) * (
            c**microiter
        ):  # Gauss-Newton step within the trust radius
            print(
                "  Trust Region Optimization Step ",
                microiter,
                ": Gauss-Newton",
                flush=True,
            )
            dx = dx_gn
        elif t * norm(dx_sd) > max(1.0, norm(xold)) * (
            c**microiter
        ):  # GN step outside, SD step also outside
            print(
                "  Trust Region Optimization Step ",
                microiter,
                ": Steepest Descent",
                flush=True,
            )
            dx = (c**microiter) / norm(dx_sd) * dx_sd
        else:  # GN step outside, SD step inside (dog leg step)
            # dx := t*dx_sd + s (dx_gn - t*dx_sd) s.t. ||dx|| = c^p
            print(
                "  Trust Region Optimization Step ", microiter, ": Dog Leg", flush=True
            )
            tdx_sd = t * dx_sd
            diff = dx_gn - tdx_sd
            # s = (-dx_sd.T@diff + sqrt((dx_sd.T@diff)**2 -
            #   norm(diff)**2*(norm(dx_sd)**2-(c ** microiter)**2)))
            #   / (norm(dx_sd))**2
            # s is largest value in [0, 1] s.t. ||dx|| \le trust radius
            s = 1
            dx = tdx_sd + s * diff
            while norm(dx) > c**microiter and s > 0:
                s -= 0.001
                dx = tdx_sd + s * diff
        if prevdx is None or not all(dx == prevdx):
            # Actual Reduction := f(x_k) - f(x_k + dx)
            fnew = func(xold + dx)
            ared = 0.5 * (norm(fold) ** 2 - norm(fnew) ** 2)
            # Predicted Reduction := q(0) - q(dx) where q = (1/2) ||F_k + B_k d||^2
            pred = 0.5 * (norm(fold) ** 2 - norm(fold + B @ dx) ** 2)
        # Trust Region convergence criteria
        # r = ared/pred \le rho
        ratio = ared / pred
        microiter += 1
        if prevdx is None or not all(dx == prevdx):
            logger.debug(f"    ||δx||: {norm(dx)}")
            logger.debug(
                f"    Reduction Ratio (Actual / Predicted): {ared} / {pred} = {ratio}"
            )
        prevdx = dx
    return xold + dx, fnew  # xnew


class OptimizerStallError(Exception):
    """Exception raised when an optimizer detects a stall condition.

    This is a generic exception that can be raised by any optimizer
    (Broyden, trust region, SciPy, etc.) when it detects that progress
    has stalled and further iterations are unlikely to help.

    Attributes
    ----------
    message : str
        Human-readable description of the stall condition
    iter_count : int
        Iteration number when stall was detected
    consecutive_stalls : int
        Number of consecutive iterations without progress
    last_error : float or None
        Last density matching error value
    optimizer_name : str
        Name of the optimizer that detected the stall
    """

    def __init__(self, message, iter_count, consecutive_stalls, last_error, optimizer_name="Broyden"):
        self.message = message
        self.iter_count = iter_count
        self.consecutive_stalls = consecutive_stalls
        self.last_error = last_error
        self.optimizer_name = optimizer_name
        super().__init__(self.message)


# Backwards compatibility alias
BroydenStallError = OptimizerStallError


class FrankQN:
    """Quasi Newton Optimization

    Performs quasi newton optimization. Interfaces many functionalities of the
    frankestein code originally written by Hong-Zhou Ye
    """

    def __init__(self, func, x0, f0, J0, trust=0.5, max_space=500, stall_threshold=3):
        self.x0 = x0
        self.n = x0.size
        self.f0 = f0
        self.func = func

        self.B0 = pinv(J0)

        self.tol_gmres = 1.0e-6
        self.xnew = None  # new errvec
        self.xold = None  # old errvec
        self.fnew = None  # new jacobian?
        self.fold = None  # old jacobian?
        self.max_subspace = max_space
        self.dxs = empty([self.max_subspace, self.n])
        self.fs = empty([self.max_subspace, self.n])
        self.us = empty([self.max_subspace, self.n])  # u_m = B_m @ f_m
        self.vs = empty([self.max_subspace, self.n])  # v_m = B_0 @ f_{m+1}
        self.B = None
        self.trust = trust

        # Stall detection
        self.stall_threshold = stall_threshold  # Number of consecutive stalls before raising error
        self.consecutive_stalls = 0
        self.is_stalled = False
        self.last_step_size = None  # Track line search alpha
        self.error_history = []  # Track density matching error progression

    def next_step(self, iter, trust_region=False):
        import numpy as np

        # Check if already stalled
        if self.is_stalled:
            raise OptimizerStallError(
                f"Optimizer stalled after {self.consecutive_stalls} consecutive failed updates. "
                f"Last density error: {self.error_history[-1] if self.error_history else 'N/A'}. "
                f"This typically indicates VQE RDMs are not responding to chemical potential changes.",
                iter_count=iter,
                consecutive_stalls=self.consecutive_stalls,
                last_error=self.error_history[-1] if self.error_history else None,
                optimizer_name="Broyden"
            )

        if iter == 0:
            self.xnew = self.x0
            self.fnew = self.func(self.xnew) if self.f0 is None else self.f0
            self.fs[0] = self.fnew.copy()
            self.us[0] = self.B0 @ self.fnew
            self.Binv = self.B0.copy()
            self.error_history.append(float(norm(self.fnew)))

        # Book keeping
        if not iter == 0:
            dx_i = self.xnew - self.xold
            df_i = self.fnew - self.fold

        self.xold = self.xnew.copy()
        self.fold = self.fnew.copy()

        # Flag to track if Broyden update was skipped
        skip_update = False
        stall_this_iter = False

        if not iter == 0:
            # Compute denominator for Broyden update
            denominator = dx_i @ self.Binv @ df_i

            # DEBUG: Check for numerical instability
            print(f"\n{'='*80}")
            print(f"DEBUG optqn.py:202 - Broyden Update (iter {iter})")
            print(f"{'='*80}")
            print(f"||dx_i|| = {np.linalg.norm(dx_i):.6e}")
            print(f"||df_i|| = {np.linalg.norm(df_i):.6e}")
            print(f"denominator (dx_i @ Binv @ df_i) = {denominator:.6e}")
            print(f"Max abs value in dx_i: {np.max(np.abs(dx_i)):.6e}")
            print(f"Max abs value in df_i: {np.max(np.abs(df_i)):.6e}")

            # Enhanced safeguards for Broyden stability
            min_denominator = 1e-12
            max_allowed_step = 2.0  # Limit chemical potential changes to 2 Ha
            max_allowed_update = 1e2  # Much more conservative: 100 instead of 1000
            max_allowed_binv = 10.0   # Limit maximum values in Binv matrix

            if abs(denominator) < min_denominator:
                print(f"⚠️  WARNING: Denominator too small ({denominator:.6e} < {min_denominator:.6e})")
                print(f"⚠️  Skipping Broyden update to prevent numerical explosion")
                print(f"{'='*80}\n")
                skip_update = True
                stall_this_iter = True
            else:
                # Standard Broyden update
                tmp__ = outer(dx_i - self.Binv @ df_i, dx_i @ self.Binv) / denominator
                max_update = np.max(np.abs(tmp__))
                print(f"Max abs value in Broyden update: {max_update:.6e}")

                # Scale down large updates more aggressively
                if max_update > max_allowed_update:
                    print(f"⚠️  WARNING: Broyden update too large ({max_update:.6e} > {max_allowed_update:.6e})")
                    scale_factor = max_allowed_update / max_update
                    tmp__ *= scale_factor
                    print(f"Applied scale factor: {scale_factor:.6e}")

                self.Binv += tmp__

                # Additional safeguard: clamp Binv values to prevent matrix corruption
                max_binv_before = np.max(np.abs(self.Binv))
                if max_binv_before > max_allowed_binv:
                    print(f"⚠️  WARNING: Binv matrix too large ({max_binv_before:.6e} > {max_allowed_binv:.6e})")
                    print(f"⚠️  Clamping Binv values to prevent matrix corruption")
                    self.Binv = np.clip(self.Binv, -max_allowed_binv, max_allowed_binv)

                print(f"Max abs value in Binv after update: {np.max(np.abs(self.Binv)):.6e}")

                # Additional check: limit the step size to prevent large chemical potential jumps
                step_proposal = -self.Binv @ self.fold
                max_step = np.max(np.abs(step_proposal))
                if max_step > max_allowed_step:
                    print(f"⚠️  WARNING: Proposed step too large ({max_step:.6f} Ha > {max_allowed_step:.6f} Ha)")
                    print(f"⚠️  Chemical potential changes should be modest for stability")
                    # Don't apply additional damping here - let line search handle it

                print(f"{'='*80}\n")

        # If Broyden update was skipped, also skip line search
        # (optimization has stalled, keep current potentials)
        if skip_update:
            # Keep xnew = xold, fnew = fold (no change)
            self.xnew = self.xold.copy()
            self.fnew = self.fold.copy()
            self.last_step_size = 0.0
        elif trust_region:
            self.xnew, self.fnew = trustRegion(
                self.func, self.xold, self.fold, self.Binv, c=self.trust
            )
        else:
            self.us[iter] = self.get_Bnfn(iter)

            alpha, self.xnew, self.fnew = line_search_LF(
                self.func, self.xold, self.fold, -self.us[iter], iter
            )
            self.last_step_size = alpha

            # Detect line search stall (alpha = 0 means step was rejected)
            if alpha < 1e-10:
                stall_this_iter = True
                print(f"⚠️  WARNING: Line search returned α ≈ 0, step rejected")

            # udpate vs, dxs, and fs
            self.vs[iter] = self.B0 @ self.fnew
        self.dxs[iter] = self.xnew - self.xold

        # Track error history
        current_error = float(norm(self.fnew))
        self.error_history.append(current_error)

        # Update stall detection
        if stall_this_iter:
            self.consecutive_stalls += 1
            print(f"⚠️  STALL DETECTED: {self.consecutive_stalls}/{self.stall_threshold} consecutive stalls")

            if self.consecutive_stalls >= self.stall_threshold:
                self.is_stalled = True
                print(f"\n{'!'*80}")
                print(f"!!! OPTIMIZER STALLED !!!")
                print(f"{'!'*80}")
                print(f"The Broyden optimizer has stalled after {self.consecutive_stalls} consecutive failed updates.")
                print(f"This typically happens when VQE RDMs don't respond predictably to chemical potential changes.")
                print(f"")
                print(f"Error history: {[f'{e:.4e}' for e in self.error_history[-5:]]}")
                print(f"Last step size (α): {self.last_step_size}")
                print(f"")
                print(f"Possible solutions:")
                print(f"  1. Use FCI instead of VQE for this system")
                print(f"  2. Modify optimizer to be more tolerant of noisy gradients")
                print(f"  3. Use a hybrid FCI→VQE approach")
                print(f"{'!'*80}\n")

                raise OptimizerStallError(
                    f"Optimizer stalled after {self.consecutive_stalls} consecutive failed updates",
                    iter_count=iter,
                    consecutive_stalls=self.consecutive_stalls,
                    last_error=current_error,
                    optimizer_name="Broyden"
                )
        else:
            # Reset stall counter on successful step
            self.consecutive_stalls = 0

        if iter + 1 < self.max_subspace:
            self.fs[iter + 1] = self.fnew.copy()
        else:
            print("Reached the maximum number of iterations:", self.max_subspace)

    def get_Bnfn(self, n):
        # self.us; self.dxs; self.vs
        if n == 0:
            return self.us[0]

        vs = [None] * n
        for i in range(n):
            vs[i] = self.vs[n - i - 1]
        for i in range(1, n + 1):
            un_ = self.us[i - 1]
            dxn_ = self.dxs[i - 1]
            vps = [None] * (n - i + 1)
            for j in range(n - i + 1):
                a = vs[j]
                b = vs[n - i] - un_

                vps[j] = a + (dxn_ @ a) / (dxn_ @ b) * (dxn_ - b)

            vs = vps

        return vs[0]


def get_be_error_jacobian(n_frag, Fobjs, jac_solver="HF"):
    Jes = [None] * n_frag
    Jcs = [None] * n_frag
    xes = [None] * n_frag
    xcs = [None] * n_frag
    ys = [None] * n_frag
    alphas = [None] * n_frag

    if jac_solver.upper() == "MP2":
        res_func = mp2res_func
    elif jac_solver.upper() == "CCSD":
        res_func = ccsdres_func
    elif jac_solver.upper() == "HF":
        res_func = hfres_func
    else:
        raise NotImplementedError("Jacobian solver input not implemented")

    Ncout = [None] * n_frag
    for A in range(n_frag):
        Jes[A], Jcs[A], xes[A], xcs[A], ys[A], alphas[A], Ncout[A] = (
            get_atbe_Jblock_frag(Fobjs[A], res_func)
        )

    alpha = sum(alphas)

    # build Jacobian
    """ ignore!
    F0-M1 F1-M2M2 F2-M3M3 F3-M4
       M1   M2  M2 M3  M3 M4
    M1 E0   C1-1
    M2 C0-0 E1  E1
    M2      E1  E1 C2-2
    M3      C1-1   E2  E2
    M3             E2  E2 C3-3
    M4             C2-2   E3
    """
    N_ = sum(Ncout)
    J = zeros((N_ + 1, N_ + 1))
    cout = 0

    for findx, fobj in enumerate(Fobjs):
        J[cout : Ncout[findx] + cout, cout : Ncout[findx] + cout] = Jes[findx]
        J[cout : Ncout[findx] + cout, N_:] = array(xes[findx]).reshape(-1, 1)
        J[N_:, cout : Ncout[findx] + cout] = ys[findx]

        coutc = 0
        coutc_ = 0
        for cindx, cens in enumerate(fobj.relAO_in_ref_per_edge):
            coutc += Jcs[fobj.ref_frag_idx_per_edge[cindx]].shape[0]
            start_ = sum(Ncout[: fobj.ref_frag_idx_per_edge[cindx]])
            end_ = start_ + Ncout[fobj.ref_frag_idx_per_edge[cindx]]
            J[cout + coutc_ : cout + coutc, start_:end_] += Jcs[
                fobj.ref_frag_idx_per_edge[cindx]
            ]
            J[cout + coutc_ : cout + coutc, N_:] += array(
                xcs[fobj.ref_frag_idx_per_edge[cindx]]
            ).reshape(-1, 1)
            coutc_ = coutc
        cout += Ncout[findx]
    J[N_:, N_:] = alpha

    return J


def get_atbe_Jblock_frag(
    fobj: Frags | pFrags, res_func
) -> tuple[Matrix[float64], Matrix[float64], list, list, list, float, int]:
    assert (
        fobj._mo_coeffs is not None
        and fobj.nsocc is not None
        and fobj.fock is not None
        and fobj.heff is not None
        and fobj.nao is not None
    )
    vpots = get_vpots_frag(fobj.nao, fobj.relAO_per_edge, fobj.AO_in_frag)
    eri_ = get_eri(fobj.dname, fobj.nao, eri_file=fobj.eri_file)
    dm0 = 2.0 * (fobj._mo_coeffs[:, : fobj.nsocc] @ fobj._mo_coeffs[:, : fobj.nsocc].T)
    mf_ = get_scfObj(fobj.fock + fobj.heff, eri_, fobj.nsocc, dm0=dm0)

    dPs, dP_mu = res_func(mf_, vpots, eri_, fobj.nsocc)

    Je = []
    Jc = []
    y = []
    xe = []
    xc = []
    cout = 0

    for edge in fobj.relAO_per_edge:
        for j_ in range(len(edge)):
            for k_ in range(len(edge)):
                if j_ > k_:
                    continue
                # response w.r.t matching pot
                # edges
                tmpje_ = []

                for edge_ in fobj.relAO_per_edge:
                    lene = len(edge_)

                    for j__ in range(lene):
                        for k__ in range(lene):
                            if j__ > k__:
                                continue

                            tmpje_.append(dPs[cout][edge_[j__], edge_[k__]])
                y_ = 0.0
                for fidx, fval in enumerate(fobj.AO_in_frag):
                    if not any(fidx in sublist for sublist in fobj.relAO_per_edge):
                        y_ += dPs[cout][fidx, fidx]

                y.append(y_)

                tmpjc_ = []
                # center on the same fragment
                # for cen in fobj.efac[1]:
                for j_relAO in fobj.relAO_per_origin:
                    for k_relAO in fobj.relAO_per_origin:
                        if j_relAO > k_relAO:
                            continue
                        tmpjc_.append(-dPs[cout][j_relAO, k_relAO])

                Je.append(tmpje_)

                Jc.append(tmpjc_)

                # response w.r.t. chem pot
                # edge
                xe.append(dP_mu[edge[j_], edge[k_]])
                cout += 1

    alpha = 0.0
    for fidx, _ in enumerate(fobj.AO_in_frag):
        if not any(fidx in sublist for sublist in fobj.relAO_per_edge):
            alpha += dP_mu[fidx, fidx]

    for j_relAO in fobj.relAO_per_origin:
        for k_relAO in fobj.relAO_per_origin:
            if j_relAO > k_relAO:
                continue
            xc.append(-dP_mu[j_relAO, k_relAO])

    return array(Je).T, array(Jc).T, xe, xc, y, alpha, cout


def get_be_error_jacobian_selffrag(self, jac_solver="HF"):
    Jes = [None] * self.Nfrag
    xes = [None] * self.Nfrag
    xcs = [None] * self.Nfrag
    ys = [None] * self.Nfrag
    alphas = [None] * self.Nfrag

    if jac_solver == "MP2":
        res_func = mp2res_func
    elif jac_solver == "CCSD":
        res_func = ccsdres_func
    elif jac_solver == "HF":
        res_func = hfres_func
    else:
        raise NotImplementedError("Jacobian solver option not implemented.")

    Jes, _, xes, xcs, ys, alphas, Ncout = get_atbe_Jblock_frag(self.Fobjs[0], res_func)

    N_ = Ncout
    J = zeros((N_ + 1, N_ + 1))

    J[:Ncout, :Ncout] = Jes
    J[:Ncout, N_:] = array(xes).reshape(-1, 1)
    J[N_:, :Ncout] = ys
    J[:Ncout, N_:] += array([*xcs, *xcs]).reshape(-1, 1)
    J[N_:, N_:] = alphas

    return J


def hfres_func(mf, vpots, eri, nsocc) -> tuple[list[Matrix[float64]], Matrix[float64]]:
    C = mf.mo_coeff
    moe = mf.mo_energy
    eri = mf._eri
    no = nsocc

    us = cphf_kernel_batch(C, moe, eri, no, vpots)
    dPs = [get_rhf_dP_from_u(C, no, us[I]) for I in range(len(vpots) - 1)]
    dP_mu = get_rhf_dP_from_u(C, no, us[-1])

    return dPs, dP_mu


def mp2res_func(mf, vpots, eri, nsocc):
    C = mf.mo_coeff
    moe = mf.mo_energy
    eri = mf._eri
    no = nsocc

    dPs_an = get_dPmp2_batch_r(C, moe, eri, no, vpots, aorep=True)
    dPs_an = array([dp_ * 0.5 for dp_ in dPs_an])
    dP_mu = dPs_an[-1]

    return dPs_an[:-1], dP_mu


def ccsdres_func(mf, vpots, eri, nsocc):
    C = mf.mo_coeff
    moe = mf.mo_energy
    eri = mf._eri
    no = nsocc

    dPs_an = get_dPccsdurlx_batch_u(C, moe, eri, no, vpots)

    dP_mu = dPs_an[-1]

    return dPs_an[:-1], dP_mu


def get_vpots_frag(
    nao: int,
    rel_AO_per_edge: SeqOverEdge[Sequence[RelAOIdx]],
    AO_in_frag: Sequence[GlobalAOIdx],
) -> list[Matrix[float64]]:
    vpots: list[Matrix[float64]] = []

    for edge_ in rel_AO_per_edge:
        lene = len(edge_)
        for j__ in range(lene):
            for k__ in range(lene):
                if j__ > k__:
                    continue

                tmppot = zeros((nao, nao))
                tmppot[edge_[j__], edge_[k__]] = tmppot[edge_[k__], edge_[j__]] = 1
                vpots.append(tmppot)

    # only the centers
    # outer edges not included
    tmppot = zeros((nao, nao))
    for fidx, fval in enumerate(AO_in_frag):
        if not any(fidx in sublist for sublist in rel_AO_per_edge):
            tmppot[fidx, fidx] = -1

    vpots.append(tmppot)
    return vpots
