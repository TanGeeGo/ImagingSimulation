import os
import time
import json
import torch
import inspect
import numpy as np
from functools import partial
from datetime import datetime
from .analysis import Analysis

class Optimize(Analysis):
    """
    Optimize the optical parameters 

    Initial inputs:
        merit_dict:
            format -> {"merit_func_01": weight, ...}
    """
    def __init__(self, system, views, wavelengths, merit_dict, wz_grads,
                 dtype=torch.float64, device=torch.device('cpu')):
        super().__init__(system, views, wavelengths, dtype, device)

        # set differentiable optical parameters
        if os.path.splitext(self.system.config)[-1] == '.txt':
            raise TypeError('TXT file is not support for optimization!')
        elif os.path.splitext(self.system.config)[-1] == '.json':
            self.load_variable_from_json(self.system.config, wz_grads)

        # set merit function 
        # first get the merit function recorded in the merit directory
        # than fetch the evaluation from Analysis cls
        self.merit_funclist = []
        self.merit_functarget = []
        self.merit_funcweight = []
        for merit_item in merit_dict:
            self.merit_funcweight.append(merit_dict[merit_item].pop('weight')) # pop the weight
            self.merit_functarget.append(
                torch.tensor(merit_dict[merit_item].pop('target'), 
                             dtype=self.dtype, device=self.device)) # pop target of this merit
            merit_func = getattr(super(), str(merit_item)) # method in Analysis
            mfunc_args = list(inspect.signature(merit_func).parameters.keys())
            # partial the pre-args
            for name in mfunc_args:
                # ATTENTION: assign the args by sequence
                merit_func = partial(merit_func, merit_dict[merit_item][str(name)])
            self.merit_funclist.append(merit_func)
            
        assert len(self.merit_funclist) == len(self.merit_funcweight)
        self.INFVAL = 1E14

    def load_variable_from_json(self, config, wz_grad=True):
        """ Load the variables, the bounds of each variable
        
        Parameters
        ----------
        config : str
            Path to the config file of system
        wz_grad : bool
            If set `requires_grad` True of variables for directly back propagation

        Returns or Setattrs
        ----------
        Optimize.variable_names : list
            list to set down the name of variables in system, formulated as ['1_c', ..., 'n_i']. 
            Here `n` is the index of surface, `i` is the attributes of surface.
        Optimize.variables : list
            list to set down the value of variables in system, arranged as the sequence of `variable_names`.
        Optimize.bounds: list
            list to set down the bounds of variables in system, formulated as [[lb_1, ub_1], ..., [lb_i, ub_i]]. 
            Here `lb_i` is the lower bounds of the `ith variable`, and `ub_i` is the upper bounds.
        """
        self.variable_names = []
        self.variables = []
        self.bounds = []
        with open(config) as file:
            vary_dict = json.load(file)

        for item in vary_dict:
            # set variable of the surface
            if item == 'Description':
                continue
            else:
                if vary_dict[item]['variable'] is None:
                    continue
                else:
                    # check the length of bounds
                    if len(vary_dict[item]['bounds']) == 1 or \
                        len(vary_dict[item]['bounds'] < vary_dict[item]['variable']):
                        # number of bounds less than the variables, broadcast the bounds
                        vary_dict[item]['bounds'] = [bds for bds in vary_dict[item]['bounds'] \
                                                     for i in range(len(vary_dict[item]['variable']))]
                        
                    for v_idx, v in enumerate(vary_dict[item]['variable']):
                        self.variable_names.append(str(vary_dict[item]['index']) + '_' + v)
                        # set require grad if needs back propagation
                        try:
                            exec('self.system[{index}].{variable}.requires_grad = {requires_g}'.
                                 format(index=vary_dict[item]['index'], variable=v, requires_g=wz_grad))
                        except:
                            exec('self.system[{index}].{variable}.requires_grad = {requires_g}'.
                                 format(index=vary_dict[item]['index'], variable=v, requires_g=wz_grad))
                            
                        exec('self.variables.append(self.system[{index}].{variable})'.
                             format(index=vary_dict[item]['index'], variable=v))
                        self.bounds.append(vary_dict[item]['bounds'][v_idx])

    def update_geometry(self, variables_update, if_save=False):
        """ Update the geometry of system with the updated variables
        """
        # turn the updated variables tensor into lists
        variables_update_list = list(variables_update.clone())
        assert len(variables_update_list) == len(self.variable_names), \
            ValueError('Incorrect length of updated variables {} and name {}'.
                       format(len(variables_update_list), len(self.variable_names)))

        for var_idx, variable_item in enumerate(self.variable_names):
            s_i, var = variable_item.split('_') # surface index and variable name
            # set the attr of the system
            setattr(self.system[int(s_i)], var, variables_update_list[var_idx])

        if if_save:
            self.system.save_to_json('./lens_file/tmp.json') # TODO: the path of the saved json

        # check the validation of updated system
        assert self.system.geometric_val(), ValueError('The updated system is not valid in geometric!')
        # update the radius of sensor plane according to the new system configuration
        # self.system.update_image(self.views[-1], self.wavelengths)
        # update the radius of all elements according to the new system configuration
        self.update_radius() # Note here the stop radius is fixed!

    ##################################################################################################
    # The methods below are for optimizing the lens system with trust region reflective method
    # 
    # Reference
    # ----------
    # .. [STIR] Branch, M.A., T.F. Coleman, and Y. Li, "A Subspace, Interior,
    #       and Conjugate Gradient Method for Large-Scale Bound-Constrained
    #       Minimization Problems," SIAM Journal on Scientific Computing,
    #       Vol. 21, Number 1, pp 1-23, 1999.
    # .. [JJMore] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
    #     and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
    ##################################################################################################

    def prepare_bounds(self, bounds, n):
        """ Prepare the bounds of the variables, release them into two tuples `lb` and `ub` """
        if len(bounds) != n:
            raise ValueError('The length of `bounds` {} must align with the `variables` {}'\
                             .format(len(bounds), n))
        b = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        # if the value from json out of the INFVALUE, set to infinite
        b[b > self.INFVAL] = torch.inf
        b[b < -self.INFVAL] = -torch.inf
        lb, ub = b[..., 0], b[..., 1]
        return lb, ub

    def in_bounds(self, x, lb, ub):
        return torch.all((x >= lb) & (x <= ub))

    def find_active_constraints(self, x, lb, ub, rtol=1e-10):
        """Determine which constraints are active in a given point.

        The threshold is computed using `rtol` and the absolute value of the
        closest bound.

        Returns
        -------
        active : ndarray of int with shape of x
            Each component shows whether the corresponding constraint is active:

                *  0 - a constraint is not active.
                * -1 - a lower bound is active.
                *  1 - a upper bound is active.
        """
        active = torch.zeros_like(x, dtype=torch.int)

        if rtol == 0:
            active[x <= lb] = -1
            active[x >= ub] = 1
            return active

        lower_dist = x - lb
        upper_dist = ub - x

        lower_threshold = rtol * torch.maximum(
            torch.tensor(1., dtype=self.dtype, device=self.device), torch.abs(lb))
        upper_threshold = rtol * torch.maximum(
            torch.tensor(1., dtype=self.dtype, device=self.device), torch.abs(ub))

        lower_active = (torch.isfinite(lb) &
                        (lower_dist <= torch.minimum(upper_dist, lower_threshold)))
        active[lower_active] = -1

        upper_active = (torch.isfinite(ub) &
                        (upper_dist <= torch.minimum(lower_dist, upper_threshold)))
        active[upper_active] = 1

        return active

    def make_strictly_feasible(self, x, lb, ub, rstep=1e-10):
        """Shift a point to the interior of a feasible region.

        Each element of the returned vector is at least at a relative distance
        `rstep` from the closest bound. If ``rstep=0`` then `np.nextafter` is used.
        """
        x_new = x.clone()

        active = self.find_active_constraints(x, lb, ub, rstep)
        lower_mask = torch.equal(active, 
                                 torch.tensor(-1., dtype=self.dtype, device=self.device))
        upper_mask = torch.equal(active, 
                                 torch.tensor(1., dtype=self.dtype, device=self.device))
        
        x_new[lower_mask] = (lb[lower_mask] + rstep * torch.maximum(
            torch.tensor(1., dtype=self.dtype, device=self.device), torch.abs(lb[lower_mask])))
        x_new[upper_mask] = (ub[upper_mask] - rstep * torch.maximum(
            torch.tensor(1., dtype=self.dtype, device=self.device), torch.abs(ub[upper_mask])))

        tight_bounds = (x_new < lb) | (x_new > ub)
        x_new[tight_bounds] = 0.5 * (lb[tight_bounds] + ub[tight_bounds])

        return x_new

    def soft_l1(self, z, rho, cost_only):
        t = 1 + z
        rho[0] = 2 * (t**0.5 - 1)
        if cost_only:
            return
        rho[1] = t**-0.5
        rho[2] = -0.5 * t**-1.5

    def construct_loss_function(self, m, loss, f_scale):
        if loss == 'linear':
            return None
        elif loss == 'soft_l1':
            loss = self.soft_l1
            rho = torch.empty((3, m))

            def loss_function(f, cost_only=False):
                z = (f / f_scale) ** 2
                loss(z, rho, cost_only=cost_only)
                if cost_only:
                    return 0.5 * f_scale ** 2 * torch.sum(rho[0])
                rho[0] *= f_scale ** 2
                rho[2] /= f_scale ** 2
                return rho
            
        return loss_function
    
    ##################################################################################
    # jacobian with finite difference

    def _compute_absolute_step(self, x0, method):
        """
        Computes an absolute step from a relative step for finite difference
        calculation.
        """
        sign_x0 = (x0 >= 0).clone() * 2 - 1

        # Calculates relative EPS step to use for a given data type and numdiff step method.
        # Attention: the x0 and the f0 are in under the same data type
        eps_finite = torch.finfo(self.dtype).eps
        if method == '2-point':
            rstep = eps_finite ** 0.5
        elif method == '3-point':
            rstep = eps_finite ** (1/3)
        else:
            raise NotImplementedError("Have not implemented!")
        
        abs_step = rstep * sign_x0 * torch.maximum(
            torch.tensor(1., dtype=self.dtype, device=self.device), torch.abs(x0))
        
        return abs_step
    
    def _adjust_scheme_to_bounds(self, x0, h, num_steps, scheme, lb, ub):
        """
        Adjust final difference scheme to the presence of bounds.
        """
        if scheme == '1-sided':
            use_one_sided = torch.ones_like(h, dtype=torch.bool)
        elif scheme == '2-sided':
            h = torch.abs(h)
            use_one_sided = torch.zeros_like(h, dtype=torch.bool)
        else:
            raise ValueError("`scheme` must be '1-sided' or '2-sided'.")

        if torch.all((lb == -torch.inf) & (ub == torch.inf)):
            return h, use_one_sided

        h_total = h * num_steps
        h_adjusted = h.clone()

        lower_dist = x0 - lb
        upper_dist = ub - x0

        if scheme == '1-sided':
            x = x0 + h_total
            violated = (x < lb) | (x > ub)
            fitting = torch.abs(h_total) <= torch.maximum(lower_dist, upper_dist)
            h_adjusted[violated & fitting] *= -1

            forward = (upper_dist >= lower_dist) & ~fitting
            h_adjusted[forward] = upper_dist[forward] / num_steps
            backward = (upper_dist < lower_dist) & ~fitting
            h_adjusted[backward] = -lower_dist[backward] / num_steps
        elif scheme == '2-sided':
            central = (lower_dist >= h_total) & (upper_dist >= h_total)

            forward = (upper_dist >= lower_dist) & ~central
            h_adjusted[forward] = torch.minimum(
                h[forward], 0.5 * upper_dist[forward] / num_steps)
            use_one_sided[forward] = True

            backward = (upper_dist < lower_dist) & ~central
            h_adjusted[backward] = -torch.minimum(
                h[backward], 0.5 * lower_dist[backward] / num_steps)
            use_one_sided[backward] = True

            min_dist = torch.minimum(upper_dist, lower_dist) / num_steps
            adjusted_central = (~central & (torch.abs(h_adjusted) <= min_dist))
            h_adjusted[adjusted_central] = min_dist[adjusted_central]
            use_one_sided[adjusted_central] = False

        return h_adjusted, use_one_sided

    def approx_derivative(self, func, x0, method='3-point', f0=None, lb=None, ub=None):
        """
        Compute finite difference approximation of the derivatives of a
        vector-valued function.

        If a function maps from R^n to R^m, its derivatives form m-by-n matrix
        called the Jacobian, where an element (i, j) is a partial derivative of
        f[i] with respect to x[j].
        """

        if lb.shape != x0.shape or ub.shape != x0.shape:
            raise ValueError("Inconsistent shapes between bounds and `x0`.")
        
        if f0 is None:
            f0 = func(x0)
        else:
            assert len(f0.shape) == 1, ValueError('`func` return value has \
                                                  more than 1 dimension')
            
        if not self.in_bounds(x0, lb, ub):
            raise ValueError("`x0` is infeasible")
        
        # evaluate the step for grad calculation
        h = self._compute_absolute_step(x0, method)

        if method == '2-point':
            h, use_one_sided = self._adjust_scheme_to_bounds(
                x0, h, 1, '1-sided', lb, ub
            )
        elif method == '3-point':
            h, use_one_sided = self._adjust_scheme_to_bounds(
                x0, h, 1, '2-sided', lb, ub
            )
        else:
            raise NotImplementedError("Have not implemented!")
        
        # dense difference to calculate the jacobian matrix
        n = x0.shape[0]
        m = f0.shape[0]
        J_transposed = torch.empty((n, m), dtype=self.dtype, device=self.device)
        h_vecs = torch.diag(h)

        for i in range(h.shape[0]):
            if method == '2-point':
                x = x0 + h_vecs[i]
                dx = x[i] - x0[i] # Recompute dx as exactly representable number.
                df = func(x) - f0
            elif method == '3-point' and use_one_sided[i]:
                x1 = x0 + h_vecs[i]
                x2 = x0 + 2 * h_vecs[i]
                dx = x2[i] - x0[i]
                f1 = func(x1)
                f2 = func(x2)
                df = -3.0 * f0 + 4 * f1 - f2
            elif method == '3-point' and not use_one_sided[i]:
                x1 = x0 - h_vecs[i]
                x2 = x0 + h_vecs[i]
                dx = x2[i] - x1[i]
                f1 = func(x1)
                f2 = func(x2)
                df = f2 - f1
            else:
                raise NotImplementedError("Have not implemented!")
            
            J_transposed[i] = df / dx

        if m == 1:
            J_transposed = torch.ravel(J_transposed)
            # J is guaranteed not sparse with 2 dimension (for transpose)
            J_transposed = torch.atleast_2d(J_transposed)
            
        return J_transposed.T

    ##################################################################################

    ##################################################################################
    # built in trust region reflective algorithm

    def scale_for_robust_loss_function(self, J, f, rho):
        """Scale Jacobian and residuals for a robust loss function."""
        J_scale = rho[1] + 2 * rho[2] * f**2
        J_scale[J_scale < torch.finfo(torch.float64).eps] = torch.finfo(torch.float64).eps
        J_scale **= 0.5

        f *= rho[1] / J_scale

        # rescale the jacobian
        J *= J_scale[..., None]
        return J, f
        
    # copied from scipy.optimize
    def solve_lsq_trust_region(self, n, m, uf, s, V, Delta, initial_alpha=None,
                               rtol=0.01, max_iter=10):
        """Solve a trust-region problem arising in least-squares minimization.

        This function implements a method described by J. J. More [1]_ and used
        in MINPACK, but it relies on a single SVD of Jacobian instead of series
        of Cholesky decompositions. Before running this function, compute:
        ``U, s, VT = svd(J, full_matrices=False)``.

        Parameters
        ----------
        n : int
            Number of variables.
        m : int
            Number of residuals.
        uf : ndarray
            Computed as U.T.dot(f).
        s : ndarray
            Singular values of J.
        V : ndarray
            Transpose of VT.
        Delta : float
            Radius of a trust region.
        initial_alpha : float, optional
            Initial guess for alpha, which might be available from a previous
            iteration. If None, determined automatically.
        rtol : float, optional
            Stopping tolerance for the root-finding procedure. Namely, the
            solution ``p`` will satisfy ``abs(norm(p) - Delta) < rtol * Delta``.
        max_iter : int, optional
            Maximum allowed number of iterations for the root-finding procedure.

        Returns
        -------
        p : ndarray, shape (n,)
            Found solution of a trust-region problem.
        alpha : float
            Positive value such that (J.T*J + alpha*I)*p = -J.T*f.
            Sometimes called Levenberg-Marquardt parameter.
        n_iter : int
            Number of iterations made by root-finding procedure. Zero means
            that Gauss-Newton step was selected as the solution.

        References
        ----------
        .. [1] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
            and Theory," Numerical Analysis, ed. G. A. Watson, Lecture Notes
            in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
        """
        def phi_and_derivative(alpha, suf, s, Delta):
            """Function of which to find zero.

            It is defined as "norm of regularized (by alpha) least-squares
            solution minus `Delta`". Refer to [1]_.
            """
            denom = s**2 + alpha
            p_norm = torch.norm(suf / denom)
            phi = p_norm - Delta
            phi_prime = -torch.sum(suf ** 2 / denom**3) / p_norm
            return phi, phi_prime

        suf = s * uf

        # Check if J has full rank and try Gauss-Newton step.
        if m >= n:
            threshold = torch.finfo(self.dtype).eps * m * s[0]
            full_rank = s[-1] > threshold
        else:
            full_rank = False

        if full_rank:
            p = torch.einsum('ij, j->i', -V, uf/s) # -V.dot(uf / s)
            if torch.norm(p) <= Delta:
                return p, 0.0, 0

        alpha_upper = torch.norm(suf) / Delta

        if full_rank:
            phi, phi_prime = phi_and_derivative(0.0, suf, s, Delta)
            alpha_lower = -phi / phi_prime
        else:
            alpha_lower = 0.0

        if initial_alpha is None or not full_rank and initial_alpha == 0:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)
        else:
            alpha = initial_alpha

        for it in range(max_iter):
            if alpha < alpha_lower or alpha > alpha_upper:
                alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)

            phi, phi_prime = phi_and_derivative(alpha, suf, s, Delta)

            if phi < 0:
                alpha_upper = alpha

            ratio = phi / phi_prime
            alpha_lower = max(alpha_lower, alpha - ratio)
            alpha -= (phi + Delta) * ratio / Delta

            if torch.abs(phi) < rtol * Delta:
                break

        p = torch.einsum('ij, j->i', -V, suf / (s**2 + alpha)) # -V.dot(suf / (s**2 + alpha))

        # Make the norm of p equal to Delta, p is changed only slightly during
        # this. It is done to prevent p lie outside the trust region (which can
        # cause problems later).
        p *= Delta / torch.norm(p)

        return p, alpha, it + 1
    
    def evaluate_quadratic(self, J, g, s, diag=None):
        """Compute values of a quadratic function arising in least squares.
        The function is 0.5 * s.T * (J.T * J + diag) * s + g.T * s.
        """
        if s.ndim == 1:
            Js = torch.einsum('ij, j->i', J, s) # J_h.dot(step_h)
            q = torch.dot(Js, Js)
            if diag is not None:
                q += torch.dot(s * diag, s)
        else:
            Js = torch.einsum('ij, ij->i', J, s.T)
            q = torch.sum(Js ** 2, dim=0)
            if diag is not None:
                q += torch.sum(diag * s**2, axis=1)

        l = torch.dot(s, g)

        return 0.5 * q + l

    def update_tr_radius(self, Delta, actual_reduction, predicted_reduction,
                         step_norm, bound_hit):
        """Update the radius of a trust region based on the cost reduction.

        Returns
        -------
        Delta : float
            New radius.
        ratio : float
            Ratio between actual and predicted reductions.
        """
        if predicted_reduction > 0:
            ratio = actual_reduction / predicted_reduction
        elif predicted_reduction == actual_reduction == 0:
            ratio = 1
        else:
            ratio = 0

        if ratio < 0.25:
            Delta = 0.25 * step_norm
        elif ratio > 0.75 and bound_hit:
            Delta *= 2.0

        return Delta, ratio

    def check_termination(self, dF, F, dx_norm, x_norm, ratio, ftol, xtol):
        """Check termination condition for nonlinear least squares."""
        ftol_satisfied = dF < ftol * F and ratio > 0.25
        xtol_satisfied = dx_norm < xtol * (xtol + x_norm)

        if ftol_satisfied and xtol_satisfied:
            return 4
        elif ftol_satisfied:
            return 2
        elif xtol_satisfied:
            return 3
        else:
            return None

    def trf_no_bounds(self, func, jac, x0, f0, J0, 
                      ftol, xtol, gtol, max_nfev, loss_function):
        x = x0.clone()
        f = f0
        f_true = f.clone()
        nfev = 1

        J = J0
        njev = 1
        m, n = J.shape

        if loss_function is not None:
            rho = loss_function(f)
            cost = 0.5 * torch.sum(rho[0])
            J, f = self.scale_for_robust_loss_function(J, f, rho)
        else:
            cost = 0.5 * torch.dot(f, f)
        
        g = torch.einsum('ij, j->i', J.T, f) # J.T.dot(f) calculate the grad
        
        # before update, we need to compute variables scale based on the jacobian 
        scale_inv = torch.sum(J**2, dim=0) ** 0.5
        scale_inv[scale_inv == 0] = 1
        scale = 1 / scale_inv

        Delta = torch.norm(x0 * scale_inv)
        if Delta == 0:
            Delta = torch.tensor(1., dtype=self.dtype, device=self.device)

        if max_nfev is None:
            max_nfev = n * 100 # 100 multiply the size of x0
        
        alpha = 0.0 # "Levenberg-Marquardt" parameter, also called damped factor

        termination_status = None
        iteration = 0
        step_norm = None
        actual_reduction = None

        while True:
            g_norm = torch.max(torch.abs(g))
            if g_norm < gtol:
                termination_status = 1

            if termination_status is not None or nfev == max_nfev:
                break

            d = scale
            g_h = d * g

            # solve with exact method because our jacobian is dense descriped
            J_h = J * d
            U, s, V = torch.linalg.svd(J_h, full_matrices=False)
            V = V.T
            uf = torch.einsum('ij, j->i', U.T, f)
            
            actual_reduction = -1
            while actual_reduction <=0 and nfev < max_nfev:
                # Solve a trust-region problem in least-squares minimization
                step_h, alpha, n_iter = self.solve_lsq_trust_region(
                    n, m, uf, s, V, Delta, initial_alpha=alpha
                )

                # compute values of a quadratic function in least squares
                predicted_reduction = -self.evaluate_quadratic(J_h, g_h, step_h)
                step = d * step_h
                x_new = x + step
                f_new = func(x_new)
                nfev += 1

                step_h_norm = torch.norm(step_h)

                if not torch.all(torch.isfinite(f_new)):
                    Delta = 0.25 * step_h_norm
                    continue    

                cost_new = 0.5 * torch.dot(f_new, f_new)
                actual_reduction = cost - cost_new

                Delta_new, ratio = self.update_tr_radius(
                    Delta, actual_reduction, predicted_reduction,
                    step_h_norm, step_h_norm > 0.95 * Delta)
                
                step_norm = torch.norm(step)
                termination_status = self.check_termination(
                    actual_reduction, cost, step_norm, torch.norm(x), ratio, ftol, xtol)

                if termination_status is not None:
                    break
                
                alpha *= Delta / Delta_new
                Delta = Delta_new

            # check the validation of solution if actual reduction larger than 0
            if actual_reduction > 0:
                x = x_new
                f = f_new
                f_true = f.clone()

                cost = cost_new

                J = jac(x, f)
                njev += 1

                if loss_function is not None:
                    rho = loss_function(f)
                    J, f = self.scale_for_robust_loss_function(J, f, rho)

                g = torch.einsum('ij, j->i', J.T, f)

                scale_inv = torch.sum(J**2, dim=0) ** 0.5
                scale_inv[scale_inv == 0] = 1
                scale = 1 / scale_inv
                # save the system configuration
                self.system.save_to_json('./lens_file/' + self.system.LensName +
                                         datetime.now().strftime('_%y%m%d_%H%M%S') + 
                                         '.json')

            else:
                step_norm = 0
                actual_reduction = 0

            # return the optmized solution in this situation
            print('<iter>: {:4d}, merit func norm: {:.8e}, cost: {:.8e}, \
                  first-order optimality {:.4e}, termination status: {}'
                  .format(iteration, torch.norm(f_true).item(), cost.item(), 
                          g_norm.item(), termination_status))
            iteration += 1

        return {'nfev': nfev, 'cost': cost.item(), 'optimality': g_norm.item()}

    ##################################################################################
    def step_size_to_bound(self, x, s, lb, ub):
        """Compute a min_step size required to reach a bound."""
        non_zero = torch.nonzero(s)
        s_non_zero = s[non_zero]
        steps = torch.empty_like(x)
        steps[...] = torch.inf # default step
        steps[non_zero] = torch.maximum((lb - x)[non_zero] / s_non_zero,
                                        (ub - x)[non_zero] / s_non_zero)
        min_step = torch.min(steps)
        return min_step, torch.int(torch.equal(steps, min_step) * torch.sign(s))
    
    def intersect_trust_region(self, x, s, Delta):
        """Find the intersection of a line with the boundary of a trust region.

        This function solves the quadratic equation with respect to t
        ||(x + s*t)||**2 = Delta**2."""
        a = torch.dot(s, s)
        if a == 0:
            raise ValueError("`step` is zero.")
        
        b = torch.dot(x, s)

        c = torch.dot(x, x) - Delta**2
        if c > 0:
            raise ValueError("`x` is not within the trust region.")
        
        d = torch.sqrt(b*b - a*c) # Root from one fourth of the discriminant

        # avoid loss of significance
        q = -(b + torch.sign(b) * d)
        t1 = q / a
        t2 = c / q

        if t1 < t2:
            return t1, t2
        else:
            return t2, t1
        
    def build_quadratic_1d(self, J, g, s, diag=None, s0=None):
        """Parameterize a multivariate quadratic function along a line.

        The resulting univariate quadratic function is given as follows::

            f(t) = 0.5 * (s0 + s*t).T * (J.T*J + diag) * (s0 + s*t) +
                g.T * (s0 + s*t)

        Parameters
        ----------
        J : ndarray, sparse matrix or LinearOperator shape (m, n)
            Jacobian matrix, affects the quadratic term.
        g : ndarray, shape (n,)
            Gradient, defines the linear term.
        s : ndarray, shape (n,)
            Direction vector of a line.
        diag : None or ndarray with shape (n,), optional
            Addition diagonal part, affects the quadratic term.
            If None, assumed to be 0.
        s0 : None or ndarray with shape (n,), optional
            Initial point. If None, assumed to be 0.

        Returns
        -------
        a : float
            Coefficient for t**2.
        b : float
            Coefficient for t.
        c : float
            Free term. Returned only if `s0` is provided.
        """
        v = torch.einsum('ij, j->i', J, s)
        a = torch.dot(v, v)
        if diag is not None:
            a += torch.dot(s * diag, s)
        a *= 0.5

        b = torch.dot(g, s)

        if s0 is not None:
            u = torch.einsum('ij, j->i', J, s0)
            b += torch.dot(u, v)
            c = 0.5 * torch.dot(u, u) + torch.dot(g, s0)
            if diag is not None:
                b += torch.dot(s0 * diag, s)
                c += 0.5 * torch.dot(s0 * diag, s0)
            return a, b, c
        else:
            return a, b
        
    def minimize_quadratic_1d(self, a, b, lb, ub, c=0):
        """Minimize a 1-D quadratic function subject to bounds.

        The free term `c` is 0 by default. Bounds must be finite.

        Returns
        -------
        t : float
            Minimum point.
        y : float
            Minimum value.
        """
        t = [lb, ub]
        if a != 0:
            extremum = -0.5 * b / a
            if lb < extremum < ub:
                t.append(extremum)
        t = torch.tensor(t, dtype=self.dtype, device=self.device)
        y = t * (a * t + b) + c
        min_index = torch.argmin(y)
        return t[min_index], y[min_index]

    def CL_scaling_vector(self, x, g, lb, ub):
        """Compute Coleman-Li scaling vector and its derivatives.

        Components of a vector v are defined as follows::

                    | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
            v[i] =  | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
                    | 1,           otherwise
        Derivatives of v with respect to x take value 1, -1 or 0 depending on a
        case.
        """
        v = torch.ones_like(x)
        dv = torch.zeros_like(x)

        mask = (g < 0) & torch.isfinite(ub)
        v[mask] = ub[mask] - x[mask]
        dv[mask] = -1

        mask = (g > 0) & torch.isfinite(lb)
        v[mask] = x[mask] - lb[mask]
        dv[mask] = 1
    
        return v, dv

    def select_step(self, x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta):
        """Select the best step according to Trust Region Reflective algorithm."""
        if self.in_bounds(x + p, lb, ub):
            p_value = self.evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
            return p, p_h, -p_value

        p_stride, hits = self.step_size_to_bound(x, p, lb, ub)

        # Compute the reflected direction.
        r_h = p_h.clone().detach()
        r_h[torch.bool(hits.astype)] *= -1
        r = d * r_h

        # Restrict trust-region step, such that it hits the bound.
        p *= p_stride
        p_h *= p_stride
        x_on_bound = x + p

        # Reflected direction will cross first either feasible region or trust region boundary.
        _, to_tr = self.intersect_trust_region(p_h, r_h, Delta)
        to_bound, _ = self.step_size_to_bound(x_on_bound, r, lb, ub)

        # Find lower and upper bounds on a step size along the reflected
        # direction, considering the strict feasibility requirement. There is no
        # single correct way to do that, the chosen approach seems to work best
        # on test problems.
        r_stride = min(to_bound, to_tr)
        if r_stride > 0:
            r_stride_l = (1 - theta) * p_stride / r_stride
            if r_stride == to_bound:
                r_stride_u = theta * to_bound
            else:
                r_stride_u = to_tr
        else:
            r_stride_l = 0
            r_stride_u = -1

        # Check if reflection step is available.
        if r_stride_l <= r_stride_u:
            a, b, c = self.build_quadratic_1d(J_h, g_h, r_h, s0=p_h, diag=diag_h)
            r_stride, r_value = self.minimize_quadratic_1d(
                a, b, r_stride_l, r_stride_u, c=c)
            r_h *= r_stride
            r_h += p_h
            r = r_h * d
        else:
            r_value = np.inf

        # Now correct p_h to make it strictly interior.
        p *= theta
        p_h *= theta
        p_value = self.evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)

        ag_h = -g_h
        ag = d * ag_h

        to_tr = Delta / torch.norm(ag_h)
        to_bound, _ = self.step_size_to_bound(x, ag, lb, ub)
        if to_bound < to_tr:
            ag_stride = theta * to_bound
        else:
            ag_stride = to_tr

        a, b = self.build_quadratic_1d(J_h, g_h, ag_h, diag=diag_h)
        ag_stride, ag_value = self.minimize_quadratic_1d(a, b, 0, ag_stride)
        ag_h *= ag_stride
        ag *= ag_stride

        if p_value < r_value and p_value < ag_value:
            return p, p_h, -p_value
        elif r_value < p_value and r_value < ag_value:
            return r, r_h, -r_value
        else:
            return ag, ag_h, -ag_value

    def trf_bounds(self, func, jac, x0, f0, J0, lb, ub, 
                   ftol, xtol, gtol, max_nfev, loss_function):
        """"""
        x = x0.clone()
        f = f0
        f_true = f.clone()
        nfev = 1

        J = J0
        njev = 1
        m, n = J.shape
        
        if loss_function is not None:
            rho = loss_function(f)
            cost = 0.5 * torch.sum(rho[0])
            J, f = self.scale_for_robust_loss_function(J, f, rho)
        else:
            cost = 0.5 * torch.dot(f, f)

        g = torch.einsum('ij, j->i', J.T, f) # J.T.dot(f) calculate the grad
        
        # before update, we need to compute variables scale based on the jacobian 
        scale_inv = torch.sum(J**2, dim=0) ** 0.5
        scale_inv[scale_inv == 0] = 1
        scale = 1 / scale_inv

        # compute the scaling vector and its derivatives
        v, dv = self.CL_scaling_vector(x, g, lb, ub)
        v[dv != 0] *= scale_inv[dv != 0]
        Delta = torch.norm(x0 * scale_inv / v**0.5)
        if Delta == 0:
            Delta = torch.tensor(1., dtype=self.dtype, device=self.device)

        g_norm = torch.max(torch.abs(g * v))

        f_augmented = torch.zeros(m+n, dtype=self.dtype, device=self.device)
        J_augmented = torch.empty((m+n, n), dtype=self.dtype, device=self.device) # augment for exact solver

        if max_nfev is None:
            max_nfev = n * 100 # 100 multiply the size of x0

        alpha = 0.0 # "Levenberg-Marquardt" parameter
        termination_status = None
        iteration = 0
        step_norm = None
        actual_reduction = None

        while True:
            v, dv = self.CL_scaling_vector(x, g, lb, ub)

            g_norm = torch.max(torch.abs(g))
            if g_norm < gtol:
                termination_status = 1

            if termination_status is not None or nfev == max_nfev:
                break
            
            # v is recomputed in the variables after applying `x_scale`, note that
            # components which were identically 1 not affected.
            v[dv != 0] *= scale_inv[dv != 0]

            d = v ** 0.5 * scale # here, two type of scaling is applied

            # C = diag(g * scale) Jv
            diag_h = g * dv * scale

            # "hat" gradient
            g_h = d * g
            
            # similar to the svd in no bounds
            f_augmented[:m] = f
            J_augmented[:m] = J * d
            J_h = J_augmented[:m]  # Memory view.
            J_augmented[m:] = torch.diag(diag_h**0.5)
            U, s, V = torch.linalg.svd(J_augmented, full_matrices=False)
            V = V.T
            uf = torch.einsum('ij, j->i', U.T, f_augmented)

            # theta controls step back step ratio from the bounds
            theta = max(0.995, 1 - g_norm)

            actual_reduction = -1
            while actual_reduction <= 0 and nfev <= max_nfev:
                # Solve a trust-region problem in least-squares minimization
                p_h, alpha, n_iter = self.solve_lsq_trust_region(
                    n, m, uf, s, V, Delta, initial_alpha=alpha
                )

                p = d * p_h  # Trust-region solution in the original space.
                step, step_h, predicted_reduction = self.select_step(
                    x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)

                x_new = self.make_strictly_feasible(x + step, lb, ub, rstep=0)
                f_new = func(x_new)
                nfev += 1

                step_h_norm = torch.norm(step_h)

                if not torch.all(torch.isfinite(f_new)):
                    Delta = 0.25 * step_h_norm
                    continue    

                cost_new = 0.5 * torch.dot(f_new, f_new)
                actual_reduction = cost - cost_new

                Delta_new, ratio = self.update_tr_radius(
                    Delta, actual_reduction, predicted_reduction,
                    step_h_norm, step_h_norm > 0.95 * Delta)
                
                step_norm = torch.norm(step)
                termination_status = self.check_termination(
                    actual_reduction, cost, step_norm, torch.norm(x), ratio, ftol, xtol)

                if termination_status is not None:
                    break
                
                alpha *= Delta / Delta_new
                Delta = Delta_new

            # check the validation of solution if actual reduction larger than 0
            if actual_reduction > 0:
                x = x_new
                f = f_new
                f_true = f.clone()

                cost = cost_new

                J = jac(x, f)
                njev += 1

                if loss_function is not None:
                    rho = loss_function(f)
                    J, f = self.scale_for_robust_loss_function(J, f, rho)

                g = torch.einsum('ij, j->i', J.T, f)

                scale_inv = torch.sum(J**2, dim=0) ** 0.5
                scale_inv[scale_inv == 0] = 1
                scale = 1 / scale_inv
                
                # save the system configuration
                self.system.save_to_json('./lens_file/' + self.system.LensName +
                                         datetime.now().strftime('_%y%m%d_%H%M%S') + 
                                         '.json')

            else:
                step_norm = 0
                actual_reduction = 0

            # return the optmized solution in this situation
            print('<iter>: {:4d}, merit func norm: {:.8e}, cost: {:.8e}, first-order optimality {:.4e}, termination status: {}'
                  .format(iteration, torch.norm(f_true).item(), cost.item(), 
                          g_norm.item(), termination_status))
            iteration += 1

        return {'nfev': nfev, 'cost': cost.item(), 'optimality': g_norm.item()}

    ##################################################################################

    def optimize_wz_trf(self, jac='2-point', loss='soft_l1', f_scale=1.0, 
                        ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=None):
        """
        optmize the lens geometry with the trust region reflective algorithm
            Input:
                jac : {'2-point', '3-point', 'autograd'}, optional
                    Method of computing the Jacobian matrix (an m-by-n matrix, where
                    element (i, j) is the partial derivative of f[i] with respect to
                    x[j]). The keywords select a finite difference scheme for numerical
                    estimation. The scheme '3-point' is more accurate, but requires
                    twice as many operations as '2-point' (default). The scheme 'cs'
                    uses complex steps, and while potentially the most accurate, it is
                    applicable only when `fun` correctly handles complex inputs and
                    can be analytically continued to the complex plane. Method 'lm'
                    always uses the '2-point' scheme. If callable, it is used as
                    ``jac(x, *args, **kwargs)`` and should return a good approximation
                    (or the exact value) for the Jacobian as an array_like (np.atleast_2d
                    is applied), a sparse matrix (csr_matrix preferred for performance) or
                    a `scipy.sparse.linalg.LinearOperator`.

                loss : str or callable, optional
                    Determines the loss function. The following keyword values are allowed:

                        * 'linear' (default) : ``rho(z) = z``. Gives a standard
                        least-squares problem.
                        * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
                        approximation of l1 (absolute value) loss. Usually a good
                        choice for robust least squares.

                f_scale : float, optional
                    Value of soft margin between inlier and outlier residuals, default
                    is 1.0. The loss function is evaluated as follows
                    ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`,
                    and ``rho`` is determined by `loss` parameter. This parameter has
                    no effect with ``loss='linear'``, but for other `loss` values it is
                    of crucial importance.
        """
        # get the initial merit function value
        x0 = torch.tensor(self.variables, dtype=self.dtype, device=self.device) # list to tensor
        # prepare the bound of all parameters
        lb, ub = self.prepare_bounds(self.bounds, len(self.variables))
        if not self.in_bounds(x0, lb, ub):
            raise ValueError("`x0` is infeasible")
        
        # construct a built-in function for integrate all evaluations
        def fun_integrated(variables_update, if_save=False):
            # update geometry
            self.update_geometry(variables_update, if_save)
            f = []
            for func_idx in range(len(self.merit_funclist)):
                merit_funcval = self.merit_funclist[func_idx]()
                assert merit_funcval.shape == self.merit_functarget[func_idx].shape, \
                    ValueError("the output shape of the `merit function`: {} \
                                must align with the shape of `merit target`: {}!".format(
                                   merit_funcval.shape, self.merit_functarget[func_idx].shape))
                
                f.append(self.merit_funcweight[func_idx] * 
                         torch.abs(self.merit_funclist[func_idx]() - self.merit_functarget[func_idx]))

            return torch.concatenate(f, dim=0)
        
        # shift a point to the interior of a feasible region.
        x0 = self.make_strictly_feasible(x0, lb, ub)
        
        # TODO: f0 is a tensor with the shape of views (what about the wavelengths?)
        f0 = fun_integrated(x0)
        if f0.ndim != 1:
            raise ValueError("`merit function` must return at most 1-d array_like. "
                             "f0.shape: {}".format(f0.shape))

        if not torch.all(torch.isfinite(f0)):
            raise ValueError("Residuals are not finite in the initial point.")

        n = x0.shape[0]
        m = f0.shape[0]
        
        ##
        # judge the relation between m and n
        # if m < n, this nonlinear problem is underdetermined, 
        # if m = n, this nonlinear problem is positive definite,
        # if m > n, this nonlinear system is overdetermined.
        # in trust region reflective algorithm, we don't need to judge the situation
        ##

        # construct the loss function for robust optimization
        loss_function = self.construct_loss_function(m, loss, f_scale)
        if loss_function is not None:
            initial_cost = loss_function(f0, cost_only=True)
        else:
            initial_cost = 0.5 * torch.dot(f0, f0)
        
        # dense differencing to calculate the jacobian matrix
        # could be calculated by finite differences or autograd, i.e., torch.autograd.functional.jvp()
        def jac_calc(x, f):
            J = self.approx_derivative(fun_integrated, x0=x, method=jac, f0=f, lb=lb, ub=ub)
            return J

        J0 = jac_calc(x0, f0)
        # check the shape of J0
        if J0.shape != (m, n):
            raise ValueError(
                "The return value of `jac` has wrong shape: expected {}, "
                "actual {}.".format((m, n), J0.shape))
        

        if torch.all(lb == -torch.inf) and torch.all(ub == torch.inf):
            result_dict = self.trf_no_bounds(fun_integrated, jac_calc, x0, f0, J0,
                                      ftol, xtol, gtol, max_nfev, loss_function)
            print('Optimized without bounds, merit function evaluating times: {:4d}, initial cost {:.4e}, final cost {:.4e}, first-order optimality {:.2e}'
                  .format(result_dict['nfev'], initial_cost.item(), 
                          result_dict['cost'], result_dict['optimality']))
        else:
            # raise NotImplementedError('Havent implemented!')
            result_dict = self.trf_bounds(fun_integrated, jac_calc, x0, f0, J0, lb, ub,
                                   ftol, xtol, gtol, max_nfev, loss_function)
            print('Optimized with bounds, merit function evaluating times: {:4d}, initial cost {:.4e}, final cost {:.4e}, first-order optimality {:.2e}'
                  .format(result_dict['nfev'], initial_cost.item(), 
                          result_dict['cost'], result_dict['optimality']))

    # for testing
    def optimize_wz_scipy(self, jac='2-point', bounds=(-torch.inf, torch.inf), loss='soft_l1', 
                        f_scale=1.0, ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=None):
        from scipy.optimize import least_squares, minimize
        # get the initial merit function value
        x0 = torch.tensor(self.variables).numpy()

        # construct a built-in function for integrate all evaluations        
        def fun_integrated(variables_update, if_save):
            # update geometry
            # print('variables now:', variables_update)
            self.update_geometry(torch.tensor(variables_update, dtype=self.dtype, device=self.device), if_save)
            f = []
            for func_idx in range(len(self.merit_funclist)):
                merit_funcval = self.merit_funclist[func_idx]()
                assert merit_funcval.shape == self.merit_functarget[func_idx].shape, \
                    ValueError("the output shape of the `merit function`: {} \
                               must align with the shape of `merit target`: {}!".format(
                                   merit_funcval.shape, self.merit_functarget[func_idx].shape))
                
                f.append(self.merit_funcweight[func_idx] * 
                         torch.abs(self.merit_funclist[func_idx]() - self.merit_functarget[func_idx]))
            # print('merit value', torch.concatenate(f, dim=0).sum().numpy())
            return torch.concatenate(f, dim=0).numpy()

        result = least_squares(fun_integrated, x0, x_scale='jac', bounds=(0, 10.0), loss='soft_l1', tr_solver='exact')
        # result = minimize(fun_integrated, x0)

    ##################################################################################################