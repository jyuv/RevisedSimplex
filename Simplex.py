import math
import sys
from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy.special import comb
from scipy import linalg
from enum import Enum


class PickingRule(Enum):
    BLAND_RULE = 1
    DANTZING_RULE = 2


class ResultCode(Enum):
    INFEASIBLE = 1
    FINITE_OPTIMAL = 2
    UNBOUNDED_OPTIMAL = 3
    CYCLE_DETECTED = 4


@dataclass
class SimplexResult(object):
    res_code: ResultCode
    assignment: Union[None, np.ndarray]
    optimal_score: Union[None, float]


EPSILON = sys.float_info.epsilon
PAIRS_LIMIT = 5


def is_close_to_zero(arr, or_below=False, or_above=False) -> np.ndarray:
    if or_below and or_above:
        return np.full(arr.shape, True)
    elif not or_below and not or_above:
        return np.isclose(arr, 0, atol=EPSILON)
    elif or_below:
        return np.logical_or(np.isclose(arr, 0, atol=EPSILON), arr <= 0)
    else:
        return np.logical_or(np.isclose(arr, 0, atol=EPSILON), arr >= 0)


# For solving maximum problems
class Simplex(object):
    def __init__(self):
        self.A = None
        self.cur_assignment = None
        self.c = None

    def _load_scenario(self, A, b, c):
        self.b = b.astype(float).reshape((-1, 1))
        self.nrows, self.original_nvars = A.shape
        self.A = np.hstack((A, np.eye(self.nrows)))
        self.c = np.hstack((c, np.zeros(self.nrows)))
        self.nvars = self.A.shape[1]

        self.xb = np.arange(self.nvars - self.nrows, self.nvars)
        self.xn = np.arange(0, self.nvars - self.nrows)

        self.cur_assignment = np.hstack((np.zeros(self.original_nvars),
                                         self.b.copy().reshape(-1)))
        self.cur_assignment = self.cur_assignment.astype(float)

        self.estimated_B = np.eye(self.nrows).astype(float)
        self.estimated_inverse_B = np.eye(self.nrows).astype(float)

        self.cur_iteration = 1
        self.max_iterations = comb(self.nvars, self.nrows)

    # z is a representation of the objective function only using the basis vars
    def _reconstruct_z_coefs(self):
        return self.c[self.xn] - self.c[self.xb] @ self.estimated_inverse_B @\
               self.A[:, self.xn]

    # This will get us options for variables to insert to the basis
    # (the picked one will be replaced with a var currently in the basis)
    def _get_entering_options(self, z_coefs, rule: PickingRule):
        if rule == PickingRule.DANTZING_RULE:
            sorted_indices = z_coefs.argsort()[-PAIRS_LIMIT:][::-1]
            return self.xn[sorted_indices]
        elif rule == PickingRule.BLAND_RULE:
            sorted_positive_vars = sorted(self.xn[np.where(z_coefs > 0)])
            return sorted_positive_vars[:PAIRS_LIMIT]
        else:
            raise ValueError("Unrecognized picking rule")

    # leaving var is the var we want to remove from the basis
    def _pick_leaving_var(self, d):
        bounded_vars_idxs = self.xb[d > 0]
        bounds_vector = self.cur_assignment[bounded_vars_idxs] / d[d > 0]
        t = min(bounds_vector)
        leaving_idx = bounded_vars_idxs[np.argmin(bounds_vector)]
        return leaving_idx, t

    # enables numerical stability option for diag element in eta
    # returns tuple entering_idx, leaving_idx, d, t
    # if found unbounded return entering_idx, None, d, None
    def _pick_pair_to_swap(self, z_coefs, rule: PickingRule):
        potential_entering = self._get_entering_options(z_coefs, rule)
        optional_pairs = []
        for entering_var_idx in potential_entering:
            d = self._get_d(entering_var_idx)
            if all(is_close_to_zero(d, or_below=True)):
                return entering_var_idx, None, d, None

            leaving_var_idx, t = self._pick_leaving_var(d)
            leaving_basis_loc = np.where(self.xb == leaving_var_idx)[0][0]
            eta_diag_element = d[leaving_basis_loc]
            if eta_diag_element > EPSILON:
                return entering_var_idx, leaving_var_idx, d, t
            else:
                optional_pairs.append((eta_diag_element, entering_var_idx,
                                      leaving_var_idx, d, t))

        optional_etas = [x[0] for x in optional_pairs]
        largest_eta_option_index = int(np.argmax(optional_etas))
        return optional_pairs[largest_eta_option_index][1:]

    # d is the negation the entering_var coefs in the equations
    def _get_d(self, entering_var_idx):
        return self.estimated_inverse_B @ self.A[:, entering_var_idx]

    def _get_eta_and_inverse(self, loc_replaced, d):
        new_eta = np.eye(self.nrows).astype(float)
        new_eta[:, loc_replaced] = d

        new_inverse_eta = new_eta.copy()
        new_loc_on_diag = 1.0 / new_inverse_eta[loc_replaced, loc_replaced]
        new_inverse_eta[:, loc_replaced] *= -new_loc_on_diag
        new_inverse_eta[loc_replaced, loc_replaced] = new_loc_on_diag
        return new_eta, new_inverse_eta

    def _refactorize_if_needed(self):
        if np.linalg.cond(self.A[:, self.xb]) >= 1/EPSILON:
            # singular matrix
            return False

        estimated_b = self.estimated_B @ self.cur_assignment[self.xb]
        estimated_b = estimated_b.reshape((-1, 1))
        if not np.allclose(estimated_b, self.b, atol=EPSILON):
            self._refactorize_basis()
            return True
        return False

    def _refactorize_basis(self):
        l, u = linalg.lu(self.A[:, self.xb], True)
        self.estimated_inverse_B = linalg.inv(u) @ linalg.inv(l)

    def _update_estimated_b_and_inverse(self, loc_replaced, d):
        new_eta, new_inverse_eta = self._get_eta_and_inverse(loc_replaced, d)
        self.estimated_B = self.estimated_B @ new_eta
        self.estimated_inverse_B = new_inverse_eta @ self.estimated_inverse_B

    def _swap_vars(self, entering_var_idx, leaving_var_idx):
        d = self._get_d(entering_var_idx)
        loc_in_xb = np.where(self.xb == leaving_var_idx)[0][0]
        loc_in_xn = np.where(self.xn == entering_var_idx)[0][0]
        self.xb[loc_in_xb] = entering_var_idx
        self.xn[loc_in_xn] = leaving_var_idx

        self._update_estimated_b_and_inverse(loc_in_xb, d)

    def _update_assignment(self, d, t, entering_var_idx):
        self.cur_assignment[self.xn] = 0
        self.cur_assignment[self.xb] -= t * d
        self.cur_assignment[entering_var_idx] = t

    def _init_aux_problem(self, A, b, c):
        x_0_coefs = np.full((self.nrows, 1), -1)
        A = np.hstack((x_0_coefs, A))
        c = np.hstack((np.array([-1]), np.zeros(c.shape[0])))
        self._load_scenario(A, b, c)
        self.cur_assignment = np.zeros(self.cur_assignment.shape[0])

        most_negative_b = np.argmin(b)
        leaving_var_idx = self.xb[most_negative_b]
        self._swap_vars(0, leaving_var_idx)

        # adapt assignment
        self.cur_assignment[self.xn] = 0
        self.cur_assignment[self.xb] = self.b.reshape(-1) - float(min(b))
        self.cur_assignment[0] = - min(b)

    def _pivot_to_original_problem(self, original_c):
        self.A = self.A[:, 1:]
        self.xb -= 1
        self.xn = self.xn[np.where(self.xn != 0)] - 1
        self.c = np.hstack((original_c, np.zeros(self.nrows)))
        self.cur_assignment = self.cur_assignment[1:]
        self.cur_iteration = 1

    def _get_optimal_from_feasible_solution(self, rule: PickingRule)\
            -> SimplexResult:
        while self.cur_iteration <= self.max_iterations:
            z_coefs = self._reconstruct_z_coefs()
            if all(is_close_to_zero(z_coefs, or_below=True)):
                optimal_score = sum(self.cur_assignment * self.c)
                return SimplexResult(ResultCode.FINITE_OPTIMAL,
                                     self.cur_assignment, optimal_score)

            entering_idx, leaving_idx, d, t = self._pick_pair_to_swap(z_coefs,
                                                                      rule)
            if all(is_close_to_zero(d, or_below=True)):
                return SimplexResult(ResultCode.UNBOUNDED_OPTIMAL,
                                     self.cur_assignment, math.inf)

            self._swap_vars(entering_idx, leaving_idx)
            self._update_assignment(d, t, entering_idx)

            self.cur_iteration += 1
            self._refactorize_if_needed()

        cur_score = sum(self.cur_assignment * self.c)
        return SimplexResult(ResultCode.CYCLE_DETECTED, self.cur_assignment,
                             cur_score)

    def get_optimal_solution(self, A, b, c, rule=PickingRule.BLAND_RULE)\
            -> SimplexResult:
        self._load_scenario(A, b, c)

        if not all(is_close_to_zero(b, or_above=True)):
            self._init_aux_problem(A, b, c)
            aux_res = self._get_optimal_from_feasible_solution(rule=rule)
            if aux_res.optimal_score != 0:
                return SimplexResult(ResultCode.INFEASIBLE, None, None)
            self._pivot_to_original_problem(c)

        return self._get_optimal_from_feasible_solution(rule)

    def has_feasible_solution(self, A, b, c) -> bool:
        self._load_scenario(A, b, c)
        if all(is_close_to_zero(b, or_above=True)):
            return True
        else:
            self._init_aux_problem(A, b, c)
            aux_optimal_score = self._get_optimal_from_feasible_solution(
                rule=PickingRule.BLAND_RULE).optimal_score
            if aux_optimal_score == 0:
                return True
            return False
