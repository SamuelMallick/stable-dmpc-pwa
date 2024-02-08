from typing import Any

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag

from model import get_cent_system, get_cost_matrices, get_IC

np.random.seed(1)


class Network(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """Netork of dynamically coupled PWA systems."""

    Q_x_l, Q_u_l = get_cost_matrices()

    n = 3  # num sub-systems

    x0 = get_IC()

    def __init__(self) -> None:
        self.Q_x = block_diag(*[self.Q_x_l] * self.n)
        self.Q_u = block_diag(*[self.Q_u_l] * self.n)
        self.sys = get_cent_system()

        super().__init__()

    def set_IC(self, IC):
        self.x0 = IC

    def reset(
        self,
        *,
        seed: int = None,
        options: dict[str, Any] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        self.x = self.x0
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes the stage cost."""

        return state.T @ self.Q_x @ state + action.T @ self.Q_u @ action

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the system."""

        action = action.full()
        r = self.get_stage_cost(self.x, action)

        x_new = None
        for i in range(len(self.sys["S"])):
            if all(
                self.sys["S"][i] @ self.x + self.sys["R"][i] @ action
                <= self.sys["T"][i]
            ):
                x_new = (
                    self.sys["A"][i] @ self.x
                    + self.sys["B"][i] @ action
                    + self.sys["c"][i]
                )
        if x_new is None:
            raise RuntimeError("No PWA region found for system.")

        x_new = x_new
        self.x = x_new
        return x_new, r, False, False, {}


def random_vector_with_infinity_norm(max_norm):
    # Generate a random vector with inf norm less than max_norm
    random_vector = np.random.uniform(-max_norm, max_norm, size=(6, 1))
    return random_vector
