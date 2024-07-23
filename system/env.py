from typing import Any

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag

from system.model import Model


class Network(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """Netork of dynamically coupled PWA systems."""

    def __init__(self, model: Model) -> None:
        """Initialize the network of PWA systems.

        Parameters
        ----------
        model : Model
            The model of the system.
        """
        self.Q_x_l, self.Q_u_l = model.get_cost_matrices()
        self.n = model.n
        self.Q_x = block_diag(*[self.Q_x_l] * self.n)
        self.Q_u = block_diag(*[self.Q_u_l] * self.n)
        self.sys = model.get_cent_system()
        self.x0 = model.get_IC()
        self.model = model
        super().__init__()

    def set_IC(self, IC: np.ndarray) -> None:
        """Set the initial condition of the system.

        Parameters
        ----------
        IC : np.ndarray
            The initial condition.
        """
        self.x0 = IC

    def reset(
        self,
        *,
        seed: int = None,
        options: dict[str, Any] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        if options is not None and "IC_index" in options:
            self.x = self.model.get_IC(options["IC_index"])
        else:
            self.x = self.x0
        return self.x, {}

    def get_stage_cost(self, state: np.ndarray, action: np.ndarray) -> float:
        """Computes the stage cost.

        Parameters
        ----------
        state : np.ndarray
            The state of the system.
        action : np.ndarray
            The action of the system.

        Returns
        -------
        float
            The stage cost, a quadratic function of the state and action (x^T Q_x x + u^T Q_u u).
        """
        return state.T @ self.Q_x @ state + action.T @ self.Q_u @ action

    def step(
        self, action: cs.DM | np.ndarray
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the system."""
        if isinstance(action, cs.DM):
            action = action.full()
        r = self.get_stage_cost(self.x, action)

        x_new = None
        for i in range(len(self.sys["S"])):  # TODO: get rid of loop
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

    def random_vector_with_infinity_norm(self, max_norm) -> np.ndarray:
        """Generate a random vector with inf norm less than max_norm.

        Parameters
        ----------
        max_norm : float
            The maximum norm of the vector.

        Returns
        -------
        np.ndarray
            The random vector.
        """
        random_vector = self.np_random.uniform(-max_norm, max_norm, size=(6, 1))
        return random_vector
