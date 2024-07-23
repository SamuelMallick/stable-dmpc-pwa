import numpy as np
from csnlp.wrappers import Mpc
from dmpcpwa.agents.g_admm_coordinator import PwaAgent


# override PWA agent class to add in a terminal controller that switches with regions and uses final region in sequence
class PwaAgentTerminal(PwaAgent):
    """A PWA agent that switches to terminal controllers when in terminal sets."""

    def __init__(
        self,
        mpc: Mpc,
        fixed_parameters: dict,
        pwa_system: dict,
        use_terminal_sequence: bool = False,
    ) -> None:
        super().__init__(mpc, fixed_parameters, pwa_system, use_terminal_sequence=True)

    K: list[np.ndarray] = []  # terminal controllers

    def set_K(self, K):
        self.K = K

    def set_sequence(self, s: list[int]):
        if len(self.K) == 0:
            raise RuntimeError("Linear controller must be set before sequence is set.")
        for i in range(len(s)):
            self.fixed_parameters[f"K_{i}"] = self.K[s[i]]
        return super().set_sequence(s)
