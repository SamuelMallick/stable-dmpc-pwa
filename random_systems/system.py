import numpy as np
from dmpcpwa.utils.pwa_models import cent_from_dist


class RandomNetwork:

    adj = np.array([[0, 1], [1, 0]])

    # four quadrant PWA nature
    S1 = np.array([[-1, 0], [0, -1]])
    R1 = np.zeros((2, 1))
    T1 = np.zeros((2, 1))

    S2 = np.array([[-1, 0], [0, 1]])
    R2 = np.zeros((2, 1))
    T2 = np.zeros((2, 1))

    S3 = np.array([[1, 0], [0, -1]])
    R3 = np.zeros((2, 1))
    T3 = np.zeros((2, 1))

    S4 = np.array([[1, 0], [0, 1]])
    R4 = np.zeros((2, 1))
    T4 = np.zeros((2, 1))

    S = [S1, S2, S3, S4]
    R = [R1, R2, R3, R4]
    T = [T1, T2, T3, T4]

    x_lim = 10
    D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    E = np.array([[x_lim], [x_lim], [x_lim], [x_lim]])

    u_lim = 1
    F = np.array([[1], [-1]])
    G = np.array([[u_lim], [u_lim]])

    def __init__(self, np_random: np.random.Generator):
        self.np_random = np_random
        self.reset()

    def reset(self):
        self.A = []
        self.B = []
        self.c = []
        for i in range(4):
            self.A.append(self.np_random.uniform(-1.5, 1.5, size=(2, 2)))
            self.B.append(self.np_random.uniform(-1.5, 1.5, size=(2, 1)))
            self.c.append(self.np_random.uniform(-1.5, 1.5, size=(2, 1)))

        self.A_c = self.np_random.uniform(-0.1, 0.1, size=(2, 2))

        self.system = {
            "S": self.S,
            "R": self.R,
            "T": self.T,
            "A": self.A,
            "B": self.B,
            "c": self.c,
            "D": self.D,
            "E": self.E,
            "F": self.F,
            "G": self.G,
        }

    def get_local_coupled_systems(self) -> list[dict]:
        """Returns the local system dynamics as PWA dictionaries {S, R, T, A, B, c, Ac, D, E, F, G}.
        The coupling is incluyded in the dynamics description.

        Returns
        -------
        list[dict]
            The local system dynamics with coupling."""
        systems = []  # list of systems, 1 for each agent
        systems.append(self.system.copy())
        Ac_i = [self.A_c]
        systems[0]["Ac"] = []
        for _ in range(len(self.system["S"])):
            systems[0]["Ac"] = systems[0]["Ac"] + [Ac_i]

        systems.append(self.system.copy())
        Ac_i = [self.A_c]
        systems[1]["Ac"] = []
        for _ in range(len(self.system["S"])):
            systems[1]["Ac"] = systems[1]["Ac"] + [Ac_i]

        return systems

    def get_cent_system(self) -> dict:
        """Returns the centralized system dynamics as a PWA dictionary {S, R, T, A, B, c, D, E, F, G}."""
        sys_1 = self.system.copy()
        sys_1["Ac"] = [[self.A_c] for _ in range(4)]
        sys_2 = self.system.copy()
        sys_2["Ac"] = [[self.A_c] for _ in range(4)]

        return cent_from_dist([sys_1, sys_2], self.adj)
