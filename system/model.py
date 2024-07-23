import numpy as np
from dmpcpwa.utils.pwa_models import cent_from_dist

class Model:
    """State model class for the multi-agent system."""
    @property
    def n(self) -> int:
        """Returns the number of subsystems."""
        return 3

    @property
    def terminal_controllers(self) -> list[np.ndarray]:
        """Returns the terminal controllers."""
        return self.K
    
    @property
    def terminal_costs(self) -> list[np.ndarray]:
        """Returns the terminal cost matrix."""
        return self.P

    # PWA dynamics
    A1 = np.array([[0.6324, 0.2785], [0.0975, 0.5469]])
    A2 = np.array([[0.6555, 0.7060], [0.1712, 0.0318]])
    A3 = np.array([[0.6324, 0.2785], [0.0975, 0.5469]])
    A4 = np.array([[0.6555, 0.7060], [0.1712, 0.0318]])

    B1 = np.array([[1], [0]])
    B2 = np.array([[1], [0]])
    B3 = np.array([[1], [0]])
    B4 = np.array([[1], [0]])

    c1 = np.zeros((2, 1))
    c2 = np.zeros((2, 1))
    c3 = np.zeros((2, 1))
    c4 = np.zeros((2, 1))

    A = [A1, A2, A3, A4]
    B = [B1, B2, B3, B4]
    c = [c1, c2, c3, c4]

    D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    E = np.array([[20], [20], [20], [20]])

    u_lim = 3
    F = np.array([[1], [-1]])
    G = np.array([[u_lim], [u_lim]])

    S1 = np.array([[1, -1], [-1, -1]])
    R1 = np.zeros((2, 1))
    T1 = np.zeros((2, 1))

    S2 = np.array([[-1, 1], [-1, -1]])
    R2 = np.zeros((2, 1))
    T2 = np.zeros((2, 1))

    S3 = np.array([[-1, 1], [1, 1]])
    R3 = np.zeros((2, 1))
    T3 = np.zeros((2, 1))

    S4 = np.array([[1, -1], [1, 1]])
    R4 = np.zeros((2, 1))
    T4 = np.zeros((2, 1))

    S = [S1, S2, S3, S4]
    R = [R1, R2, R3, R4]
    T = [T1, T2, T3, T4]

    # cost matrices for local systems
    Q_x = np.array([[2, 0], [0, 2]])
    Q_u = 0.2 * np.array([[1]]) 

    # coupling description
    adj = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]])

    ICs = [
        np.array([[-11, -18, 2, -19, 15, 19]]).T,
        np.array([[-17,-18,18,-19,-18,19]]).T,
        np.array([[17, 18, 18, 15, -18, 15]]).T,
        np.array([[0, 19, 19, 0, -19, 0]]).T,
        np.array([[0, -19, -18, -15, 18, -15]]).T,
        np.array([[-10, 18, 10, 18, 18, -10]]).T,
        np.array([[18, 10, -10, -18, 10, -18]]).T,
        np.array([[-17, -18, 19, 0, 18, -15]]).T,
        np.array([[0, -19, 18, -19, 10, -18]]).T,
        np.array([[-18, 15, 19, 0, 10, 18]]).T,
        np.array([[-12, -16, 4, -15, 14, 16]]).T
    ]

    # terminal controllers
    K = [
        np.array([[-0.0544, -0.1398]]),
        np.array([[-0.1544, -0.0295]]),
        np.array([[-0.0544, -0.1398]]),
        np.array([[-0.1544, -0.0295]]),
    ]

    # terminal set
    g = 47
    v = np.array([[7.8514, 8.1971], [8.1957, -7.8503]])
    A_t = np.vstack([v, -v])
    b_t = g*np.ones((4, 1))

    def __init__(self, strong_coupling: bool = False) -> None:
        """Initialize the model.
        
        Parameters
        ----------
        strong_coupling : bool, optional
            If true, the inter-agent coupling is 1.6e-1 I, otherwise it is 2e-3 I."""        
        if strong_coupling:
            self.A_c = 1.6e-1 * np.eye(2)
            self.P = [
                np.array([[40.98181315, 28.2907376], [28.2907376, 43.73493886]]),
                np.array([[32.07159554, 20.89652306], [20.89652306, 35.91201348]]),
                np.array([[31.96518077, 20.83443871], [20.83443871, 35.06598081]]),
            ]
        else:
            self.A_c = 2e-3 * np.eye(2)
            self.P = [
                np.array([[3.93926703, 1.26288722], [1.26288722, 4.34681378]]),
                np.array([[3.93905806, 1.26279049], [1.26279049, 4.34682604]]),
                np.array([[3.93885254, 1.26254341], [1.26254341, 4.34670963]]),
            ]

    def get_IC(self, idx: int = 0) -> np.ndarray:
        """Returns one of the predefined the initial conditions.
        
        Parameters
        ----------
        idx : int, optional
            The index of the initial condition to return. If not provided, the first one is returned."""
        if idx >= len(self.ICs):
            raise ValueError(f"Index {idx} is out of bounds for the {len(self.ICs)} initial conditions.")
        return self.ICs[idx]
    
    def get_cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the cost matrices Q_x and Q_u."""
        return self.Q_x, self.Q_u
    
    def get_local_system(self) -> dict:
        """Returns the local system dynamics as a PWA dictionary {S, R, T, A, B, c, D, E, F, G}. 
        No coupling is included in the dynamics description.
        
        Returns
        -------
        dict
            The local system dynamics."""
        return {
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
        """Returns the local system dynamics as PWA dictionaries {S, R, T, A, B, c, D, E, F, G}.
        The coupling is incluyded in the dynamics description.
        
        Returns
        -------
        list[dict]
            The local system dynamics with coupling."""
        system = self.get_local_system()
        systems = []  # list of systems, 1 for each agent
        systems.append(system.copy())
        Ac_i = [self.A_c]
        systems[0]["Ac"] = []
        for _ in range(len(system["S"])):
            systems[0]["Ac"] = systems[0]["Ac"] + [Ac_i]

        systems.append(system.copy())
        Ac_i = [self.A_c, self.A_c]
        systems[1]["Ac"] = []
        for _ in range(len(system["S"])):
            systems[1]["Ac"] = systems[1]["Ac"] + [Ac_i]

        systems.append(system.copy())
        Ac_i = [self.A_c]
        systems[2]["Ac"] = []
        for _ in range(len(system["S"])):
            systems[2]["Ac"] = systems[2]["Ac"] + [Ac_i]

        return systems
    
    def get_cent_system(self) -> dict:
        """Returns the centralized system dynamics as a PWA dictionary {S, R, T, A, B, c, D, E, F, G}."""
        sys_1 = self.get_local_system()
        sys_1["Ac"] = [[self.A_c] for _ in range(4)]
        sys_2 = self.get_local_system()
        sys_2["Ac"] = [[self.A_c, self.A_c] for _ in range(4)]
        sys_3 = self.get_local_system()
        sys_3["Ac"] = [[self.A_c] for _ in range(4)]
        return cent_from_dist([sys_1, sys_2, sys_3], self.adj)
    
    def get_inv_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the invariant set A x <= b as (A, b).
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The invariant set A x <= b."""
        return self.A_t, self.b_t


# HIGH_COUPLING = True


# # hard coded IC's
# def get_IC():
#     # return np.array([[-11, -18, 2, -19, 15, 19]]).T
#     # return np.array([[-17,-18,18,-19,-18,19]]).T
#     # return np.array([[17, 18, 18, 15, -18, 15]]).T
#     # return np.array([[0, 19, 19, 0, -19, 0]]).T
#     # return np.array([[0, -19, -18, -15, 18, -15]]).T
#     # return np.array([[-10, 18, 10, 18, 18, -10]]).T
#     # return np.array([[18, 10, -10, -18, 10, -18]]).T
#     # return np.array([[-17, -18, 19, 0, 18, -15]]).T
#     # return np.array([[0, -19, 18, -19, 10, -18]]).T
#     return np.array([[-18, 15, 19, 0, 10, 18]]).T 
#     # return np.array([[-12, -16, 4, -15, 14, 16]]).T


# # cost matrices
# Q_x = np.array([[2, 0], [0, 2]])
# Q_u = 0.2 * np.array([[1]])





# # PWA dynamics
# A1 = np.array([[0.6324, 0.2785], [0.0975, 0.5469]])
# A2 = np.array([[0.6555, 0.7060], [0.1712, 0.0318]])
# A3 = np.array([[0.6324, 0.2785], [0.0975, 0.5469]])
# A4 = np.array([[0.6555, 0.7060], [0.1712, 0.0318]])

# B1 = np.array([[1], [0]])
# B2 = np.array([[1], [0]])
# B3 = np.array([[1], [0]])
# B4 = np.array([[1], [0]])

# c1 = np.zeros((2, 1))
# c2 = np.zeros((2, 1))
# c3 = np.zeros((2, 1))
# c4 = np.zeros((2, 1))

# A = [A1, A2, A3, A4]
# B = [B1, B2, B3, B4]
# c = [c1, c2, c3, c4]

# D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
# E = np.array([[20], [20], [20], [20]])

# u_lim = 3
# F = np.array([[1], [-1]])
# G = np.array([[u_lim], [u_lim]])

# S1 = np.array([[1, -1], [-1, -1]])
# R1 = np.zeros((2, 1))
# T1 = np.zeros((2, 1))

# S2 = np.array([[-1, 1], [-1, -1]])
# R2 = np.zeros((2, 1))
# T2 = np.zeros((2, 1))

# S3 = np.array([[-1, 1], [1, 1]])
# R3 = np.zeros((2, 1))
# T3 = np.zeros((2, 1))

# S4 = np.array([[1, -1], [1, 1]])
# R4 = np.zeros((2, 1))
# T4 = np.zeros((2, 1))

# S = [S1, S2, S3, S4]
# R = [R1, R2, R3, R4]
# T = [T1, T2, T3, T4]


# def get_local_system():
#     return {
#         "S": S,
#         "R": R,
#         "T": T,
#         "A": A,
#         "B": B,
#         "c": c,
#         "D": D,
#         "E": E,
#         "F": F,
#         "G": G,
#     }


# # coupling description
# Adj = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
# if HIGH_COUPLING:
#     A_c = 1.6e-1 * np.eye(2)
# else:
#     A_c = 2e-3 * np.eye(2)


# def get_adj():
#     return Adj


# def get_A_c():
#     return A_c


# def get_local_coupled_systems():
#     # manually construct system descriptions with coupling added
#     system = get_local_system()
#     systems = []  # list of systems, 1 for each agent
#     systems.append(system.copy())
#     Ac_i = [A_c]
#     systems[0]["Ac"] = []
#     for i in range(len(system["S"])):
#         systems[0]["Ac"] = systems[0]["Ac"] + [Ac_i]

#     systems.append(system.copy())
#     Ac_i = [A_c, A_c]
#     systems[1]["Ac"] = []
#     for i in range(len(system["S"])):
#         systems[1]["Ac"] = systems[1]["Ac"] + [Ac_i]

#     systems.append(system.copy())
#     Ac_i = [A_c]
#     systems[2]["Ac"] = []
#     for i in range(len(system["S"])):
#         systems[2]["Ac"] = systems[2]["Ac"] + [Ac_i]

#     return systems


# def get_cent_system():
#     sys_1 = get_local_system()
#     sys_1["Ac"] = [[A_c] for i in range(4)]
#     sys_2 = get_local_system()
#     sys_2["Ac"] = [[A_c, A_c] for i in range(4)]
#     sys_3 = get_local_system()
#     sys_3["Ac"] = [[A_c] for i in range(4)]

#     return cent_from_dist([sys_1, sys_2, sys_3], Adj)


# # linear terminal controllers
# K = [
#     np.array([[-0.0544, -0.1398]]),
#     np.array([[-0.1544, -0.0295]]),
#     np.array([[-0.0544, -0.1398]]),
#     np.array([[-0.1544, -0.0295]]),
# ]


# def get_terminal_K():
#     return K


# # invariant set
# def get_inv_set_vertices():
#     return np.array(
#         [
#             [5.8850, 0.1257],
#             [-0.1265, 5.8549],
#             [-5.8550, -0.1257],
#             [0.1265, -5.8549],
#             [5.8850, 0.1257],
#         ]
#     )
# g = 47
# P = np.array([[7.8514, 8.1971], [8.1957, -7.8503]])
# A_t = np.vstack([P, -P])
# b_t = g*np.ones((4, 1))

# def get_inv_set():
#     return A_t, b_t


# def get_warm_start():
#     raise RuntimeError("get_warm_start not implemented for model")


# # terminal costs
# if HIGH_COUPLING:
#     P = [
#         np.array([[40.98181315, 28.2907376], [28.2907376, 43.73493886]]),
#         np.array([[32.07159554, 20.89652306], [20.89652306, 35.91201348]]),
#         np.array([[31.96518077, 20.83443871], [20.83443871, 35.06598081]]),
#     ]
# else:
#     P = [
#         np.array([[3.93926703, 1.26288722], [1.26288722, 4.34681378]]),
#         np.array([[3.93905806, 1.26279049], [1.26279049, 4.34682604]]),
#         np.array([[3.93885254, 1.26254341], [1.26254341, 4.34670963]]),
#     ]


# def get_terminal_costs():
#     return P
