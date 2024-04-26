# Distributed MPC for PWA Systems Based on Switching ADMM

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/stable-dmpc-pwa/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in example 1 of [Distributed MPC for PWA Systems Based on Switching ADMM](https://arxiv.org/abs/2404.16712) submitted to [IEEE Transactions on Automatic Control](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9).

In this work, we propose a novel approach for distributed MPC for PWA systems. The approach is based on a switching ADMM procedure that is developed to solve the globally formulated non-convex MPC optimal control problem distributively.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{
    mallick2024distributed,
    title={Distributed MPC for PWA Systems Based on Switching ADMM}, 
    author={Samuel Mallick and Azita Dabiri and Bart De Schutter},
    year={2024},
    eprint={2404.16712},
    archivePrefix={arXiv}
}
```

---

## Installation

The code was created with `Python 3.11`. To access it, clone the repository

```bash
git clone https://github.com/SamuelMallick/stable-dmpc-pwa
cd stable-dmpc-pwa
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`env.py`** contains the environment for simulating the system, i.e., stepping the dynamics and generating costs.
- **`model.py`** contains all the relevant numerical definitions for the state space model of the system.
- **`plotting.py`** contains functions for plotting the state trajectories of the system.
- **`terminal_cost_calculations.py`** calculates appropriate terminal costs for the system via semi-definite programming.
- **`generate_ICs.py`** generates the csv file ICS.csv.
- **`ICs.csv`** contains the randomly generated initial conditions used in Distributed MPC for PWA Systems Based on Switching ADMM.
- **`cent_MLD.py`** simulates the system under a centralized mixed logical dynamical MPC controller.
- **`g_admm.py`** simulates the system under the distributed controller proposed in Distributed MPC for PWA Systems Based on Switching ADMM.
- **`plt_traj/compare_trajs.py`** are scripts for generating the images in Distributed MPC for PWA Systems Based on Switching ADMM.
- **`data`** contains data files for the results in Distributed MPC for PWA Systems Based on Switching ADMM.
- **`MATLAB`** contains auxilarry Matlab scripts for the calculation of terminal sets for the system.

```

## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/stable-dmpc-pwa/blob/main/LICENSE) file included with this repository.

---

## Author

[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2023 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “stable-dmpc-pwa” (Distributed MPC for PWA Systems Based on Switching ADMM) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.
