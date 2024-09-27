# Distributed Model Predictive Control for Piecewise Affine Systems Based on Switching ADMM

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/stable-dmpc-pwa/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in [Distributed Model Predictive Control for Piecewise Affine Systems Based on Switching ADMM](https://arxiv.org/abs/2404.16712) submitted to [IEEE Transactions on Automatic Control](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9).

In this work, we propose a novel approach for distributed MPC for PWA systems. The approach is based on a switching ADMM procedure that is developed to solve the globally formulated non-convex MPC optimal control problem distributively.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{mallick2024distributed,
  title={Distributed MPC for PWA Systems Based on Switching ADMM},
  author={Mallick, Samuel and Dabiri, Azita and De Schutter, Bart},
  journal={arXiv preprint arXiv:2404.16712},
  year={2024}
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

- **`run_switching_admm_sim.py`** simulates the system under the proposed novel controller.
- **`run_centralized_MLD_sim.py`** simulates the system under a centralized controller.
- **`agents`** contains classes that define control agents.
- **`mpcs`** contains classes that define the MPC controllers.
- **`system`** contains the system dynamics and the environment that simulates them.
- **`utils`** contains scripts for plotting and anayzing data.
- **`random_systems`** contains scripts for generating and comparing the controller's performances on randomized systems.
- **`data`** contains data files for the results in Distributed Model Predictive Control for Piecewise Affine Systems Based on Switching ADMM.
- **`MATLAB`** contains auxilarry Matlab scripts for the calculation of terminal sets for the system.

```

## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/stable-dmpc-pwa/blob/main/LICENSE) file included with this repository.

---

## Author

[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2024 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “stable-dmpc-pwa” (Distributed Model Predictive Control for Piecewise Affine Systems Based on Switching ADMM) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.
