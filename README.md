# Distributed MPC for PWA Systems Based on Switching ADMM

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/stable-dmpc-pwa/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in [Distributed MPC for PWA Systems Based on Switching ADMM](PUT LINK) submitted to [PUT JOURNAL](PUT JOURNAL LINK).

In this work, we propose a novel approach for distributed MPC for PWA systems. The approach is based on a switching ADMM procedure that is developed to solve the globally formulated non-convex MPC optimal control problem distributively.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{
}
```

---

## Installation

The code was created with `Python 3.9`. To access it, clone the repository

```bash
git clone https://github.com/SamuelMallick/hybrid-vehicle-platoon
cd hybrid-vehicle-platoon
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`ACC_env.py`** contains the environment for simulating the platoon. This updates the state of the platoon according to the nonlinear hybrid model, and generates the cost penalties for given states.
- **`ACC_model.py`** contains all functions and data structures related to the modelling of the vehicles.
- **`bash_scripts`** contains contains bash scripts for automated running of tests.
- **`data`** contains '.pkl' files for data used in A Comparison Benchmark for Distributed Hybrid MPC Control Methods: Distributed Vehicle Platooning.
- **`results_analysis`** contains scripts for generating the images and tables used in A Comparison Benchmark for Distributed Hybrid MPC Control Methods: Distributed Vehicle Platooning.
- **`fleet_{cent_mld, decent_mld, seq_mld, event_based, naive_admm}.py`** launch simulations for the five controllers used in A Comparison Benchmark for Distributed Hybrid MPC Control Methods: Distributed Vehicle Platooning.

```

## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/hybrid-vehicle-platoon/blob/main/LICENSE) file included with this repository.

---

## Author

[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2023 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “hybrid-vehicle-platoon” (A Comparison Benchmark for Distributed Hybrid MPC Control Methods: Distributed Vehicle Platooning) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.
