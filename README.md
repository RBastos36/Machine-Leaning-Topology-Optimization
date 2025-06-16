# Enhancing Topology Optimization through Machine Learning Models
### Master's Thesis – 2025

This repository contains the code for the machine learning training and dataset generation used in the research project “Enhancing Topology Optimization through Machine Learning Models,” as part of a Master's Thesis at the University of Aveiro. The project explores how deep learning architectures (notably the U-Net architecture) can predict displacement fields in topology optimization structures, accelerating conventional simulation workflows.

---

## Table of Contents
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Machine Learning Architecture](#machine-learning-architecture)
- [Benchmark and Results](#benchmark-and-results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Key Features

- Scripts to generate, preprocess, and use large-scale topology optimization datasets.
- Deep learning models (e.g., U-Net) for predicting mechanical responses from structural layouts.
- Reproducible experiments and benchmarking tools.
- Support for various topology optimization problems: cantilever beams, MBB beams, L-brackets, and more.
- Open-source dataset and code for academic use.

---

## Prerequisites

The following versions were used for the main experiments:
- Python 3.12
- NumPy 2.2.2
- SciPy 1.15.1
- PyTorch 2.7.0
- Matplotlib 3.10.0
- CVXOPT 1.3.2
- H5py 3.13.0
- tqdm 4.67.1

These requirements can all be installed with `pip install`.

---

## Dataset

Inside the `dataset-creation` directory, several scripts for generating datasets on various topology optimization problems can be found (such as cantilever beams, MBB beams, L-brackets, etc.). These scripts utilize the Solid Isotropic Material with Penalization (SIMP) method to solve the topology optimization problems and save the information to the dataset.

These scripts build on the Python framework based on Anderson et al. (2011), adapted by [Niels Aage and Villads Egede](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-in-python) to optimize its Finite Element Method solver by using Cholesky factorization (CHOLMOD solver).

- Dataset DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15243092.svg)](https://doi.org/10.5281/zenodo.15243092)

**Cantilever Beam Dataset**  
- 9 240 parameter groups, 119 812 total instances.
- The dataset was parameterized by changing:
  - Volume fraction
  - Load’s relative position on the right edge
  - Horizontal Load Magnitude
  - Vertical Load Magnitude

---

## Machine Learning Architecture

- **Input:** 5-channel tensor (domain density distribution, loads in _x_- and _y_-directions, constraints in _x_- and _y_-directions)
- **Output:** 2-channel tensor (displacements in _x_- and _y_-directions)
- **Best Model:** U-Net

|![U-Net Architecture](https://github.com/user-attachments/assets/055f8e10-5bfd-4da0-a131-907a5ad09fbd)|
|:--:|
| **Figure 1:** U-Net architecture.|

---

## Benchmark and Results

The U-Net architecture with max pooling achieved a Mean Squared Error (MSE) loss of $2.34 \times 10^{−4}$. Z-score normalization was used to preprocess the data, and the model was trained with a learning rate of 0.001, a batch size of 16, and the Adam optimizer.

|![model_loss](https://github.com/user-attachments/assets/e2a791d2-4c89-41e9-9c48-3b7ce4ae0293)|
|:--:|
| **Figure 2:** Training loss history for the U-Net model.|

The machine learning topology optimization framework achieved a 6.5 times speedup for the 180 by 60 domain size of the cantilever beam problem (compared to the traditional SIMP method). The predictions were visually very similar, as illustrated below in a random test sample (homogeneous domain density of 0.8, relative load position of 0.2, horizontal load of 50, and vertical load of 80).

|![Random_Test_Sample](https://github.com/user-attachments/assets/f5612043-9894-40de-af0a-944ea3c26fa3)|
|:--:|
| **Figure 3:** Random test sample showing (left) ground truth displacements, (middle) predicted displacements, and (right) error between them.|

However, pixel-wise errors were still large enough to deviate the material layout in the topology optimization, as shown below, which makes the solution diverge, or converge towards non-optimal designs.

|![Topology-Optimization_U-Net](https://github.com/user-attachments/assets/0f1117c6-3332-4676-89a4-dbd4bff7f951)|
|:--:|
| **Figure 4:** Evolution of the topology optimization using the U-Net displacement predictor.|

Despite these shortcomings, this displacement predictor can serve as a standalone solver, offering significant acceleration for engineering tasks at the cost of some precision. Also, this machine learning framework can be implemented in non-gradient-based optimizers (such as genetic algorithms or particle swarm optimization). Since these methods do not rely on sensitivity calculations, the pixel-wise errors will not destabilize the topology optimization process, as compliance values will guide the optimization.


---

## Citation

To use this code or the generated dataset in other research, please cite:

```bibtex
@mastersthesis{bastos2025mlto,
  title={Enhancing Topology Optimization through Machine Learning Models},
  author={Bastos, Ricardo António Oliveira},
  year={2025},
  school={[University of Aveiro]},
  type={Master's Thesis}
}

@dataset{bastos2025dataset,
  author       = {Bastos, Ricardo António Oliveira},
  title        = {Topology Optimization Dataset},
  month        = apr,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {2.0},
  doi          = {10.5281/zenodo.15243092},
  url          = {https://doi.org/10.5281/zenodo.15243092},
}
```

---

## License

This project is licensed under:
- 🖥️ Code: MIT License – see [LICENSE](./LICENSE)
- 📊 Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) via [Zenodo](https://zenodo.org/records/15243092)

---

## Contact

- **LinkedIn:** [Ricardo Bastos](https://www.linkedin.com/in/ricardo-bastos-rantonio/)
- **Institutional Email:** [r.bastos@ua.pt](mailto:r.bastos@ua.pt)

---

## Acknowledgments

- **Thesis advisor:** Prof. Dr. João Oliveira, University of Aveiro
- **Co-advisor:** Dr. Mafalda Gonçalves, INEGI

---
*This repository is part of a Master's Thesis research project on "Enhancing Topology Optimization through Machine Learning Models".*
