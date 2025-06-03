# Code repository for "Enhancing Topology Optimization through Machine Learning Models"
### Master's Thesis - 2025




## Dataset

Inside the `dataset-creation` directory, several scripts for multiple topology optimization problems can be found, such as cantilever beams, MBB beams, L-brackets, and other variations with holes inside the domain. For the current work, the cantilever beam with diagonal loads (`topopt_cholmod_cantilever_beam_diagonal_load.py`) was parameterized to generate the dataset. 

These scripts are based on the Python framework built by Anderson et al. (2011), which was later adapted by [Niels Aage and Villads Egede](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python) to optimize its Finite Element Method solver by using Cholesky factorization (CHOLMOD solver).

This topology optimization dataset referenced in this work is available on [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15026508.svg)](https://doi.org/10.5281/zenodo.15026508).


This cantilever beam dataset has 9 240 information groups, totaling 119 812 data instances. It was parameterized by changing:
- Volume fraction;
- Load relative position on the right edge;
- Horizontal Load Magnitude;
- Vertical Load Magnitude.

## Prerequisites

The following versions were used on the main version of this work:

- Python 3.12
- NumPy 2.2.2
- SciPy 1.15.1
- PyTorch 2.7.0
- Matplotlib 3.10.0
- CVXOPT 1.3.2
- H5py 3.13.0
- tqdm 4.67.1

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

## License

This project is licensed under the following licenses:

🖥️ Code: MIT License - see [LICENSE](./LICENSE) for details.  
📊 Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) via [Zenodo](https://zenodo.org/records/15243092).


## Contact

- **Author**: Ricardo A. O. Bastos
- **Institutional Email**: [r.bastos@ua.pt](mailto:r.bastos@ua.pt)
- **LinkedIn**: [Ricardo Bastos](https://www.linkedin.com/in/ricardo-bastos-rantonio/)

## Acknowledgments

- **Thesis advisor**: Prof. Dr. João Oliveira, University of Aveiro
- **Thesis co-advisor**: Dr. Mafalda Gonçalves, INEGI
---

*This repository is part of a Master's Thesis research project on "Enhancing Topology Optimization through Machine Learning Models".*
