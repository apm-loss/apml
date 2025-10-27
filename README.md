# APML: Adaptive Probabilistic Matching Loss for Robust 3D Point Cloud Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
[![arXiv](https://img.shields.io/badge/arXiv-2509.08104-b31b1b)](https://arxiv.org/abs/2509.08104)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blueviolet)](https://neurips.cc/virtual/2025/poster/118183)

> **News**: Accepted at NeurIPS 2025 (Poster). The repository includes both a pure-PyTorch APML and a CUDA-accelerated APML (sparse) implementation.
> The repository includes both a **pure PyTorch APML** and a **CUDA accelerated APML (sparse)** implementation.

![cover](docs/images/apml_comparison.png)

## Abstract

Loss design is central to point-set prediction. Chamfer-style losses rely on nearest neighbors and often create many-to-one matches that degrade structure in dense regions. Earth Mover Distance promotes one-to-one alignment but with high computational cost. APML introduces a differentiable, assignment-aware objective based on temperature-scaled similarities and Sinkhorn normalization, with an analytic temperature that enforces a minimum assignment probability. In practice, APML approaches the structural fidelity of EMD with stable gradients and competitive runtime.

---
## Paper and artifacts

- NeurIPS 2025 poster page: https://neurips.cc/virtual/2025/poster/118183  
- Preprint: https://arxiv.org/abs/2509.08104  
- Pretrained models: https://drive.google.com/file/d/1dSuBUrZQzAWsxYH6j_EP-EAhgsRfvP5t/view?usp=sharing

---

## Key ideas

- **Assignment awareness without cubic EMD**: probabilistic matching via temperature-scaled similarities + Sinkhorn normalization.
- **Analytic temperature** from local distance gaps → no hyperparameter tuning for `τ`.
- **Stable, differentiable, and efficient**; improves coverage in sparse regions and reduces point congestion.

## Key Features

- Fully differentiable loss function
- Analytical temperature computation for stable assignments
- Compatible with architectures such as PoinTr, PCN, FoldingNet, and CSI2PC
- Improves spatial distribution of predicted points, particularly in sparse regions
- Achieves faster convergence and competitive or superior quantitative results

---

## Installation

Works with Python 3.8+ and PyTorch 1.12+.
For the CUDA path, a working NVCC and a PyTorch build with CUDA are required.

```bash
# optional: create environment
conda create -n apml python=3.10 -y
conda activate apml

# install PyTorch first (choose build from pytorch.org matching your CUDA)
pip install torch torchvision torchaudio

# repo dependencies
pip install -r requirements.txt
```

### CUDA-accelerated APML build
The CUDA version is a PyTorch extension that exposes the same interface as the Python implementation.

```bash
# from repo root
cd src/apml_cuda
python setup.py install        # or: pip install -v .
cd ../../
```

If the extension builds correctly, importing apml_sparse enables the accelerated path. Ensure CUDA_HOME points to your CUDA toolkit and that you have a C++17 compiler available.

## Quick start

### Reference APML

```python
import torch
from apml.loss.apml_loss import APML

B, N, M, D = 4, 2048, 2048, 3
pred = torch.randn(B, N, D, device="cuda")
gt   = torch.randn(B, M, D, device="cuda")

criterion = APML(p_min=0.8)
loss = criterion(pred, gt)
loss.backward()
```

### CUDA APML
```python
import torch
try:
    from apml.loss.apml_sparse_loss import APMLSparse
    criterion = APMLSparse(p_min=0.8, threshold=1e-10)
except Exception as e:
    from apml.loss.apml_loss import APML
    print(f"Falling back to reference APML: {e}")
    criterion = APML(p_min=0.8)

B, N, M, D = 4, 2048, 2048, 3
pred = torch.randn(B, N, D, device="cuda")
gt   = torch.randn(B, M, D, device="cuda")
loss = criterion(pred, gt)
loss.backward()
```

### Integration

This repository provides only the implementation of the loss function. To use APML in your models, you should integrate it into an existing 3D point cloud architecture. We recommend starting with one of the following repositories:

- [PoinTr](https://github.com/yuxumin/PoinTr)
- [CSI2PointCloud](https://github.com/arritmic/csi2pointcloud)

In each case, replace the existing loss function with `apml_loss.py` provided here and ensure the input/output shapes match the expected format.

---
## Repository structure

```text
apml/
├─ src/
│  ├─ apml_cuda/                       # CUDA extension
│  │  ├─ setup.py                      # builds the torch extension apml_sparse
│  │  ├─ apml_sparse.cpp               # C++ binding
│  │  ├─ apml_sparse_kernel.cu         # CUDA kernels
│  │  └─ __init__.py
│  └─ loss/
│     ├─ apml_loss.py                  # reference APML in PyTorch
│     └─ apml_sparse_loss.py           # autograd wrapper for CUDA op
├─ scripts/
│  ├─ demo_completion.py               # example for completion. Coming soon!
│  └─ demo_generation.py               # example for generation. Coming soon!
├─ models/
│  └─ README.md                        # notes or fetch scripts for checkpoints
├─ docs/
│  ├─ images/
│  │  └─ apml_comparison.png
│  └─ apml_notes.md
├─ tests/
│  ├─ test_apml_reference.py          # Coming soon!
│  └─ test_apml_sparse.py             # Coming soon!
├─ requirements.txt
├─ LICENSE
├─ CITATION.cff
└─ README.md
```


## Citation

If you use APML in academic work, please cite both the NeurIPS entry and the preprint (until the official proceedings bib is available):

### NeurIPS (provisional)

```
@inproceedings{sharifipour2025apml,
  title     = {APML: Adaptive Probabilistic Matching Loss for Robust 3D Point Cloud Reconstruction},
  author    = {Sharifipour, Sasan and {\'A}lvarez Casado, Constantino and Sabokrou, Mohammad and Bordallo L{\'o}pez, Miguel},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {Poster},
  url       = {[https://neurips.cc/virtual/2025/poster/118183](https://neurips.cc/virtual/2025/poster/118183)}
}
```

### arXiv
```
@article{sharifipour2025apml_arxiv,
  title   = {APML: Adaptive Probabilistic Matching Loss for Robust 3D Point Cloud Reconstruction},
  author  = {Sharifipour, Sasan and {\'A}lvarez Casado, Constantino and Sabokrou, Mohammad and Bordallo L{\'o}pez, Miguel},
  journal = {arXiv preprint arXiv:2509.08104},
  year    = {2025},
  url     = {[https://arxiv.org/abs/2509.08104](https://arxiv.org/abs/2509.08104)}
}
```

## License
This project is released under the MIT License. See LICENSE.

## Acknowledgments
This research was supported by the Business Finland 6G-WISECOM project (Grant 3630/31/2024), the University of Oulu, and the Research Council of Finland (formerly Academy of Finland) through the 6G Flagship Programme (Grant 346208). Sasan Sharifipour acknowledges the funding from the Finnish Doctoral Program Network on Artificial Intelligence, AI-DOC (decision number VN/3137/2024-OKM-6), supported by Finland’s Ministry of Education and Culture and hosted by the Finnish Center for Artificial Intelligence (FCAI). The authors declare no conflict of interest. 

Computational resources were provided by CSC – IT Center for Science, Finland, with model training performed on the Mahti and Puhti supercomputers.

## Maintainers

* Sasan Sharifipour
* Constantino Álvarez Casado
* Miguel Bordallo López

For questions or issues, please open a GitHub Issue with environment details, PyTorch/CUDA versions, and reproduction steps.
