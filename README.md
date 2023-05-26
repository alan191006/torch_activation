# PyTorch Activation Collection

A collection of new, un-implemented activation functions for the PyTorch library. This project is designed for ease of use during experimentation with different activation functions (or simply for fun!). 


## Installation

```bash
$ pip install torch-activation
```

## Usage

To use the activation functions, simply import from `torch_activation`:

```python
from torch_activation import ShiLU

m = ShiLU(inplace=True)
x = torch.rand(1, 2, 2, 3)
m(x)
```


## Available Functions

| Activation Function   | Equation |
|-----------------------|----------------|
| ShiLU [[1]](#1)       |                |
| DELU [[1]](#1)        |                |
| CReLU [[2]](#2)       |                |
| GCU [[3]](#3)         |                |
| CosLU [[1]](#1)       |                |
| CoLU [[4]](#4)        |                |
| ReLUN [[1]](#1)       |                |
| SquaredReLU [[5]](#5) |                |
| ScaledSoftSign [[1]](#1) |              |
| ReGLU [[6]](#6)       |                |
| GeGLU [[6]](#6)       |                |
| SwiGLU [[6]](#6)      |                |
| SeGLU                 |                |
| LinComb [[7]](#7)     |                |
| NormLinComb [[7]](#7) |                |
| SinLU                 |                |
| DReLUs                |                |
  
## Future features
* Activations:
  * DReLUs
  * ...
* Layers:
  * Depth-wise Convolution
  * ...

## References
<a id="1">[1]</a>
Pishchik, E. (2023). Trainable Activations for Image Classification. Preprints.org, 2023010463. DOI: 10.20944/preprints202301.0463.v1.

<a id="2">[2]</a>
Shang, W., Sohn, K., Almeida, D., Lee, H. (2016). Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units. arXiv:1603.05201v2 (cs).

<a id="3">[3]</a>
Noel, M. M., Arunkumar, L., Trivedi, A., Dutta, P. (2023). Growing Cosine Unit: A Novel Oscillatory Activation Function That Can Speedup Training and Reduce Parameters in Convolutional Neural Networks. arXiv:2108.12943v3 (cs).

<a id="4">[4]</a>
Vagerwal, A. (2021). Deeper Learning with CoLU Activation. arXiv:2112.12078v1 (cs).

<a id="5">[5]</a>
So, D. R., Mańke, W., Liu, H., Dai, Z., Shazeer, N., Le, Q. V. (2022). Primer: Searching for Efficient Transformers for Language Modeling. arXiv:2109.08668v2 (cs)

<a id="6">[6]</a>
Noam, S. (2020). GLU Variants Improve Transformer. arXiv:2002.05202v1 (cs)

<a id="7">[7]</a>
Pishchik, E. (2023). Trainable Activations for Image Classification. Preprints.org, 2023010463. DOI: 10.20944/preprints202301.0463.v1