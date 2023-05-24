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

- ShiLU [[1]](#1)

  ![ShiLU Activation](https://github.com/alan191006/torch_activation/blob/63d8db5d4a2ef19e382fc8175bf47b0a5173df3e/images/activation_images/ShiLU.png "ShiLU")

- DELU [[1]](#1)

  ![DELU Activation](https://github.com/alan191006/torch_activation/blob/63d8db5d4a2ef19e382fc8175bf47b0a5173df3e/images/activation_images/DELU.png "DELU")

- CReLU [[2]](#2)

- GCU [[3]](#3)

  ![GCU Activation](https://github.com/alan191006/torch_activation/blob/63d8db5d4a2ef19e382fc8175bf47b0a5173df3e/images/activation_images/GCU.png "GCU")

- CosLU [[1]](#1)

  ![CosLU Activation](https://github.com/alan191006/torch_activation/blob/63d8db5d4a2ef19e382fc8175bf47b0a5173df3e/images/activation_images/CosLU.png "CosLU")

- CoLU [[4]](#4)

  ![CoLU Activation](https://github.com/alan191006/torch_activation/blob/63d8db5d4a2ef19e382fc8175bf47b0a5173df3e/images/activation_images/CoLU.png "CoLU")

- ReLUN [[1]](#1)

  ![ReLUN Activation](https://github.com/alan191006/torch_activation/blob/63d8db5d4a2ef19e382fc8175bf47b0a5173df3e/images/activation_images/ReLUN.png "ReLUN")


- SquaredReLU [[5]](#5)

  ![SquaredReLU Activation](https://github.com/alan191006/torch_activation/blob/63d8db5d4a2ef19e382fc8175bf47b0a5173df3e/images/activation_images/SquaredReLU.png "SquaredReLU")

- ScaledSoftSign [[1]](#1)

  ![ScaledSoftSign Activation](https://github.com/alan191006/torch_activation/blob/63d8db5d4a2ef19e382fc8175bf47b0a5173df3e/images/activation_images/ScaledSoftSign.png "ScaledSoftSign")

- ReGLU [[6]](#6)

- GeGLU [[6]](#6)

- SwiGLU [[6]](#6)

- SeGLU

- LinComb [[7]](#7)

- NormLinComb [[7]](#7)
  
## Future features
* Activations:
  * SinLU
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