# :zap: PyTorch Activations :zap:

A collection of activation functions for the PyTorch library. This project is designed for ease of use during experimentation with different activation functions (or simply for fun :wink:). 


## Installation

```bash
$ pip install torch-activation
```

## Usage

To use the activation functions, simply import from `torch_activation`:

```python
from torch_activation import ShiLU

m = ShiLU(inplace=True)
x = torch.rand(16, 3, 384, 384)
m(x)
```

List of available functions below.
## Available Functions

| Activation Functions   | Equations |
|-|-|
| **ReLU Variations** ||
| ShiLU [[1]](#1) | <img src="https://render.githubusercontent.com/render/math?math=\alpha \cdot \text{ReLU}(x) + \beta ">|
| ReLUN [[1]](#1) | <img src="https://render.githubusercontent.com/render/math?math=\min(\text{ReLU}(x), n) ">|
| CReLU [[2]](#2) | <img src="https://render.githubusercontent.com/render/math?math=\text{ReLU}(x) \oplus \text{ReLU}(-x) ">|
| SquaredReLU [[5]](#5) | <img src="https://render.githubusercontent.com/render/math?math=\text{ReLU}(x)^2 ">|
| StarReLU [[8]](#8) | <img src="https://render.githubusercontent.com/render/math?math=s \cdot \text{ReLU}(x)^2 + b">|
| **GLU Variations** ||
| ReGLU [[6]](#6) | <img src="https://render.githubusercontent.com/render/math?math=\text{ReLU} (xW + b) \odot (xV + c) ">|
| GeGLU [[6]](#6) | <img src="https://render.githubusercontent.com/render/math?math=\text{GeLU} (xW + b) \odot (xV + c) ">|
| SwiGLU [[6]](#6) | <img src="https://render.githubusercontent.com/render/math?math=\sigma (xW + b) \odot (xV + c) ">|
| SeGLU | <img src="https://render.githubusercontent.com/render/math?math=\text{SELU} (xW + b) \odot (xV + c) ">|
| **Composite Functions** ||
| DELU [[1]](#1) | <img src="https://render.githubusercontent.com/render/math?math=\begin{cases} \text{SiLU}(x), x \leqslant 0 \\x(n-1), \text{otherwise} \end{cases} ">|
| DReLUs | <img src="https://render.githubusercontent.com/render/math?math=\begin{cases} \alpha (e ^ x -1), x \leqslant 0 \\x, \text{otherwise} \end{cases} ">|
| **Trigonometry Based** ||
| GCU [[3]](#3) | <img src="https://render.githubusercontent.com/render/math?math=x \cdot \cos(x) ">|
| CosLU [[1]](#1) | <img src="https://render.githubusercontent.com/render/math?math=(x + \alpha \cdot \cos(\beta x)) \cdot \sigma(x) ">|
| SinLU | <img src="https://render.githubusercontent.com/render/math?math=(x + \alpha \cdot \sin (\beta x)) \cdot \sigma (x) ">|
| **Others** ||
| ScaledSoftSign [[1]](#1) | <img src="https://render.githubusercontent.com/render/math?math=\frac{\alpha \cdot x}{\beta + \|x\|} ">|
| CoLU [[4]](#4) | <img src="https://render.githubusercontent.com/render/math?math=\frac{x}{1-x \cdot e^{-(x + e^x)}} ">|
| **Linear Combination** ||
| LinComb [[7]](#7) | <img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1}^{n} w_i \cdot F_i(x) ">|
| NormLinComb [[7]](#7) | <img src="https://render.githubusercontent.com/render/math?math=\frac{\sum_{i=1}^{n} w_i \cdot F_i(x)}{\|\|W\|\|} ">|


## Contact

Alan Huynh - [LinkedIn](https://www.linkedin.com/in/alan-huynh-64b357194/) - hdmquan@outlook.com

Project Link: [https://github.com/alan191006/torch_activation](https://github.com/alan191006/torch_activation)


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

<a id="8">[8]</a>
Weihao, Y., et al (2022). MetaFormer Baselines for Vision. arXiv:2210.13452v2 (cs)

[Back to top](#Installation)