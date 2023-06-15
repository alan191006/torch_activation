<script src="https://cdn.jsdelivr.net/npm/katex@0.13.16/dist/katex.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.16/dist/katex.min.css" />

<h1 id="pytorch-activations">PyTorch Activations</h1>
<p>A collection of activation functions for the PyTorch library. This project is designed for ease of use during experimentation with different activation functions (or simply for fun :wink:). </p>
<h2 id="installation">Installation</h2>
<pre><code class="lang-bash">$ pip <span class="hljs-keyword">install</span> torch-activation
</code></pre>
<h2 id="usage">Usage</h2>
<p>To use the activation functions, simply import from <code>torch_activation</code>:</p>
<pre><code class="lang-python">from torch_activation import ShiLU

m = ShiLU(inplace=True)
x = torch.rand(<span class="hljs-number">16</span>, <span class="hljs-number">3</span>, <span class="hljs-number">384</span>, <span class="hljs-number">384</span>)
m(x)
</code></pre>
<p>List of available functions below.</p>
<h2 id="available-functions">Available Functions</h2>
<table>
<thead>
<tr>
<th>Activation Functions</th>
<th>Equations</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>ReLU Variations</strong></td>
<td></td>
</tr>
<tr>
<td>ShiLU <a href="#1">[1]</a></td>
<td>$ \alpha \cdot \text{ReLU}(x) + \beta $</td>
</tr>
<tr>
<td>ReLUN <a href="#1">[1]</a></td>
<td>$ \min(\text{ReLU}(x), n) $</td>
</tr>
<tr>
<td>CReLU <a href="#2">[2]</a></td>
<td>$ \text{ReLU}(x) \oplus \text{ReLU}(-x) $</td>
</tr>
<tr>
<td>SquaredReLU <a href="#5">[5]</a></td>
<td>$ \text{ReLU}(x)^2 $</td>
</tr>
<tr>
<td>StarReLU <a href="#8">[8]</a></td>
<td>$ s \cdot \text{ReLU}(x)^2 + b$</td>
</tr>
<tr>
<td><strong>GLU Variations</strong></td>
<td></td>
</tr>
<tr>
<td>ReGLU <a href="#6">[6]</a></td>
<td>$ \text{ReLU} (xW + b) \odot (xV + c) $</td>
</tr>
<tr>
<td>GeGLU <a href="#6">[6]</a></td>
<td>$ \text{GeLU} (xW + b) \odot (xV + c) $</td>
</tr>
<tr>
<td>SwiGLU <a href="#6">[6]</a></td>
<td>$ \sigma (xW + b) \odot (xV + c) $</td>
</tr>
<tr>
<td>SeGLU</td>
<td>$ \text{SELU} (xW + b) \odot (xV + c) $</td>
</tr>
<tr>
<td><strong>Composite Functions</strong></td>
<td></td>
</tr>
<tr>
<td>DELU <a href="#1">[1]</a></td>
<td>$ \begin{cases} \text{SiLU}(x), x \leqslant 0 \x(n-1), \text{otherwise} \end{cases} $</td>
</tr>
<tr>
<td>DReLUs</td>
<td>$ \begin{cases} \alpha (e ^ x -1), x \leqslant 0 \x, \text{otherwise} \end{cases} $</td>
</tr>
<tr>
<td><strong>Trigonometry Based</strong></td>
<td></td>
</tr>
<tr>
<td>GCU <a href="#3">[3]</a></td>
<td>$ x \cdot \cos(x) $</td>
</tr>
<tr>
<td>CosLU <a href="#1">[1]</a></td>
<td>$ (x + \alpha \cdot \cos(\beta x)) \cdot \sigma(x) $</td>
</tr>
<tr>
<td>SinLU</td>
<td>$ (x + \alpha \cdot \sin (\beta x)) \cdot \sigma (x) $</td>
</tr>
<tr>
<td><strong>Others</strong></td>
<td></td>
</tr>
<tr>
<td>ScaledSoftSign <a href="#1">[1]</a></td>
<td>$ \frac{\alpha \cdot x}{\beta + \</td>
<td>x\</td>
<td>} $</td>
</tr>
<tr>
<td>CoLU <a href="#4">[4]</a></td>
<td>$ \frac{x}{1-x \cdot e^{-(x + e^x)}} $</td>
</tr>
<tr>
<td><strong>Linear Combination</strong></td>
<td></td>
</tr>
<tr>
<td>LinComb <a href="#7">[7]</a></td>
<td>$ \sum_{i=1}^{n} w_i \cdot F_i(x) $</td>
</tr>
<tr>
<td>NormLinComb <a href="#7">[7]</a></td>
<td>$ \frac{\sum_{i=1}^{n} w_i \cdot F_i(x)}{\</td>
<td>\</td>
<td>W\</td>
<td>\</td>
<td>} $</td>
</tr>
</tbody>
</table>
<h2 id="contact">Contact</h2>
<p>Alan Huynh - <a href="https://www.linkedin.com/in/alan-huynh-64b357194/">LinkedIn</a> - hdmquan@outlook.com</p>
<p>Project Link: <a href="https://github.com/alan191006/torch_activation">https://github.com/alan191006/torch_activation</a></p>
<h2 id="references">References</h2>
<p><a id="1">[1]</a>
Pishchik, E. (2023). Trainable Activations for Image Classification. Preprints.org, 2023010463. DOI: 10.20944/preprints202301.0463.v1.</p>
<p><a id="2">[2]</a>
Shang, W., Sohn, K., Almeida, D., Lee, H. (2016). Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units. arXiv:1603.05201v2 (cs).</p>
<p><a id="3">[3]</a>
Noel, M. M., Arunkumar, L., Trivedi, A., Dutta, P. (2023). Growing Cosine Unit: A Novel Oscillatory Activation Function That Can Speedup Training and Reduce Parameters in Convolutional Neural Networks. arXiv:2108.12943v3 (cs).</p>
<p><a id="4">[4]</a>
Vagerwal, A. (2021). Deeper Learning with CoLU Activation. arXiv:2112.12078v1 (cs).</p>
<p><a id="5">[5]</a>
So, D. R., Mańke, W., Liu, H., Dai, Z., Shazeer, N., Le, Q. V. (2022). Primer: Searching for Efficient Transformers for Language Modeling. arXiv:2109.08668v2 (cs)</p>
<p><a id="6">[6]</a>
Noam, S. (2020). GLU Variants Improve Transformer. arXiv:2002.05202v1 (cs)</p>
<p><a id="7">[7]</a>
Pishchik, E. (2023). Trainable Activations for Image Classification. Preprints.org, 2023010463. DOI: 10.20944/preprints202301.0463.v1</p>
<p><a id="8">[8]</a>
Weihao, Y., et al (2022). MetaFormer Baselines for Vision. arXiv:2210.13452v2 (cs)</p>
<p><a href="#Installation">Back to top</a></p>
