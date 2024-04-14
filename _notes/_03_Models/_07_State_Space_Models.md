
## Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Limitations of self-attention**
- The efficacy of self-attention is attributed to its ability to route information densely within a context window, allowing it to model complex data.
- However, this property brings fundamental drawbacks: an inability to model anything outside of a finite window, and quadratic scaling with respect to the window length.

**State space models** 
- These models can be interpreted as a combination of recurrent neural networks (RNNs) and convolutional neural networks (CNNs), with inspiration from classical state space models
- This class of models can be computed very efficiently as either a recurrence or convolution, with linear or near-linear scaling in sequence length.

**We propose a new class of selective state space models, that improves on prior work on several axes to achieve the modeling power of Transformers while scaling linearly in sequence length.**

## Transformers are RNN

### Key
- we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from O (N 2) to O (N ), where N is the sequence length.
- we introduce the linear transformer model that significantly reduces the memory footprint and scales linearly with respect to the context length.

### Linear transformer

- $$T_l(x) = f_l(A_l(x) + x)$$
- $A_l$: self attention function and is the only part of the transformer that acts across sequences.
- 