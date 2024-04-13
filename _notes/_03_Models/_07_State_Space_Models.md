
## Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Limitations of self-attention**
- The efficacy of self-attention is attributed to its ability to route information densely within a context window, allowing it to model complex data.
- However, this property brings fundamental drawbacks: an inability to model anything outside of a finite window, and quadratic scaling with respect to the window length.

**State space models** 
- These models can be interpreted as a combination of recurrent neural networks (RNNs) and convolutional neural networks (CNNs), with inspiration from classical state space models
- This class of models can be computed very efficiently as either a recurrence or convolution, with linear or near-linear scaling in sequence length.

**We propose a new class of selective state space models, that improves on prior work on several axes to achieve the modeling power of Transformers while scaling linearly in sequence length.**

