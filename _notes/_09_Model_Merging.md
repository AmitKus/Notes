
## [HF: Merge models](https://huggingface.co/blog/mlabonne/merge-models)

Model merging is a technique that **combines two or more LLMs** into a single model.

### Algorithms

#### **Spherical Linear Interpolation** (SLERP)

Basic idea: Direction more important than magnitude of vectors in high-dimensional spaces.

SLERP is implemented using the following steps:

1. Normalize the input vectors to unit length, ensuring they represent directions rather than magnitudes
2. Calculate the angle between these vectors using their dot product.
3. If the vectors are nearly collinear, it defaults to linear interpolation for efficiency. Otherwise, SLERP computing scale factors based on the interpolation factor `t` (`t=0` = 100% of the first vector, `t=1` = 100% of model 2) and the angle between the vectors.
4. These factors are used to weigh the original vectors, which are then summed to obtain the interpolated vector.

**SLERP is currently the most popular merging method, but it is limited to combining only two models at a time.**

#### TIES

TIES-Merging is divided into the following three steps:

1. **Trim**: Reduces redundancy in task-specific models by retaining only a fraction the most significant parameters (density parameter) and resetting the rest to zero.
2. **Elect Sign**: Resolves sign conflicts across different models by creating a unified sign vector based on the most dominant direction (positive or negative) in terms of cumulative magnitude.
3. **Disjoint Merge**: Averages parameter values that align with the unified sign vector, excluding zero values.

**Unlike SLERP, TIES can merge multiple models at a time.**

#### DARE

Similar to TIES with couple of differences

- **Pruning**: DARE randomly reset fine-tuned weights to their original values (those of the base model).
- **Rescaling**: DARE rescales the weights to keep the expectations of model outputs approximately unchanged. It adds the rescaled weights of both (or more) models to the weights of the base model with a scale factor.

#### Passthrough

The passthrough method differs significantly from the previous ones. By concatenating layers from different LLMs, it can produce models with an **exotic number of parameters** (e.g., 9B with two 7B parameter models). These models are often referred to as "frankenmerges" or "Frankenstein models" by the community.



## Evolutionary Optimization of Model Merging Recipes

In this work, we present a methodology that leverages evolutionary algorithms to facilitate the merging of foundation models. Our approach is distinguished by its ability to navigate both parameter space (weights) and the data flow space (inference path), proposing a framework that integrates these two dimensions.

- Automated model composition
- 

### Resources
- [HF: Merge models](https://huggingface.co/blog/mlabonne/merge-models)
- [github: mergekit](https://github.com/arcee-ai/mergekit) 