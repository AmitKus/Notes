
[Lectures](https://github.com/cuda-mode/lectures)
- Triton is a DSL and generates a ptx code which is cuda assembly

## Lecture 2 (ch 1-3)

**compiler**
- nvcc (NVIDIA C compiler) is used to compile kernels into PTX
- Parallel Thread Execution (PTX) is a low-level VM & instruction set
- graphics driver translates PTX into executable binary code (SASS)

**CUDA grid**
- 2 level hierarchy: **blocks, threads**
![](attachments/8152bb9b12808d735c6db216a4b9da6a_MD5.jpeg)