
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


## Lecture 3

1. **Streaming Multiprocessors (SMs):** In NVIDIA GPUs, SMs are the fundamental units of execution. Each SM can execute multiple threads concurrently.

2. **Thread Blocks:** A thread block is a group of threads that can cooperate among themselves through shared memory and synchronization. All threads in a block are executed on the same SM. This means they can share resources such as shared memory and can synchronize their execution with each other.

3. **Shared Memory:** Shared memory is a small memory space on the GPU that is shared among the threads in a block. It is much faster than global memory (the main GPU memory), but it is also limited in size. Threads in the same block can use shared memory to share data with each other efficiently.

- The RTX 3090, based on the Ampere architecture, has 82 SMs.

- Each SM in GA10x GPUs contain 128 CUDA Cores, four third-generation Tensor Cores, a 256 KB Register File, and 128 KB of L1/Shared Memory

- In CUDA, all threads in a block have the potential to run concurrently. However, the actual concurrency depends on the number of CUDA cores per SM and the resources required by the threads.


## Lecture 4: Compute and memory basics

![](attachments/c099e90d0ed5de0e8398b905150b83fd_MD5.jpeg)

![](attachments/24d4da152b10cb7be00b1fd0d19cc21f_MD5.jpeg)

- Threads is a block are executed in parallel on the same SM
- Blocks are completely independent (exceptions in new GPUs)
- Thread block runs on an SM is divided into Warps of 32 threads
	- Each warp run on a fixed of the SMs processing unit (FP32 cores above etc)
	- All warps simultaneously assigned to the processing unit take turns but registers stay

![](attachments/4d00f2450e0a5a024370f9f342d384c4_MD5.jpeg)
![](attachments/7d75771f720475ec9e2cfb4d70191d6a_MD5.jpeg)

```python
p = torch.cuda.get_device_properties(0)
print(p)
p.regs_per_multiprocessor
```
[output]_CudaDeviceProperties(name='NVIDIA GeForce RTX 3060', major=8, minor=6, total_memory=11938MB, multi_processor_count=28)

