
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
p.max_threads_per_multi_processor
```
[output]_CudaDeviceProperties(name='NVIDIA GeForce RTX 3060', major=8, minor=6, total_memory=11938MB, multi_processor_count=28)


### Memory architecture

- Kernel fusion is the key
![](attachments/0386db23676ec5f1b40462e1b1786f58_MD5.jpeg)
#### Profiling
```python
with torch.profiler.profile() as prof:
	%timeit -n 1000 gelu(x)

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
```


![](attachments/9ed55f9e87fc815e9032508af8f62d02_MD5.jpeg)
![](attachments/8ba1d475edb127c4be077d31fc2aacef_MD5.jpeg)


## Lecture 5

![](attachments/3d034c4573b3f968ab36c5a12f6f36df_MD5.jpeg)

### Numba

- [# Debugging CUDA Python with the the CUDA Simulator](https://numba.pydata.org/numba-doc/dev/cuda/simulator.html#debugging-cuda-python-with-the-the-cuda-simulator "Permalink to this headline")
  - Use to write/debug cuda and then convert to pure Cuda

```python
from numba import cuda
```

## Lecture 6: Optimizing optimizers

- Kernel fusing
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [quanto](https://huggingface.co/blog/quanto-introduction)
- Course: [MIT Efficient ML](https://hanlab.mit.edu/courses/2023-fall-65940)

![](attachments/737ccf8a5ca4d4ba55c5ff0559285f47_MD5.jpeg)
![](attachments/bc321db0e598769d05651f56cc0f0039_MD5.jpeg)
![](attachments/8243eefe644a15db2e9c846d45018c8a_MD5.jpeg)
## Lecture 7: Quantization Cuda vs Triton

- [Torch AO](https://github.com/pytorch/ao)
![](attachments/9b7ed92615cbde9e7b79f23c86dade2a_MD5.jpeg)
- Llama inference is not compute bound so need to get weights in as fast as possible so weight only is good. Activation are already in there.
- Better to use weight only quantisation for memory-bound system.

- Segregate quantization based on channel's etc.
- Outliers, non-stationary distribution effect the quantization

![](attachments/a1186b834986ebc12707b8b51e2b5301_MD5.jpeg)
![](attachments/776b04f4da95090fdcb18e45ad2374ed_MD5.jpeg)
![](attachments/7e7ef58308c6451207900e1995d313d9_MD5.jpeg)
## Lecture 8: CUDA Performance Checklist

[Good paper](https://arxiv.org/pdf/1804.06826)

![](attachments/751f6fce60dc16437c426624db4b1bd6_MD5.jpeg)

### Performance checklist

- Coalesced Global Memory Access
- Maximize occupancy
- Understand if memory or compute bound
- Minimize control divergence
- Tiling of reused data
- Privatization
- Thread Coarsening
- Rewrite your algorithm using better math

### [Is latency stupid](http://www.stuartcheshire.org/rants/latency.html)

- Throughput is easy, latency is not: “You can get 80 phone lines in parallel, and send one single bit over each phone line, but that 100ms latency is still there.”
- Quantization: “For example, to reduce packet size, wherever possible Bolo uses bytes instead of 16-bit or 32-bit words.”

![](attachments/d86ce29da607228e6db8c73f63262a98_MD5.jpeg)

### Arithmetic intensity
- [Nvidia Doc](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf)
![](attachments/e32e4598bd5863cb604953b49a97165e_MD5.jpeg)
 
### Key takeaways

- **Bandwidth Bound Kernels: Fuse, quantize, compile** 
- **Compute Bound Kernels: Write a better algorithm**

### Privatisation

- Apply partial updates to private copy of data before writing back to global or shared memory. 
- Example: Sliding window algorithm
- 1   2  [3] [4] [5]  6   7
- Higher occupancy
- Higher compute SM throughput
- Lower DRAM throughput
![](attachments/b614d8e862ac37fe03a9f5d54ee7a49e_MD5.jpeg)

### [Online softmax](https://arxiv.org/pdf/1805.02867)

Flash Attention v1 has 2 tricks: softmax(QK^T) V
- Tile based shared memory attention computation
- Online softmax

## Lecture 9: Reductions

- **Operations that reduce the output size
- Most typical take a vector and produce a scalar
- min, max, argmax, argmin norm, sum, prod, mean, unique**

**![](attachments/cc6918c8a432e981f65dfc72250d7709_MD5.png)
![](attachments/358b1ea28c4afc9ae3779dd55d20acc2_MD5.jpeg)
![](attachments/374119864cd21f170449c969ed2801b6_MD5.jpeg)
![](attachments/e736e16e618c0e3aaf41264e18b56eb7_MD5.jpeg)
**torch.use_deterministic_algorithms(True)**: Performance penalty

![](attachments/968d6fd4204d5e9dc7c18c4e3851ef1d_MD5.jpeg)


## Lecture 11
![](attachments/ac367608f82c7852da42dbd16c5c6020_MD5.jpeg)

![](attachments/e3ea37efa33ad0cebb9fadaab152a2eb_MD5.jpeg)
![](attachments/7d8cf093a28c4a5b29a1a326219f887a_MD5.jpeg)
![](attachments/8279b81d503819a59e58e99e552629c8_MD5.jpeg)
![](attachments/4a6c22a23ce1966fc36db02451f3f5f0_MD5.jpeg)
![](attachments/0be6b9021650e19fdd0d6aad4d90c9ab_MD5.jpeg)
**1.6x in practice**

![](attachments/535b2a3896eec4a12e40483e43166cee_MD5.jpeg)

## Lecture 12: Flash attention

![](attachments/8d41b388d8c8dba347b5db35d19418fc_MD5.jpeg)
- Typically, 1 head is run on 1 block

![](attachments/4ab2dac1c5d33c7145e6c939817b8cc0_MD5.jpeg)

![](attachments/60203cf1f6e944014062d88b98af7a9b_MD5.jpeg)
## Lecture 13: Ring Attention

![](attachments/69d28b249070d9218e8183ebf9d611df_MD5.jpeg)

![](attachments/7b35eccc50788ac143dae84ac59c731f_MD5.jpeg)



![](attachments/c49397cad79061e9950069bf547e15cd_MD5.jpeg)
![](attachments/b0895833948890360e24716a169b4a14_MD5.jpeg)

![](attachments/3b0a7129897bf895568508276d96b957_MD5.jpeg)

![](attachments/f5406a6d23858f4e6bcc4d687b29e32c_MD5.jpeg)
- Split the batch sequence-wise
![](attachments/10c4254fa26f1b9ceeab798c94f3a1b9_MD5.jpeg)
![](attachments/722f933044a7c18cdfd0e551aef065ad_MD5.jpeg)
### Issues with Ring attention: idle nodes 

![](attachments/9cabfd4aee21df203896b7bbcc321c5f_MD5.jpeg)



![](attachments/a72f908dc6b074681720cb56fd6ff273_MD5.jpeg)

### Solution: Striped attention

![](attachments/8e773a016f656b100dff0e65ec4c1c43_MD5.jpeg)

## Lecture 14: Triton

[Notebook](https://github.com/cuda-mode/lectures/blob/main/lecture_014/A_Practitioners_Guide_to_Triton.ipynb)


**What is Triton**
- In short: Triton is a language to program GPUs more conventiently. You write Python-ish code, which is then compiled into ptx code (the same thing cuda code is compiled into).
- During the compilation, the Triton compiler tries to use clever tricks to rearrange the parts of your program (without changing the program's meaning!) to make it run faster.

CUDA is a high-end tool with many settings for the pros.
- full control over everything, so absolute max performance possible
- harder to get decent performance
- way more tedious to write and debug
- more complicated, so harder to learn

Triton is a very good tool for most users
- you can't control everything, as some things are left to automatic optimization; so you probably won't get absolute max performance
- way easier to get good performance
- way easier to write and debug
- easier to learn, as it has a Python-like syntax

**Triton vs torch.compile**

`torch.compile` makes your model faster by trying to use existing kernels more effectively and creating simple new kernels. This may make your model fast enough. If not, you can decide to invest time to write faster Triton kernels.

(These simple new kernels that `torch.compile` creates are actually Triton kernels. So they are a good starting point for your custom kernels. See [Mark Saroufim](https://twitter.com/marksaroufim)'s [lecture 1 of cuda mode](https://www.youtube.com/watch?v=LuhJEEJQgUM&t=2200s) for how.)

**When to use Triton**

You start with your AI model.
1. If it's not fast enough, `torch.compile` it.
2. If it's not fast enough, check if you can rewrite your code to make it more suitable for `torch.compile`.
3. If it's not fast enough, check which parts are slow and write custom Triton kernel(s) for those.
4. If it's not fast enough, check which parts are slow and write custom CUDA kernel(s) for those.

(In the unlikely case you know beforehand that you need absolute max performance, you can decide to directly start with CUDA.)

**How to run Triton**
Unlike with CUDA, we can debug Triton kernels just like any CPU program, if we set the environment variable `TRITON_INTERPRET = 1`.
