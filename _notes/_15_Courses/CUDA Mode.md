
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

### Programming model

With CUDA, we decompose the computation in 2 levels: First into blocks, then each block further into threads. All threads in a block run on the same SM and share the same Shared Memory. And each thread computes on **scalars**.

In Triton, we decompose the computation only in 1 level: Into blocks. There is no further decomposition into threads. **Triton requires us to perform operations on vectors**. Also, we don't need to and are not able to manage the shared memory. Triton does that automatically.

Example:

Let's say we want to add `x` and `y`, which are vectors of size 8, and save the output into `z` (also size 8). Let's use blocks of size 4, so we have `8 / 4 = 2` blocks.

- Cuda runs 2 blocks, each with 4 threads. Each of the 8 threads computes a single position, e.g. `z[0] = x[0] + y[0]`

- Triton also runs 2 blocks, which each performs vectorized addition. The vector size is the block size, which is 4. E.g. `z[0:3] = x[0:3] + y[0:3]`



## Lecture 15: CUTLASS

CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance matrix-matrix multiplication (GEMM) and related computations at all levels and scales within CUDA. It incorporates strategies for hierarchical decomposition and data movement similar to those used to implement cuBLAS and cuDNN. CUTLASS decomposes these "moving parts" into reusable, modular software components abstracted by C++ template classes. Primitives for different levels of a conceptual parallelization hierarchy can be specialized and tuned via custom tiling sizes, data types, and other algorithmic policy. The resulting flexibility simplifies their use as building blocks within custom kernels and applications.

![](attachments/1179a199420321a50829fa4202ea2636_MD5.jpeg)


## Lecture 22: [Speculative Decoding](https://docs.google.com/presentation/d/1p1xE-EbSAnXpTSiSI0gmy_wdwxN5XaULO3AnCWWoRe4/edit#slide=id.p)

- ChatGPT was trained on RAY

### vLLM's core principles
- Ease of use
- Great performance
- Hardware agnostic

### vLLM Performance features
- PagedAttention/tensor parallelism
- Optimized multi-LoRA
- Chunked prefill
- Automatic prefix caching
- Guided decoding
- Quantization (fp8 WIP, and others)
- Pipeline-parallelism (WIP)
- Prefill disaggregation (WIP)


### Hardware agnostic
- Nvidia, AMD, Inferentia, TPU (WIP), CPU

### Speculative decoding
https://x.com/karpathy/status/1697318534555336961?lang=en
*Speculative execution for LLMs is an excellent inference-time optimization. It hinges on the following unintuitive observation: forwarding an LLM on a single input token takes about as much time as forwarding an LLM on K input tokens in a batch (for larger K than you might think). This unintuitive fact is because sampling is heavily memory bound: most of the "work" is not doing compute, it is reading in the weights of the transformer from VRAM into on-chip cache for processing. So if you're going to do all that work of reading in all those weights, you might as well apply them to a whole batch of input vectors.*

*Now the clever idea is to use a small and cheap draft model to first generate a candidate sequence of K tokens - a "draft". Then we feed all of these together through the big model in a batch. This is almost as fast as feeding in just one token, per the above. Then we go from left to right over the logits predicted by the model and sample tokens. Any sample that agrees with the draft allows us to immediately skip forward to the next token. If there is a disagreement then we throw the draft away and eat the cost of doing some throwaway work (sampling the draft and the forward passing for all the later tokens).* 

*The reason this works in practice is that most of the time the draft tokens get accepted, because they are easy, so even a much smaller draft model gets them. As these easy tokens get accepted, we skip through those parts in leaps. The hard tokens where the big model disagrees "fall back" to original speed, but actually a bit slower because of all the extra work.* 

*So TLDR: this one weird trick works because LLMs are memory bound at inference time, in the "batch size 1" setting of sampling a single sequence of interest, that a large fraction of "local LLM" use cases fall into. And because most tokens are "easy".**

**Speculative decoding: Why works**
- **Inference for single request is memory bound**
- **Producing a single token or multiple tokens with batching takes same time**
- **What to batch?**
	- **Choose a small model that produces next K tokens**
	- **K Batch request to the Large model: predict kth token given k-1 tokens**
	- **Rejection sampling: select if $p_{large}(x) > p_{small}(x)$ else accept with prob $p_{large}/p_{small}$**
- **No loss in accuracy if rejection sampling used**
- **Same tokenizer, same vocabulary typically**


- Memory boundedness
	- LLM inference in memory bound
	- The unused compute can be used, if we find a way to use it
- Not all parameters required for every token
	- Ex: What is the capital of California?
- Idea:
	- Try to predict what large model will say
	- Get probabilities of predictions
	- Use heuristic to accept or reject the predictions based on probabilites

**Can speculative decoding help with large batch sizes which are not memory bound but compute bound?**

**vLLM is designed for throughput not for latency. Mainly for large batch size inferences**

**![](attachments/29b6b243dd35e6a9ba9440c894e25ca3_MD5.png)**

### How to evaluate speed-up?

- Simplified version:
	- Inter-token latency = step time / number of tokens per step in expectation
	- Example without speculative decoding: 30ms / 1 → 1 token per 30ms
	- Example with speculative decoding: 40ms / 2.5 → 1 token per 16ms

- Key factors
	- How long does it take to propose?
	- How accurate are the proposals?
	- How long does it take to verify / other spec framework overheads?
	
- In practice:
	- [https://github.com/vllm-project/vllm/blob/main/vllm/spec_decode/metrics.py](https://github.com/vllm-project/vllm/blob/main/vllm/spec_decode/metrics.py) 
	- Acceptance rate – “How aligned is the proposal method with the target model?”
	- System efficiency – “How efficient is the deployment compared to 100% acceptance rate?”

### Losslessness

**Is the output of speculative decoding different than the target model?**
- TL;DR No if using rejection sampling, subject to hardware numerics
- Diagram [https://github.com/vllm-project/vllm/pull/2336](https://github.com/vllm-project/vllm/pull/2336) 
- Yes if using lossly sampling technique, e.g. Medusa’s typical acceptance (but higher acceptance rate!)

Recommended reading (proof of losslessness): [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318)**

![](attachments/8f81003db5d730a4cf34859d910c8dd5_MD5.jpeg)

### Medusa

- [https://sites.google.com/view/medusa-llm](https://sites.google.com/view/medusa-llm) 
- [https://arxiv.org/pdf/2305.09781](https://arxiv.org/pdf/2305.09781) 
- [https://www.together.ai/blog/sequoia](https://www.together.ai/blog/sequoia)

### Bonus token
![](attachments/da774b4195cfef703a0ffb57d7a805bd_MD5.jpeg)

### Recovered token

![](attachments/ecc95e58788259d7f0ee410260ca2e41_MD5.jpeg)


### Dynamic speculative decoding

### Batch expansion

![](attachments/aed401ffa8a416fc4ccc629c30cd382a_MD5.jpeg)

### Future
**- More engineering
- Retrieval-acceleration [https://arxiv.org/html/2401.14021v1](https://arxiv.org/html/2401.14021v1) 
- Chunked prefill + spec decode
- Prefix caching + spec decode
- Guided decoding + spec decode
- Inferentia / TPU / CPU support
-**More modeling**
- Meta-model for speculation length
- Meta-model for speculation type
**-Large / mixed engineering+modeling**
- Multi-LoRA draft model (specialize to domains)
- Online learning draft model [https://arxiv.org/abs/2310.07177](https://arxiv.org/abs/2310.07177)
- Batched parallel decoding [https://github.com/vllm-project/vllm/issues/4303](https://github.com/vllm-project/vllm/issues/4303)


## Lecture 18: Fusing Kernels


