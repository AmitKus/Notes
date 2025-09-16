---
---

# _03_Training

## Papers

- [Scaling Data-Constrained Language Models](https://arxiv.org/pdf/2305.16264.pdf)


## Domain-adapted

- [ChipNeMo: Domain-Adapted LLMs for Chip Design](https://d1qx31qr3h6wln.cloudfront.net/publications/ChipNeMo%20%282%29.pdf)


# Notes

## Everything about Distributed Training and Efficient Finetuning

### Distributed training
LLMs require a LOT of GPU vRAM to train, 
- not just because of the large model weights 
  - Falcon 40B: 40B parameterd need around 74GB for model weights in BF16 
- but also because of optimizer states 
  - Vanilla AdamW: 12 bytes per parameter to store a copy of the model weights, the momentum and the variance parameters

### Main types

#### 1. Data parallelism
- Each GPU worker gets a fraction of the total mini-batch of data
- Computes the gradients on that fraction of the data
- The gradients are then averaged across all workers, and the model weights are updated.
- Each GPU stores a copy of the model weights, optimizer state and gradients for the fraction of the data it’s working on.

#### 2. Model parallelism/Vertical Model Parallelism (MP):
- Models are vertically sliced, with different layers of the model placed on different GPU workers.
- In naive model parallelism, only 1 GPU is active a given time
- An improvement is Pipeline Parallelism (PP), which gives you the illusion of parallelism by overlapping computation for different micro-batches of data.

![](pipeline_parallelism_vs_naive.webp)

#### 3. Tensor parallelism
- In tensor parallelism, each GPU processes only a slice of a tensor by horizontally slicing the model across GPU workers.
- Each worker processes the same batch of data, 
- computing the activations for the part of the weights they have, 
- exchanging parts that each other needs, with each worker 
- computing the gradients for the slice of the weights it has.

#### Improvements to Data Parallelism

1. Zero Redundancy Optimizer and 
2. (the closely related) Fully Sharded Data-Parallel strategies.

#### ZeRO-powered Data-Parallelism
- one of the most efficient and popular strategies for distributed training at the moment.
- DeepSpeed's Zero massively improves on memory efficiency.
- The main idea is that the 
  - ZeRO exploits memory redundancy in data-parellel training and 
  - the latest improvements in fast inter-GPU communication to improve throughput

1. ZeRO-DP (data pallelelism) and 
2. ZeRO-R (residual memory). 
3. ZeRO-Offload/Infinity (offloading computation to CPU/ NVMe disk) and 
4. ZeRO++ (with flexible multi-node training and quantized weights).

#### Fully-sharded Data Parallel

Fully-Sharded Data Parallel (FSDP) is another data-parallelism technique aimed at improving memory efficiency with limited communication overhead, and thus throughput.
Two sharding strategies:
1. Full-sharding: This is mostly the same as ZeRO-3 where you have parameters, optimizer state and gradients being sharded across workers/ devices.

![](FSDP_diag.webp)
2. Hybrid-sharding

#### Implementations

**How can you use DeepSpeed and FSDP?**
- One of the main advantages of DeepSpeed ZeRO/ FSDP is that you get the kind of memory savings and throughput in data + tensor parallelism while actually being only in a data-parallel setting.
- This means that you do not need any ad-hoc architecture changes, or change your forward pass with messy .to() device castings, or any customizations.
- Accelerate library integrates these

**What about Pipeline Parallelism and Tensor Parallelism?**
- currently require architecture changes and/or changes in the forward pass of the model.
- If you really do want PP and TP, the best option for now seems to be to use Megatron-LM and stick to the models they support (BERT, GPT-2, T5, Llama)

### Efficient fine-tuning

#### Mixed-precision
- weights, activations and gradients are stored in half-precision formats while you have a “master copy” of the weights in FP32/ single-precision.
- [Mixed-precision training](https://arxiv.org/pdf/1710.03740.pdf%EF%BC%89%E3%80%82)

#### Parameter-efficient fine-tuning
- Traininig a subset/smaller number of weights. (LoRA, $IA^3$)
- [A Conceptual Guide to LoRA in 🤗 PEFT](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [A Conceptual Guide to IA in 🤗 PEFT](https://huggingface.co/docs/peft/conceptual_guides/ia3)

#### [Flash-attention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
Flash attention is a 
- fast (you get speedup!), 
- memory-efficient: compared to vanilla attention, which is quadratic in sequence length, O(N²), this method is sub-quadratic/linear in N (O(N)).
- exact: meaning it’s not an approximation of the attention mechanism (like e.g. sparse, or low-rank matrix approximation methods) — its outputs are the same as in the “vanilla” attention mechanism.
- IO-aware: it leverages the knowledge of the memory hierarchy of the underlying hardware
- Attention is memory-bound
![](attention_memory_bound.webp)
![](standard_attention_mech.webp)

#### Gradient/ Activation Checkpointing

- Typically, in each forward pass, all the intermediate activations are retained in memory, as they are needed to compute the backward pass. 
- Gradient/Activation checkpointing is a technique to reduce memory consumption by only retaining a subset of intermediate activations, and recomputing the rest as needed. 
- The tradeoff involved is in the additional recomputation time. 
- A good rule of thumb from HuggingFace is that gradient checkpointing slows down training by 20%. The memory requirement for activations, when you have N model layers, drops off from $O(N)$ to $O(\sqrt{N})$

#### Quantization

Two types of quantization:
1. Post-training quantiation (PTQ): These are approaches aimed at efficient inference.
2. Quantization-aware training: 
   - QLora: 
     - The main idea with QLoRA is that it quantizes the base, pretrained model weights to 8/4 bits and then trains additional LoRA parameters in floating-point half/full precision. 
     - This is a very powerful strategy, enabling finetuning of 60B+ parameter models on a single GPU with 48GB vRAM.
     - Worse throughput due to de-quantization step happening whenever you compute activations for a given layer.

#### Gradient accumulation

Gradient accumulation is a way to increase your effective batch size at some drop in throughput.

#### So wait, should I always try to increase batch size?
- No!
- Increasing the batch size can effect throughput resulting in longer per-step latency.
- There’s more to simply increasing batch size because this can hurt convergence.
- One more insight from a DeepSpeed author is that a global batch size is usually fixed for large scale training runs to achieve a targeted convergence rate (source). 
- Finally, more batch size $\neq$ better!


#### How big is too big?

To provide some perspective, BLOOM-176B pretraining was done on 366B tokens with a global batch size of 2048. What is too big to hurt model convergence is not clear yet, and that too in the fine-tuning regime.


### Practical guidelines
- BF16/ FP16 by default
- Use LoRA with trainable parameters added to all the linear layers.
- Use Flash Attention if your GPU supports it.
- Use Gradient/Activation Checkpointing.
- Use an efficient sampler in your dataloader, like the multi-pack sampler.
- If you have multiple GPUs, always try BF16 + LoRA + Gradient Checkpointing + DeepSpeed ZeRO 3 first.
- Use quantization when you have very limited GPU memory. 
- With more and more GPUs (say 8 V100s or A100s), DS ZeRO-3 should be the way to go.
- In a small-scale multi-node setup, with a few nodes, the best option seems to be DeepSpeed ZeRO-3 with hierarching partitioning enabled (or FSDP with hybrid sharding).
- Gradient accumulation should be used if you’re still short on batch size after all the above optimizations.
- If you’re really short on GPU memory, then you would activate CPU/ disk offloading
- Calculate the effective batch size and adjust hyperparameters accordingly. A general guideline is to scale up the learning rate with the effective batch size.

### Fine-tuning tools

- [Axolotl]()
