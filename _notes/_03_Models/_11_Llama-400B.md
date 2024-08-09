
The Llama 3 Herd of models natively supports multilinguality, coding, reasoning, and tool usage.

Our largest model is a dense Transformer with 405B parameters and a context window of up to 128K tokens.

Trained on 15T multilingual tokens

Pretrained using $3.8 X 10^{25}$ FLOPs

**Post training**: 
- supervised finetuning (SFT), 
- rejection sampling (RS), and 
- direct preference optimization (DPO)

**Capabilities:**
- Answer questions in atleast 8 languages
- write high-quality code
- solve complex reasoning problmes
- use tools out-of-box

**Pre-training:**

### Data curation
- We find markdown is harmful to the performance of a model that is primarily trained on web data compared to plain text, so we remove all markdown markers.
- **Data mix summary** Our final data mix contains roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens.

### Parallelism
- To scale training for our largest models, we use 4D parallelism—a combination of four different types of parallelism methods—to shard the model. This approach efficiently distributes computation across many GPUs and ensures each GPU’s **model parameters, optimizer states, gradients, and activations** fit in its HBM.
	- tensor parallelism: Tensor parallelism splits individual weight tensors into multiple chunks on different devices. 
	- pipeline parallelism: Pipeline parallelism partitions the model vertically into stages by layers, so that different devices can process in parallel different stages of the full model pipeline. 
	- context parallelism: Context parallelism divides the input context into segments, reducing memory bottleneck for very long sequence length inputs.  
	- data parallelism: We use fully sharded data parallelism (FSDP; Rajbhandari et al., 2020; Ren et al., 2021; Zhao et al., 2023b), which shards the model, optimizer, and gradients while implementing data parallelism which processes data in parallel on multiple GPUs and synchronizes after each training step.

### Training recipe
The recipe used to pre-train Llama 3 405B consists of three main stages: 
1. initial pre-training, 
2. long-context pre-training, and 
3. annealing.

### Post training
