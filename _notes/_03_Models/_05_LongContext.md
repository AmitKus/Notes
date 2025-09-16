---
---


## Effective Long-Context Scaling of Foundation Models
<details>
  <summary>Notes</summary>

### Key points

- Llama2 models with context windows of up to 32 K tokens
- Ablation experiments conclusions: 
	- Having abundant long texts in the pretrain dataset is **NOT** the key to achieving strong performance
	- Long context continual pretraining is more efficient and similarly effective compared to pretraining from scratch with long sequences
- **power-law scaling**: context length is another important axis of scaling LLMs
	- validation loss decreases with increasing context-length 
- Observe modest improvements on standard short-context tasks, especially on coding, math and knowledge benchmarks, along with significant improvements on long-context tasks

![](attachments/56a6fa2acd4db1987a862330d3841579_MD5.jpeg)

</details>


## Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone

We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion tokens, whose overall performance, as measured by both academic benchmarks and internal testing, rivals that of models such as Mixtral 8x7B and GPT-3.5.

The innovation lies entirely in our dataset for training, a scaled-up version of the one used for phi-2, composed of heavily filtered web data and synthetic data.

The phi-3-mini model is a transformer decoder architecture [VSP+17], with default context length 4K. We also introduce a long context version via LongRope [DZZ+24] that extends the context length to 128K, called phi-3-mini-128K

To best benefit the open source community, phi-3-mini is built upon a similar block structure as Llama-2 [TLI+23] and uses the same tokenizer with vocabulary size of 320641.

![](attachments/1f72df336e3c282f305f1ce04c80c19e_MD5.jpeg)

Post-training. Our models went through post-training with both supervised instruction fine-tuning, and preference tuning with DPO.

The model simply does not have the capacity to store too much “factual knowledge”, which can be seen for example with low performance on TriviaQA. However, we believe such weakness can be resolved by augmentation with a search engine.


## LLama3

- standard decoder-only transformer architecture in Llama 3. 
- Llama 3 uses a tokenizer with a vocabulary of 128K tokens that encodes language much more efficiently, which leads to substantially improved model performance. 
- To improve the inference efficiency of Llama 3 models, we’ve adopted grouped query attention (GQA) across both the 8B and 70B sizes. 
- We trained the models on sequences of 8,192 tokens, using a mask to ensure self-attention does not cross document boundaries.
- Trained on 15T tokens
-  5% of the Llama 3 pretraining dataset consists of high-quality non-English data that covers over 30 languages.
- [User guide](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3)


## Mixtral 8x22B

![](attachments/59a34af01fe647110ac8d6f27874525f_MD5.jpeg)

Mixtral 8x22B is our latest open model. It sets a new standard for performance and efficiency within the AI community. It is a sparse Mixture-of-Experts (SMoE) model that uses only 39B active parameters out of 141B, offering unparalleled cost efficiency for its size.

Mixtral 8x22B comes with the following strengths:

- It is **fluent in English, French, Italian, German, and Spanish**
- It has strong mathematics and coding capabilities
- **It is natively capable of function calling**; along with the constrained output mode implemented on la Plateforme, this enables application development and tech stack modernisation at scale
- Its **64K tokens context window** allows precise information recall from large documents