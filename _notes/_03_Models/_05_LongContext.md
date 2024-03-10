
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