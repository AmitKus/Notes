
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
