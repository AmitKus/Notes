
## Sora: Video generation models as world simulators

### Key points
- We explore large-scale training of generative models on video data. 
- Train text-conditional diffusion models jointly on videos and images of variable durations, resolutions and aspect ratios. 
- We leverage a transformer architecture that operates on spacetime patches of video and image latent codes. 
- Our largest model, Sora, is capable of generating a minute of high fidelity video. 
- Our results suggest that scaling video generation models is a promising path towards building general purpose simulators of the physical world.

### Training
- Generative modeling of video data historically: recurrent neural networks, generative adversarial networks, auto-regressive transformers
- Tokens (LLM) <-> Patches (Vision)
- Videos -> Lower dimensional latent space -> space time patches

![](attachments/3f6d11a9141993010964acb91a7d7893_MD5.jpeg)

