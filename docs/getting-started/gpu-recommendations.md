# GPU Recommendations

Determining the type of GPUs needed to run an AI pipeline can be complicated.  The minimum GPU and compute requirements for matchminer-ai-inference also vary for each step. While we cannot summarize all factors which impact AI performance, this guide provides information on the following:

1. [Models used in our pipeline](#models-used-in-our-pipeline); 
2. [Estimating Minimum GPU memory for a LLM model](#minimum-gpu-memory);
3. [Minimum requirements for the software stack](#minimum-software-stack-requirements); and
4. [Our experience](#our-experience). 

## Models Used In Our Pipeline
MatchMiner-AI is a pipeline which uses 4 different models:

- An LLM for patient summarization and trial 'space' extraction. We currently use variants of Google's Gemma-4-31B-IT model.
- 'TrialSpace' - an embedding model [https://huggingface.co/ksg-dfci/TrialSpace-0526](https://huggingface.co/ksg-dfci/TrialSpace-0526) , fine-tuned from Qwen 3 0.6B Embedding [https://huggingface.co/Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
- 'TrialChecker' - a text classifier [https://huggingface.co/ksg-dfci/TrialChecker-0526](https://huggingface.co/ksg-dfci/TrialChecker-0526) , fine-tuned from ModernBERT-large [https://huggingface.co/answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large).
- 'BoilerplateChecker' - another text classifier [https://huggingface.co/ksg-dfci/BoilerplateChecker-0526](https://huggingface.co/ksg-dfci/BoilerplateChecker-0526) , fine-tuned from ModernBERT-large.

For any given model, minimum compute/GPU requirements could be determined from others' experiences either with the model itself (e.g. Gemma-4-31B-IT) or with the base (pre-trained) models (Qwen 3 0.6B Embedding, ModernBERT-large). 

## Minimum GPU Memory
Nvidia provides some guidance and a short-hand metric for determining how much GPU VRAM is required to run a given LLM. (See [GPU Memory Essentials for AI Performance](https://developer.nvidia.com/blog/gpu-memory-essentials-for-ai-performance/)).  Specifically, it recommends having at least Parameter * Precision * 2 VRAM. Where Parameter is the number of parameters in the model and Precision is the bytes per parameter for the model's precision format -- FP32, FP16, FP8 or FP4.  (See conversion table below.)

Precision Bytes per Parameter. From [GPU Memory Essentials for AI Performance](https://developer.nvidia.com/blog/gpu-memory-essentials-for-ai-performance/)

| Precision | Bytes per Parameter |
|-----------|----------------------|
| INT32/FP32 | 4 |
| INT16/FP16 | 2 |
| INT8/FP8 | 1 |
| INT4/FP4 | 0.5 |

For example, when we run [RedHatAI/gemma-4-31B-it-FP8-Dynamic](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-Dynamic) for patient and trial summarization, we should have at least

*31,000,000,000 parameter* (31 billion parameters) * *1 byte/parameter* * *2 VRAM* (~62 GB)

In practice, we run RedHatAI/gemma-4-31B-it-FP8-Dynamic on H100 (NVL PCIe) GPUs with 94GB VRAM each. 

## Minimum Software Stack Requirements 
The [dependencies in our pipeline](https://github.com/dfci/matchminer-ai-inference/blob/main/pyproject.toml) include packages which have particular compute/GPU requirments.  Currently these include:   
    
- "torch>=2.11.0"
- "vllm>=0.20.2"

Documentation for each package is available online: 
- [PyTorch (torch)](https://pytorch.org/get-started/locally/)
- [vllm](https://docs.vllm.ai/en/v0.6.5/getting_started/installation.html)
 
Other software such as sentence-transformers and transformers are dependent on the torch version.

## Our Experience
We have successfully run the entire pipeline on the following machines:

| Machine | Notes |
|---------|-------|
| RTX Pro 6000 (available on the cloud) | Has native FP4 kernels, so can run NVFP4 versions of Gemma 4, such as nvidia/Gemma-4-31B-IT-NVFP4. | 
| H100 NVL PCI / 300W, which have native FP8 but not native FP4 | Can run RedHatAI/gemma-4-31B-it-FP8-Dynamic, albeit slowly. (See comment below.) |
| NVIDIA A100 80GB | We have run the example notebook steps with google/gemma-4-31B-it as the LLM on this machine type.|

RedHatAI/gemma-4-31B-it-FP8-Dynamic at the context lengths we need runs slowly on our H100 machines. It runs faster as an NVFP4 quant (eg [https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) ) on a Blackwell GPU, eg, the RTX Pro 6000.