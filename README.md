# ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification

This repository provides the implementation for our paper "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification". Our approach introduces an adaptive KV cache mixed-precision quantization method for LLMs.

[arXiv](https://arxiv.org/pdf/2405.14256) | [BibTeX](https://scholar.googleusercontent.com/scholar.bib?q=info:0wKnyPeFiagJ:scholar.google.com/&output=citation&scisdr=ClFh5NBhEOb2yBOtPYQ:AFWwaeYAAAAAZqCrJYRNPlzMV0rmEN80DyEui-Y&scisig=AFWwaeYAAAAAZqCrJciDVFyQVLBEdGtvShl5UxQ&scisf=4&ct=citation&cd=-1&hl=zh-CN)

## Getting Started

Follow the step-by-step tutorial to set up EFFICIENTDM.

### Step 1: Setup
Create a virtual environment and install dependencies as specified by requirements.txt. Then install zipcache as follows:
```python
python3 setup.py

```

### Step 2: Download Pretrained Models
Download the pretrained LLaMA model from huggingface and modify the MODEL_PATH in zipcache_generation_demo.py.

### Step 3: Inference with ZipCache
```python
python3 zipcache_generation_demo.py

```

## BibTeX
If you find this work useful for your research, please consider citing:
```
@article{he2024zipcache,
  title={ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification},
  author={He, Yefei and Zhang, Luoming and Wu, Weijia and Liu, Jing and Zhou, Hong and Zhuang, Bohan},
  journal={arXiv preprint arXiv:2405.14256},
  year={2024}
}
```
