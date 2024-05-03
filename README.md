# CL_for_ImageSegmentation

[![Scc Count Badge](https://sloc.xyz/github/KirigiriSuzumiya/CL_for_ImageSegmentation/)](https://github.com/KirigiriSuzumiya/CL_for_ImageSegmentation/)
[![License](https://img.shields.io/npm/l/sloc.svg)](https://github.com/KirigiriSuzumiya/CL_for_ImageSegmentation/blob/master/LICENCE.txt)
[![Minified size](http://img.shields.io/badge/size-6,1K-blue.svg)](https://github.com/KirigiriSuzumiya/CL_for_ImageSegmentation)

repo for applying Continual Learning on Image Segmentation task.

Mainly consist of two parts:

- experience
  - experience on `NAIVE`, `EWC`, `LFL`, `SI` and `GEM`
  - report available on wandb [link](https://api.wandb.ai/links/kirigiri_suzumiya/nrn4i0as)
- software engineering
  - a backend system for CL on Image Segmentation using UNet as basic model
  - `Fastapi` for restful api
  - `Celery & Redis` for Distributed Task Queue
  - `Minio & Postgresql` for dataset&checkpoint storage


## Introduction

Three scenarios of **Continual Learning**

- **Task-incremental learning:** Sequentially learn to solve a number of distinct tasks
- **Domain-incremental learning:** Learn to solve the same problem in different contexts
- **Class-incremental learning**: Discriminate between incrementally observed classes

Our specific task is obviously not beyond the field of **Domain-incremental learning**


## CL Learning Strategies Evaluation

Evaluation is based on U-Net and  [Avalanche](https://github.com/ContinualAI/avalanche). 

please refer to [experience](./experience/) for notebook and more details

report available on wandb [link](https://api.wandb.ai/links/kirigiri_suzumiya/nrn4i0as)

### Batch Domain Continual Learning

- Less-Forgetful Learning (LFL):  [paper](https://arxiv.org/pdf/1607.00122.pdf) | [pdf](./reference/1607.00122.pdf)
  - Less-forgetting Learning in Deep Neural Networks
  - `Jung H, Ju J, Jung M, et al. Less-forgetting learning in deep neural networks[J]. arXiv preprint arXiv:1607.00122, 2016.`

- Synaptic Intelligence (SI): [paper](http://proceedings.mlr.press/v70/zenke17a.html) | [pdf](./reference/ContinualLearningThroughSynapticIntelligence.pdf)
  - Continual Learning Through Synaptic Intelligence
  - `Zenke F, Poole B, Ganguli S. Continual learning through synaptic intelligence[C]//International conference on machine learning. PMLR, 2017: 3987-3995.`
- Elastic Weight Consolidation (EWC): [paper](https://www.pnas.org/content/114/13/3521) | [pdf](./reference/kirkpatrick-et-al-2017-overcoming-catastrophic-forgetting-in-neural-networks.pdf)
  - Overcoming catastrophic forgetting in neural networks
  - `Kirkpatrick J, Pascanu R, Rabinowitz N, et al. Overcoming catastrophic forgetting in neural networks[J]. Proceedings of the national academy of sciences, 2017, 114(13): 3521-3526.`

### Online Domain Continual Learning

- Gradient Episodic Memory (GEM): [paper](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html) | [pdf](./reference/NIPS-2017-gradient-episodic-memory-for-continual-learning-Paper.pdf)
  - Gradient Episodic Memory for Continual Learning
  - `Lopez-Paz D, Ranzato M A. Gradient episodic memory for continual learning[J]. Advances in neural information processing systems, 2017, 30.`

### Others

- Replay Buffer
- Selection strategies
- Loss Functions
- ...

## SW engineering

please refer to [sw_service](./sw_service/) for code and more details

![img](assets/structure.png)


## ENV & Setup

please refer to [setup](./setup.md) for more details

## Contact me

email: `boyifan1@126.com`