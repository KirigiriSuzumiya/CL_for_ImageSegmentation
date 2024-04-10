# ENV setup

```shell
conda create -n cl_env python=3.8
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install avalanche-lib[all]
```