# Install & Setup
code need cuda installed, and can't run properly in CPU env!! 
## ENV setup

```shell
conda create -n cl_env python=3.8
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install avalanche-lib[all]
pip install fastapi[all]
pip install minio
pip install psycopg2-binary
pip install sqlalchemy
pip install pandas
pip install celery
pip install redis
```