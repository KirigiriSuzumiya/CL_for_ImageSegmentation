# celery_worker.py
import os
import shutil
import time
from celery import Celery
from sqlalchemy import text
from PIL import Image
from io import BytesIO
import numpy as np
from avalanche.checkpointing import save_checkpoint, maybe_load_checkpoint
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.logging import InteractiveLogger, WandBLogger

from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.evaluation.metrics import loss_metrics, timing_metrics
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.utils.data_attribute import DataAttribute

import torch
from torch import optim
from minio import Minio
import wandb
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unet import UNet
from utils.train_policy import get_strategy
from utils.data_loading import BasicDataset
from utils.db_utils import Session

app = Celery('celery_app', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')


# MinIO配置
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "root"
MINIO_SECRET_KEY = "password"
MINIO_USE_SSL = False


@app.task
def cl_training(strategy_path, dataset_path, config, task_id):
    RNGManager.set_random_seeds(1)


    # build model & logger
    model = UNet(n_channels=3, n_classes=4)

    func_name = f"{task_id}_{config['cl_strategy']}"
    wandb_logger = WandBLogger(project_name="UNet_CL_app",
                           run_name=func_name,
                           path=f"./tmp/log/checkpoint/{func_name}",
                           dir=f"./tmp/log/wandb/{func_name}")
    interactive_logger = InteractiveLogger()

    # CREATE THE STRATEGY INSTANCE
    optimizer = optim.RMSprop(model.parameters(),
                            lr=config["learning_rate"], 
                            weight_decay=config["weight_decay"], 
                            momentum=config["momentum"], 
                            foreach=True)

    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        loggers=[interactive_logger, 
                 wandb_logger
                 ]
    )

    cl_strategy = get_strategy(
        model=model, 
        optimizer=optimizer, 
        eval_plugin=eval_plugin, 
        config=config)

    # load checkpoint if exists
    if strategy_path:
        cl_strategy, _ = maybe_load_checkpoint(strategy=cl_strategy, fname=strategy_path)


    # prepare dataset
    train_dataset = BasicDataset(images_dir=dataset_path+"/img",
                                mask_dir=dataset_path+"/label")
    aval_dataset = AvalancheDataset([train_dataset],
                                    data_attributes=[
                                                    DataAttribute([0] * len(train_dataset), "targets_task_labels"),
                                                    ])

    bm = benchmark_from_datasets(train=[aval_dataset])

    print('Starting experiment...')
    for i, experience in enumerate(bm.train_stream):
        cl_strategy.train(experience)
        print('Training completed')
    wandb.finish()


    out_ckpt_path = os.path.dirname(dataset_path)+f"/{time.time()}.ckpt"
    save_checkpoint(strategy=cl_strategy, fname=out_ckpt_path)

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_USE_SSL
    )
    with open(out_ckpt_path,"rb") as f:
        minio_client.fput_object(
                "model",
                os.path.basename(out_ckpt_path),
                out_ckpt_path,
            )
    

    shutil.rmtree(os.path.dirname(out_ckpt_path))
    query = f"""UPDATE model
    SET minio_path = '{os.path.basename(out_ckpt_path)}', status='success'
    WHERE task_id='{task_id}'
    """
    
    # 创建Session实例
    with Session() as session:
        session.execute(text(query))
    return task_id


@app.task
def infer(bimg, strategy_path, config, task_id):
    RNGManager.set_random_seeds(1)


    # build model & logger
    model = UNet(n_channels=3, n_classes=4)

    func_name = f"{task_id}_{config['cl_strategy']}"
    wandb_logger = WandBLogger(project_name="UNet_CL_app",
                           run_name=func_name,
                           path=f"./tmp/log/checkpoint/{func_name}",
                           dir=f"./tmp/log/wandb/{func_name}")
    interactive_logger = InteractiveLogger()

    # CREATE THE STRATEGY INSTANCE
    optimizer = optim.RMSprop(model.parameters(),
                            lr=config["learning_rate"], 
                            weight_decay=config["weight_decay"], 
                            momentum=config["momentum"], 
                            foreach=True)

    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        loggers=[interactive_logger, wandb_logger]
    )

    cl_strategy = get_strategy(
        model=model, 
        optimizer=optimizer, 
        eval_plugin=eval_plugin, 
        config=config)
    
    # load checkpoint if exists
    if strategy_path:
        cl_strategy,_ = maybe_load_checkpoint(strategy=cl_strategy, fname=strategy_path)
    
    pil_img = Image.open(BytesIO(bimg))
    img = np.asarray(pil_img)
    img = img.transpose((2, 0, 1))
    if (img > 1).any():
        img = img / 255.0
    img = np.array([img])
    torch_img = torch.from_numpy(img).float().to('cuda:0')
    print(torch_img.shape)
    with torch.no_grad():
        out_tensor = cl_strategy.model(torch_img).cpu()
        argmax_output = torch.argmax(out_tensor, dim=1, keepdim=True) 
        img_arr = np.array(argmax_output)[0].transpose((1,2,0))
        img_arr = img_arr.reshape(img_arr.shape[:-1])
        out_img_path = os.path.dirname(strategy_path)+"/out.png"
        Image.fromarray(img_arr,"L").save(out_img_path)
    return out_img_path