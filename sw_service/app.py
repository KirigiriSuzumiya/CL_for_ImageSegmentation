from fastapi import Body, FastAPI, Depends, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from celery import signature
from typing import Dict
from minio import Minio
from io import BytesIO
import zipfile
import os
import shutil
import datetime
import time
import pandas as pd

from utils.auth import auth_username 
from utils.db_utils import update_data, query_data
from utils import BasicDataset
from celery_worker import cl_training, infer
import uuid


# MinIO配置
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "root"
MINIO_SECRET_KEY = "password"
MINIO_USE_SSL = False

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_USE_SSL
)

app = FastAPI()


@app.get("/ping")
def read_root():
    return {"msg": "Hello World!"}


@app.post("/upload-zip/")
async def upload_zip_file(file: UploadFile = File(...), 
                          annotation: str = Form(...),
                          username: str=Depends(auth_username)):
    if file.content_type != "application/zip":
        return {"error": "Only ZIP files are allowed."}

    # extract files from ZIP file
    tmp_path = './tmp/extracted_files'
    tmp_file = file.file.read()
    with zipfile.ZipFile(BytesIO(tmp_file), 'r') as zip_ref:
        zip_ref.extractall(path=tmp_path)

    # check if the extracted files are valid
    length = len(BasicDataset(images_dir=os.path.join(tmp_path,"img"),
                              mask_dir=os.path.join(tmp_path,"label")))
    shutil.rmtree(tmp_path)
    os.mkdir(tmp_path)

    # upload to MinIO
    minio_path = f"{time.time()}.zip"
    with BytesIO(tmp_file) as f:
        minio_client.put_object(
            "dataset",
            minio_path,
            f,
            len(f.getbuffer())
        )

    # update DB
    db_data = [{"filename":file.filename, 
               "size": length,
               "update_time":datetime.datetime.now(),
               "minio_path":minio_path,
               "annotation": annotation
               },]
    df = pd.DataFrame.from_dict(db_data)
    update_data("dataset",df)
    return {"message": "ZIP file is valid and uploaded to MinIO & DB."}


@app.post("/train_model/")
async def train_model(config: Dict[str, str] = Body(...),
                      username: str=Depends(auth_username)):
    # load default training parameters
    training_parameters = {
        'epochs': 1,
        'batch_size': 8,
        'learning_rate': 0.00001,
        'weight_decay': 0.00000001,
        'momentum': 0.999,
        'cl_strategy': "NAIVE",
        'ewc_lambda': 0.005,
        'lfl_lambda': 0.005,
        'si_lambda': 0.005,
        'patterns_per_exp': 10,
    }
    training_parameters.update(config)
    config = training_parameters
    task_id = uuid.uuid4()

    # extract dataset from MinIO
    if not config.get("dataset"):
        raise HTTPException(status_code=400, detail="dataset config is invalid")
    else:
        query = f"SELECT * FROM dataset WHERE id={config['dataset']}"
        df = query_data(query)
        if df.empty:
            raise HTTPException(status_code=400, detail="dataset not found")
        else:
            zip_path = f"./tmp/{task_id}/{df.iloc[0]['filename']}"
            extract_path = f"./tmp/{task_id}/dataset"
            minio_client.fget_object(
                "dataset", 
                df.iloc[0]["minio_path"], 
                zip_path,
                )
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(path=extract_path)
            os.remove(zip_path)

    model_path = ""
    if config.get("model"):
        query = f"SELECT * FROM model WHERE id={config['model']}"
        df = query_data(query)
        if df.empty or (not df.iloc[0]["minio_path"]):
            raise HTTPException(status_code=400, detail="model not found")
        else:
            model_path = f"./tmp/{task_id}/checkpoint"
            minio_client.fget_object(
                "model", 
                df.iloc[0]["minio_path"], 
                model_path,
                )
    db_data = [{"task_id":task_id, 
                "update_time":datetime.datetime.now(),
                "minio_path": "",
                "annotation": "created by fastapi",
                "dataset": config['dataset'],
                "status":"pending",
                "model_from": config.get("model"),
               },]
    task = cl_training.apply_async(args=[model_path, extract_path, config, task_id])
    df = pd.DataFrame.from_dict(db_data)
    update_data("model", df)
    return {"message": "Training task started",
            "task_id": task_id,
            "celery_id": task.id}


@app.post("/infer_image/{model_id}")
async def infer_image(model_id: str,
                      background_tasks: BackgroundTasks,
                      file: UploadFile = File(...), 
                      annotation: str = Form(...),
                      username: str=Depends(auth_username),
                      ):
    task_id = uuid.uuid4()
    if model_id:
        query = f"SELECT * FROM model WHERE id={model_id}"
        df = query_data(query)
        if df.empty or (not df.iloc[0]["minio_path"]):
            raise HTTPException(status_code=400, detail="model not found")
        else:
            model_path = f"./tmp/{task_id}/checkpoint"
            minio_client.fget_object(
                "model", 
                df.iloc[0]["minio_path"], 
                model_path,
                )
    else:
        raise HTTPException(status_code=400, detail="model must given")
    config = {
        'epochs': 1,
        'batch_size': 8,
        'learning_rate': 0.00001,
        'weight_decay': 0.00000001,
        'momentum': 0.999,
        'cl_strategy': "NAIVE",
        'ewc_lambda': 0.005,
        'lfl_lambda': 0.005,
        'si_lambda': 0.005,
        'patterns_per_exp': 10,
    }
    task = infer.apply_async(args=[file.file.read(), model_path, config, task_id])
    result = task.get()
    response = FileResponse(result)
    background_tasks.add_task(shutil.rmtree, os.path.dirname(result)) 
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)