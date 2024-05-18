from imp import reload
import gradio as gr
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import os
import json
from utils.db_utils import query_data

BACKEND_URL = "http://localhost:8000"

section_labels = [
        "Void",
        "resistanceline",
        "feature",
        "bottom"
    ]

default_config = {
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

def infer(img, model_id):
    sections = []
    # pil_img = Image.open(r"C:\Users\boyif\Desktop\CL_for_ImageSegmentation\experience\data\train\label\0\0.png")
    # img = np.asarray(pil_img)
    print(img)

    url = f"{BACKEND_URL}/infer_image/{model_id}"

    payload = {'annotation': '2333'}
    files=[
        ('file',(os.path.basename(img), open(img,'rb'),))
    ]
    headers = {
        'Authorization': 'Basic cm9vdDpwYXNzd29yZA=='
    }

    # response = requests.request("POST", url, headers=headers, data=payload, files=files)
    # pil_img = Image.open(BytesIO(response.content))
    pil_img = Image.open(open(r"C:\Users\boyif\Desktop\CL_for_ImageSegmentation\experience\data\train\label\0\0.png","rb"))
    res_img = np.asarray(pil_img)
    for i,label in enumerate(section_labels):
        mask = res_img == i
        mask = np.where(mask, 1, 0)
        sections.append((mask, label))
    return (img, sections)


def upload(filepath, anno):
    url = f"{BACKEND_URL}/upload-zip/"

    payload = {'annotation': anno}
    files=[
    ('file',('dataset.zip',open(filepath,'rb'),'application/zip'))
    ]
    headers = {
    'Authorization': 'Basic cm9vdDpwYXNzd29yZA=='
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    gr.Info(response.text)


def train(config):
    url = f"{BACKEND_URL}/train_model/"

    payload = json.dumps(config)
    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Basic cm9vdDpwYXNzd29yZA=='
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    gr.Info(response.text)



def dataset_update():
    df = query_data("select * from dataset")
    df['update_time'] = df['update_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


def model_update():
    df = query_data("select * from model")
    df['update_time'] = df['update_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


with gr.Blocks() as demo:
    
    gr.Markdown("# Simple Demo for CL_ImageSegmentation")
    gr.Markdown("If you have any question on this demo, please contact [boyifan1@126.com](boyifan1@126.com)")

    gr.Markdown("## Part Ⅰ: Image Segmentation inference")
    with gr.Row():    
        img_input = gr.Image(type="filepath")
        img_output = gr.AnnotatedImage(label="Annotated")

    with gr.Row():
        model_id = gr.Number(label="model_id")
        section_btn = gr.Button("Segement Image")
    section_btn.click(infer, [img_input, model_id], img_output)


    gr.Markdown("## Part Ⅱ: Dataset Upload")
    with gr.Row():
        with gr.Column(scale=1):
            dataset_input = gr.File(type="filepath",file_count="single")
            dataset_ann = gr.Textbox(label="Dataset Annotation")
        dataset_list = gr.Dataframe(label="Dataset List", value=dataset_update, every=5,scale=2)
    dataset_btn = gr.Button("Upload Dataset")
    dataset_btn.click(upload, [dataset_input, dataset_ann])

    gr.Markdown("## Part Ⅲ: Model Training")
    with gr.Row():
        with gr.Column(scale=1):
            model_input = gr.Textbox(label="Config", value=json.dumps(default_config))
        model_list = gr.Dataframe(label="Dataset List", value=model_update, every=5,scale=2)
    model_btn = gr.Button("Assign Training Task")
    model_btn.click(train, model_input)

if __name__ == "__main__":
    demo.launch(debug=True,auth=("root","password"))