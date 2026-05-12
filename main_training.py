from ultralytics import YOLO
import torch

def train_segmentation_model( dataset_path:str):
    
    print(torch.cuda.is_available())
    
    model = YOLO('yolov8n-seg.pt')

    results = model.train(
        data=dataset_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        optimizer='AdamW',
        freeze=12,
        lr0=1e-3,
        cos_lr=True,
        close_mosaic=15,
        patience=20,
        project='aimodel/runs/training',
        name='finetune-v0'
    )
    print("Training complete!")


train_segmentation_model('newtypeofdataset/data.yaml')