import numpy as np
from pathlib import Path
from torchvision import transforms
import torch
from PIL import Image


def transform_image(img_path):
    
    img = Image.open(img_path, mode='r').convert('RGB')
    
    trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # Imagenet standards
    ])
    
    return trans(img)


def predict_heartbeat(model, img_path, DIR_ROOT):

    # Load model into memory
    model = torch.load(Path(DIR_ROOT, 'model', model + '.pt'), 
                       map_location=torch.device('cpu')
            )

    # Image transformation
    img = transform_image(img_path)

    # Output + Prediction
    output = model(img.unsqueeze(0))
    predict = output.data.cpu().numpy().argmax()

    return output, predict