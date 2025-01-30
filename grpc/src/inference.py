from typing import List
import torch
import torchvision.transforms as T
from PIL import Image    #handles images in python
from torchvision.models import ResNet34_Weights, resnet34

preprocess = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()


@torch.no_grad()  #disable gradient calculation

def inference(images: List[Image.Image]) -> List[int]:#specifies the input to the func is a list of imgs and output is a list of integers
    batch = torch.stack([preprocess(image) for image in images])# Preprocess all images and stack them into a batch
    logits = model(batch)# Forward pass: input the batch into the model
    preds = logits.argmax(dim=1).tolist()# Get the predicted class index for each image in the batch
    return preds


if __name__ == "__main__":
    image = Image.open("./examples/cats.jpeg")
    print(inference([image]))