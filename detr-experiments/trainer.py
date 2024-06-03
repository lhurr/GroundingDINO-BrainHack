import json
import os
from PIL import Image
import os
import torchvision
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import albumentations as A
import numpy as np
import cv2
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
from coco import CocoDetection

input_file = 'advanced/vlm.jsonl'
image_dir = 'advanced/images/'
output_file = 'path/to/save/coco_format.json'

# Function to calculate area of a bounding box
def calculate_area(bbox):
    return bbox[2] * bbox[3]


images = []
annotations = []
categories = []
unique_categories = set()
image_id = 0
annotation_id = 0

# First pass to collect unique categories
with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        for annotation in data['annotations']:
            caption = annotation['caption'].split(' ')[-1]
            unique_categories.add(caption)

# Create a category to ID mapping
category_mapping = {category: idx +1 for idx, category in enumerate(sorted(unique_categories))}

label2id = category_mapping
id2label = {l:i for i,l in label2id.items()}


for category, category_id in category_mapping.items():
    categories.append({
        "id": category_id,
        "name": category
    })


with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        image_filename = data['image']
        image_id += 1

        image_path = os.path.join(image_dir, image_filename)

        width = 1520
        height = 870
        # Add image info to images list
        images.append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        for annotation in data['annotations']:
            bbox = annotation['bbox']
            area = calculate_area(bbox)
            category_id = category_mapping[annotation['caption'].split(' ')[-1]]

            # Add annotation info to annotations list
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0  # Assuming all annotations are not crowd annotations
            })
            annotation_id += 1


coco_format = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Save the COCO format dictionary to a JSON file
with open('coco_format.json', 'w') as f:
    json.dump(coco_format, f, indent=4)


# model settings
MAX_EPOCHS = 50
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-101'

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
model.to(DEVICE)

# data settings
ANNOTATION_FILE_NAME = "coco_format.json"
TRAIN_DIRECTORY = 'advanced/images'

TRAIN_DATASET = CocoDetection(
    image_directory_path=TRAIN_DIRECTORY,
    image_processor=image_processor,
    annotation_file_path = ANNOTATION_FILE_NAME,
    train=True,
    transforms = False
)


def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible
    # to directly batch together images. Hence they pad the images to the biggest
    # resolution in a given batch, and create a corresponding binary pixel_mask
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET,num_workers=4, collate_fn=collate_fn, batch_size=1, shuffle=True)



class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER
    def val_dataloader(self):
        return TRAIN_DATALOADER



model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

trainer.fit(model)