from PIL import Image
import numpy as np
import albumentations as A
import torchvision.transforms as T
import torchvision
import cv2

transform = A.Compose([
                    A.ShiftScaleRotate(p=1, shift_limit=0.03, scale_limit=0.05, rotate_limit=10, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.ISONoise(p=0.5),
                    A.GaussNoise(p=0.8, var_limit=(200, 6000)),
                    A.HorizontalFlip(),

                ], bbox_params=A.BboxParams(format='coco', label_fields=['category_id']),)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        image_processor,
        annotation_file_path,
        transforms=None,  # Add transforms parameter
        train: bool = True,

    ):
        self._transforms = transforms
        # annotation_file_path =  ANNOTATION_FILE_NAME
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]

        
        bboxes = [anno['bbox'] for anno in annotations]
        category_ids = [anno['category_id'] for anno in annotations]

        if self._transforms:
            transformed = self._transforms(
                image=np.array(images),
                bboxes=bboxes,
                category_id=category_ids
            )
            images = Image.fromarray(transformed['image'])
            bboxes = transformed['bboxes']
            category_ids = transformed['category_id']
            for i, anno in enumerate(annotations):
                anno['bbox'] = bboxes[i]
                anno['category_id'] = category_ids[i]
        annotations = {'image_id': image_id, 'annotations': annotations}


        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target