import albumentations as A
import cv2
import random

def build_transforms(args):
    data_transforms = {
        "train": A.Compose([
            A.OneOf([
                A.Resize(*args.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
                # A.Normalize(mean=[ 54.921,  76.704, 126.565], std=[40.818, 52.777, 75.755])
                # A.Normalize(mean=[54.921, 76.704, 126.565], std=[40.818, 52.777, 75.755])
            ], p=1),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=None, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=args.img_size[0] // 20, max_width=args.img_size[1] // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),


        "valid_test": A.Compose([
            A.Resize(*args.img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }
    return data_transforms


