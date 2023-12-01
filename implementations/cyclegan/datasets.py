import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train", train_val_split=0.2):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.root = root
        self.mode = mode
        self.train_val_split = train_val_split # portion of validation
        self.files_A, self.files_B = self.preprocess()

    def preprocess(self):
        files_A = sorted(glob.glob(os.path.join(self.root, "trainA") + "/*.*"))
        files_B = sorted(glob.glob(os.path.join(self.root, "trainB") + "/*.*"))
        length_A = len(files_A)
        length_B = len(files_B)

        if self.mode == "train":
            files_A = files_A[: int(length_A * (1-self.train_val_split))]
            files_B = files_B[: int(length_B * (1-self.train_val_split))]
        elif self.mode == "test":
            files_A = files_A[int(length_A * (1-self.train_val_split)): ]
            files_B = files_B[int(length_B * (1-self.train_val_split)): ]
        else:
            raise ValueError("mode undefined")
        
        return files_A, files_B

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
