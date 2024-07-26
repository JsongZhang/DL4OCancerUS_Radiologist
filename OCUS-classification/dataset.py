import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Retrieve all image files
        self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if f.endswith('.jpg')]
        # Extract unique class names from the folders
        self.classes = sorted(set(img.split(os.sep)[-2] for img in self.images))
        # Create a map of class names to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert('RGB')  # Ensure image is RGB
        if self.transform:
            image = self.transform(image)

        # Extract label from the directory name
        label = self.class_to_idx[img_name.split(os.sep)[-2]]

        return image, torch.tensor(label)  # Ensure label is a tensor

# You can define specific transforms here.
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop(300),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    # transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])
val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

train_dataset = ImageDataset(root_dir='/data1/zhangjiansong/py_project/Hosptial_other_project/Fujian_2/Doc_WangYanli/data/OvarianCancerUS/train/', transform=train_transform)
val_dataset = ImageDataset(root_dir='/data1/zhangjiansong/py_project/Hosptial_other_project/Fujian_2/Doc_WangYanli/data/OvarianCancerUS/val/', transform=val_transform)
test_dataset = ImageDataset(root_dir='/data1/zhangjiansong/py_project/Hosptial_other_project/Fujian_2/Doc_WangYanli/data/OvarianCancerUS/External_test_0619', transform=test_transform)
