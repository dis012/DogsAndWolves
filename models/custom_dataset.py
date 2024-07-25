from torch.utils.data import Dataset
from PIL import Image
import os

# Custom dataset
# Data is organized as:
#archive/
#   data/
#       dogs/
#           img1.jpg
#           ...
#       wolves/
#           img1.jpg
#           ...       

class DogsAndWolvesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['dogs', 'wolves']
        self.img_paths = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_dir):
                self.img_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(idx) # Label 0 for dog and 1 for wolf

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label