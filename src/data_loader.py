import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T
import nltk
nltk.download('punkt')

class CocoDataset(Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.root = root
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.root}/{img_info['file_name']}"
        img = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        captions = [ann['caption'] for ann in self.coco.loadAnns(ann_ids)]
        caption = captions[0]
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = ['<start>'] + tokens + ['<end>']

        mask = self.coco.annToMask(self.coco.loadAnns(ann_ids)[0])

        if self.transform:
            img = self.transform(img)

        return img, caption, mask

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CocoDataset(
    root='/Volumes/ExternalHD/ICP/coco/images/train2017',
    ann_file='/Volumes/ExternalHD/ICP/coco/annotations/captions_train2017.json',
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
