from pycocotools.coco import COCO
coco = COCO('/Volumes/ExternalHD/ICP/coco/annotations/captions_train2017.json')
print(f"Number of images: {len(coco.imgs)}")
