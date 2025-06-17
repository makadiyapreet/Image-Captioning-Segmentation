from pycocotools.coco import COCO
from collections import Counter
import nltk
import json

def build_vocab(ann_file, threshold=5):
    coco = COCO(ann_file)
    counter = Counter()
    for ann in coco.anns.values():
        caption = ann['caption'].lower()
        tokens = nltk.tokenize.word_tokenize(caption)
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    for i, word in enumerate(words, 4):
        vocab[word] = i

    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    return vocab

vocab = build_vocab('/Volumes/ExternalHD/ICP/coco/annotations/captions_train2017.json')
