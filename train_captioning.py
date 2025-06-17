import torch
import torch.nn as nn
from torch.optim import Adam
from src.data_loader import CocoDataset, transform
from src.models.captioning_model import CaptioningModel
from src.vocabulary import build_vocab
from torch.utils.data import DataLoader

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
vocab = build_vocab('/Volumes/ExternalHD/ICP/coco/annotations/captions_train2017.json')
dataset = CocoDataset(
    root='/Volumes/ExternalHD/ICP/coco/images/train2017',
    ann_file='/Volumes/ExternalHD/ICP/coco/annotations/captions_train2017.json',
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = CaptioningModel(embed_size=256, hidden_size=512, vocab_size=len(vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for images, captions, _ in dataloader:
        images = images.to(device)
        captions = torch.tensor([[vocab.get(word, vocab['<unk>']) for word in cap] for cap in captions]).to(device)
        outputs = model(images, captions)
        loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    torch.save(model.state_dict(), f'/Volumes/ExternalHD/ICP/models/captioning_epoch_{epoch}.pth')
