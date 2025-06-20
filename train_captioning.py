import os
import torch
import torch.nn as nn
from torch.optim import Adam
from src.data_loader import CocoDataset, transform, collate_fn
from src.models.captioning_model import CaptioningModel
from src.vocabulary import build_vocab
from torch.utils.data import DataLoader

device = torch.device('cpu')  # Use CPU due to 8GB RAM
vocab = build_vocab('/Volumes/ExternalHD/ICP/coco/annotations/captions_train2017.json')
dataset = CocoDataset(
    root='/Volumes/ExternalHD/ICP/coco/images/train2017',
    ann_file='/Volumes/ExternalHD/ICP/coco/annotations/captions_train2017.json',
    transform=transform,
    vocab=vocab
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

model = CaptioningModel(embed_size=128, hidden_size=256, vocab_size=len(vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
optimizer = Adam(model.parameters(), lr=0.001)

# Checkpoint loading
model_dir = '/Volumes/ExternalHD/ICP/models/'
os.makedirs(model_dir, exist_ok=True)
start_epoch = 0
latest_checkpoint = None
# Find the latest checkpoint
for epoch in range(4, -1, -1):  # Check from epoch 4 down to 0
    checkpoint_path = f'{model_dir}captioning_epoch_{epoch}.pth'
    if os.path.exists(checkpoint_path):
        latest_checkpoint = checkpoint_path
        start_epoch = epoch + 1
        break

if latest_checkpoint:
    print(f'Resuming from {latest_checkpoint}')
    model.load_state_dict(torch.load(latest_checkpoint))
else:
    print('Starting from scratch')

total_steps = len(dataloader)
for epoch in range(start_epoch, 5):
    model.train()
    for i, (images, captions) in enumerate(dataloader):
        caption_indices = torch.tensor([[vocab.get(word, vocab['<unk>']) for word in cap] for cap in captions]).to(device)
        images = images.to(device)
        outputs = model(images, caption_indices[:, :-1])
        loss = criterion(outputs[:, 1:].contiguous().view(-1, len(vocab)), 
                         caption_indices[:, 1:-1].contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/5, Step {i+1}/{total_steps}, Loss: {loss.item()}')
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    torch.save(model.state_dict(), f'{model_dir}captioning_epoch_{epoch}.pth')
