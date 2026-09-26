import torch
import json
import pickle
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')
from model.attention_decoder import AttentionCaptionModel

device = torch.device('cuda')
checkpoint = torch.load('weights/decoder_torch.pt', map_location=device)
vocab_size = checkpoint['config']['vocab_size']
max_length = checkpoint['config']['max_length']

model = AttentionCaptionModel(vocab_size=8427, max_length=35).to(device)
model.train()

with open('weights/tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)

# Quick test forward pass
images = torch.randn(2, 3, 224, 224, device='cuda')
captions = torch.randint(0, 8427, (2, 35), device='cuda')

print('Testing forward pass...')
logits = model(images, torch.randint(0, 8427, (2, 35), device='cuda'))
print('Logits shape:', logits.shape)
print('Logits requires_grad:', logits.requires_grad)
print('Logits grad_fn:', logits.grad_fn)

# Test loss
target = torch.randint(0, 8427, (2,), device='cuda')
criterion = torch.nn.CrossEntropyLoss()
loss = torch.nn.functional.cross_entropy(logits, target)
print('Loss:', loss.item())
print('Loss requires_grad:', loss.requires_grad)
print('Loss grad_fn:', loss.grad_fn)

# Test backward
loss.backward()
print('Backward OK!')