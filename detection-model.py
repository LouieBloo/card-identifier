from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from collections import Counter

# Define the path to save checkpoints
checkpoint_path = 'model_checkpoint.pth'
output_model_name = 'magic_card_classifier_v3.pth'

# Initialize variables
start_epoch = 0
num_epochs = 10  # Total number of epochs you want to train

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('classified_images/train', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

val_dataset = datasets.ImageFolder('classified_images/val', transform=train_transforms)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

num_classes = len(train_dataset.classes)

print(f'Class Amount: {num_classes}')

# Compute class counts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

targets = [s[1] for s in train_dataset.samples]
class_counts = Counter(targets)
total_samples = len(train_dataset)
class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
class_weights = torch.tensor(class_weights).to(device)


model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, num_classes)
model = model.cuda()
model = model.to(device)

#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if a checkpoint exists
if os.path.isfile(checkpoint_path):
    print(f"Loading checkpoint '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    print(f"Resuming training from epoch {start_epoch}...")
else:
    print("No checkpoint found, starting training from scratch.")

# Move model to GPU if available
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model.to(device)

# Optionally, initialize scaler for mixed precision training
scaler = torch.amp.GradScaler('cuda')

for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0

    train_loader_tqdm = tqdm(train_loader, desc='Training', leave=False)
    for inputs, labels in train_loader_tqdm:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        # Update progress bar with current loss
        train_loader_tqdm.set_postfix(loss=loss.item())
    
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
    
    # Validation Phase
    model.eval()
    correct = 0
    total = 0

    val_loader_tqdm = tqdm(val_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for inputs, labels in val_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
    
    accuracy = correct.double() / total
    print(f'Validation Accuracy: {accuracy:.4f}')
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'accuracy': accuracy.item(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

print("saving model...")
torch.save(model.state_dict(), output_model_name)
print("Training complete.")