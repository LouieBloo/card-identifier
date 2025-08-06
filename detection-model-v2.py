import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
import timm

# --- Configuration ---
# Path to save and load model checkpoints
checkpoint_path = 'model_checkpoint_v2.pth'
# Final model name after training is complete
output_model_name = 'magic_card_classifier_efficientnet_v2.pth'
# Path to your training data
training_folder = "/mnt/e/Photos/TableStream/training_images"

# --- Training Parameters ---
# Adjust these based on your specific hardware and dataset
# Using a larger batch size can speed up training if your GPU has enough memory
BATCH_SIZE = 256
# The number of CPU cores to use for data loading.
# Increase this if you notice a bottleneck in data preprocessing.
NUM_WORKERS = 8
# Total number of epochs to train for
NUM_EPOCHS = 8
# Learning rate for the optimizer
LEARNING_RATE = 0.001
# The specific EfficientNetV2 model from timm to use.
# 'tf_efficientnetv2_s.in21k_ft_in1k' is a solid choice with good performance,
# pretrained on ImageNet-21k and fine-tuned on ImageNet-1k.
MODEL_NAME = 'tf_efficientnetv2_s.in21k_ft_in1k'

# --- Main Script ---

def train_model():
    """
    Main function to set up and run the training process.
    """
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU will be very slow.")
        device = torch.device('cpu')
    else:
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
        # Enabling allows cudnn to benchmark and find the best algorithms for your hardware
        torch.backends.cudnn.benchmark = True

    # --- Data Augmentation and Loading ---
    # Define separate transforms for training and validation.
    # Training transforms include augmentation to make the model more robust.
    train_transforms = transforms.Compose([
        # More robust than Resize, as it handles different aspect ratios
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), # A good auto-augmentation policy
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation transforms should not include random augmentations.
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(os.path.join(training_folder, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(training_folder, 'val'), transform=val_transforms)

    # Create DataLoaders
    # persistent_workers=True avoids re-initializing workers every epoch, saving time.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    num_classes = len(train_dataset.classes)
    print(f'Found {num_classes} classes.')

    # --- Class Weighting for Imbalanced Datasets ---
    print("Calculating class weights for imbalanced dataset...")
    targets = [s[1] for s in train_dataset.samples]
    class_counts = Counter(targets)
    total_samples = len(train_dataset)
    class_weights = torch.tensor(
        [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)
    print("Class weights calculated.")

    # --- Model Initialization ---
    print(f"Initializing model: {MODEL_NAME}")
    # Use timm.create_model to get the EfficientNetV2 model.
    # `pretrained=True` loads weights from ImageNet.
    # `num_classes` automatically replaces the final classifier layer.
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # --- MAJOR SPEEDUP: torch.compile() ---
    # This compiles the model into a faster, optimized version for your hardware.
    # It can provide a 20-30%+ speedup.
    # Note: Requires PyTorch 2.0+
    try:
        print("Compiling the model with torch.compile()...")
        model = torch.compile(model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Could not compile the model: {e}. Continuing without compilation.")


    # --- Loss Function, Optimizer, and Scaler ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # GradScaler for automatic mixed-precision (AMP) training.
    # This uses float16 to speed up computations and reduce memory usage.
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # --- Checkpoint Loading ---
    start_epoch = 0
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}...")
    else:
        print("No checkpoint found, starting training from scratch.")

    # --- Training Loop ---
    best_val_accuracy = 0.0
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # Training phase
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc='Training', leave=False)

        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Use autocast for mixed-precision operations
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Scaler handles the backward pass and optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch {epoch + 1} Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loader_tqdm = tqdm(val_loader, desc='Validation', leave=False)

        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    outputs = model(inputs)

                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch + 1} Validation Accuracy: {accuracy:.4f}')

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'accuracy': accuracy,
        }
        # Save the checkpoint only if validation accuracy improves
        if accuracy > best_val_accuracy:
            print(f"Validation accuracy improved from {best_val_accuracy:.4f} to {accuracy:.4f}. Saving checkpoint...")
            torch.save(checkpoint, checkpoint_path)
            best_val_accuracy = accuracy
        else:
            print(f"Validation accuracy did not improve from {best_val_accuracy:.4f}.")


    print("\nTraining complete.")
    print(f"Saving final model to '{output_model_name}'...")
    # To save the final model, we can load the best checkpoint and save its state_dict
    final_checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(final_checkpoint['model_state_dict'])
    torch.save(model.state_dict(), output_model_name)
    print("Model saved successfully.")


if __name__ == '__main__':
    train_model()
