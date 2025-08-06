import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
import timm
import torch._dynamo

# --- Configuration ---
# Path for the best model checkpoint (saved at end of epoch)
best_checkpoint_path = 'model_checkpoint_best.pth'
# Path for interim checkpoints (saved during an epoch)
interim_checkpoint_path = 'model_checkpoint_interim.pth'
# Final model name after training is complete
output_model_name = 'magic_card_classifier_efficientnet_v2.pth'
# Path to your training data
training_folder = "/mnt/e/Photos/TableStream/training_images"

# --- Training Parameters ---
# Adjust these based on your specific hardware and dataset
BATCH_SIZE = 128
# The number of CPU cores to use for data loading.
NUM_WORKERS = 4
# Total number of epochs to train for
NUM_EPOCHS = 4
# Learning rate for the optimizer
LEARNING_RATE = 0.001
# How often to save an interim checkpoint (in number of batches)
SAVE_FREQ_BATCHES = 100
# The specific EfficientNetV2 model from timm to use.
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
        torch.backends.cudnn.benchmark = True

    # --- Data Augmentation and Loading ---
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Initializing datasets...")
    train_dataset = datasets.ImageFolder(os.path.join(training_folder, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(training_folder, 'val'), transform=val_transforms)
    print("Datasets initialized.")

    print("Initializing DataLoaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    print("DataLoaders initialized.")

    num_classes = len(train_dataset.classes)
    print(f'Found {num_classes} classes.')

    # --- Class Weighting ---
    print("Calculating class weights...")
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
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    print("Attempting to compile the model with torch.compile()...")
    model = torch.compile(model)
    print("Model compiled successfully.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # --- Checkpoint Loading ---
    start_epoch = 0
    start_batch = 0
    best_val_accuracy = 0.0
    # Prioritize loading the interim checkpoint for most recent progress
    if False and os.path.isfile(interim_checkpoint_path):
        print(f"Loading interim checkpoint '{interim_checkpoint_path}'...")
        checkpoint = torch.load(interim_checkpoint_path)
        # We only care about the batch index when loading an interim checkpoint
        start_batch = checkpoint.get('batch_idx', 0) + 1
    elif os.path.isfile(best_checkpoint_path):
        print(f"Loading best checkpoint '{best_checkpoint_path}'...")
        checkpoint = torch.load(best_checkpoint_path)
    else:
        checkpoint = None
        print("No checkpoint found, starting training from scratch.")

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        print(f"Resuming training from epoch {start_epoch + 1} with best accuracy {best_val_accuracy:.4f}...")


     # --- Training Loop ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        model.train()
        running_loss = 0.0
        
        # Set the start batch for the tqdm progress bar
        train_loader_tqdm = tqdm(
            enumerate(train_loader), 
            desc=f'Training Epoch {epoch+1}/{NUM_EPOCHS}', 
            total=len(train_loader),
            initial=start_batch, # Start the progress bar at the correct batch
            leave=False
        )

        for batch_idx, (inputs, labels) in train_loader_tqdm:
            # --- FIX: Skip batches if resuming from an interim checkpoint ---
            if batch_idx < start_batch:
                continue
            # --- Reset start_batch after the first iteration to avoid skipping in future epochs ---
            if start_batch > 0:
                print(f"\nResumed training at batch {start_batch}.")
                start_batch = 0

            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())
            
            # --- Save Interim Checkpoint ---
            if (batch_idx + 1) % SAVE_FREQ_BATCHES == 0:
                print(f"\nSaving interim checkpoint at epoch {epoch+1}, batch {batch_idx+1}...")
                interim_state = {
                    'epoch': epoch,
                    'batch_idx': batch_idx, # Save the current batch index
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_accuracy': best_val_accuracy,
                }
                torch.save(interim_state, interim_checkpoint_path)
                print("Interim checkpoint saved.")


        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch {epoch + 1} Training Loss: {epoch_loss:.4f}')

        # --- Validation phase ---
        model.eval()
        correct = 0
        total = 0
        
        val_loader_tqdm = tqdm(val_loader, desc='Validation', leave=False)

        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    outputs = model(inputs)

                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch + 1} Validation Accuracy: {accuracy:.4f}')

        # --- Save Best Checkpoint ---
        if accuracy > best_val_accuracy:
            print(f"Validation accuracy improved from {best_val_accuracy:.4f} to {accuracy:.4f}. Saving best checkpoint...")
            best_val_accuracy = accuracy
            best_state = {
                'epoch': epoch, # Save the completed epoch
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': accuracy,
                'best_val_accuracy': best_val_accuracy,
            }
            torch.save(best_state, best_checkpoint_path)
        else:
            print(f"Validation accuracy did not improve from {best_val_accuracy:.4f}.")


    print("\nTraining complete.")
    if os.path.isfile(best_checkpoint_path):
        print(f"Saving final model from best checkpoint to '{output_model_name}'...")
        final_checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(final_checkpoint['model_state_dict'])
        torch.save(model.state_dict(), output_model_name)
        print("Model saved successfully.")
    else:
        print("No best checkpoint was saved. Saving the current model state as final.")
        torch.save(model.state_dict(), output_model_name)
        print("Model saved successfully.")


if __name__ == '__main__':
    train_model()
