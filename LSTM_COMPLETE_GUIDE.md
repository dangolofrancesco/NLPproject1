# Complete LSTM Implementation Guide

## ✅ What Has Been Implemented

### 1. **Preprocessing for LSTM** (in `preprocessing.py`)
- ✅ `GloveSequenceEmbedder` - Preserves word sequences (doesn't average)
- ✅ Tokenization using NLTK
- ✅ Padding to fixed length
- ✅ Sequence length tracking

### 2. **Dataset for LSTM** (in `dataset.py`)
- ✅ `RNNEmpathyDataset` - Training/validation dataset for LSTM
- ✅ `RNNInferenceDataset` - Test dataset for LSTM
- ✅ Returns sequences with padding and actual lengths

### 3. **LSTM Model** (in `lstm_model.py`)
- ✅ `LSTMEmpathyNet` - Complete LSTM implementation
- ✅ **Embeddings**: Pre-computed GloVe vectors (100-dim)
- ✅ **Recurrent Layers**: LSTM with bidirectional option
- ✅ **Task-Specific Heads**: 3 separate outputs (intensity, empathy, polarity)

---

## 🏗️ LSTM Architecture

```
Input: Pre-computed GloVe Embeddings
  Shape: (batch_size, seq_len, 100)
  
         ↓
         
LSTM Layers (2 layers, bidirectional)
  - Processes sequences left-to-right (and right-to-left if bidirectional)
  - Maintains hidden state and cell state
  - Captures temporal dependencies
  
         ↓
         
Last Hidden State
  - Extract final representation
  - Shape: (batch_size, hidden_dim * 2) if bidirectional
  
         ↓
         
Dropout (0.3)
  - Regularization
  
         ↓
         
Task-Specific Heads (3 parallel outputs):
  ├─→ Intensity Head (Linear) → Emotion intensity (regression)
  ├─→ Empathy Head (Linear) → Empathy score (regression)
  └─→ Polarity Head (Linear) → Polarity class (classification)
```

---

## 🚀 Complete Usage Example

### Step 1: Imports

```python
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import your custom classes
from preprocessing import GloveSequenceEmbedder
from dataset import RNNEmpathyDataset, RNNInferenceDataset
from lstm_model import LSTMEmpathyNet
```

### Step 2: Setup Device and Paths

```python
# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Data paths
DATA_PATH = "Dataset"
train_csv_path = f"{DATA_PATH}/trac2_CONVT_train.csv"
eval_csv_path = f"{DATA_PATH}/trac2_CONVT_dev.csv"
test_csv_path = f"{DATA_PATH}/trac2_CONVT_test.csv"

# Model save path
MODELS_SAVE_PATH = "Saved Models"
import os
os.makedirs(MODELS_SAVE_PATH, exist_ok=True)
```

### Step 3: Initialize Sequence Embedder

```python
# Create embedder that preserves sequences (NOT averaging!)
sequence_embedder = GloveSequenceEmbedder(
    model_name="glove-wiki-gigaword-100",
    max_length=50  # Adjust based on your data
)

print(f"Embedder vector size: {sequence_embedder.vector_size}")
print(f"Max sequence length: {sequence_embedder.max_length}")
```

### Step 4: Create Datasets and DataLoaders

```python
# Create datasets
print("\n--- Creating Training Dataset ---")
train_dataset = RNNEmpathyDataset(
    csv_path=train_csv_path,
    sequence_embedder=sequence_embedder
)

print("\n--- Creating Evaluation Dataset ---")
eval_dataset = RNNEmpathyDataset(
    csv_path=eval_csv_path,
    sequence_embedder=sequence_embedder
)

# Create DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n✅ DataLoaders created successfully")
print(f"   Training batches: {len(train_loader)}")
print(f"   Evaluation batches: {len(eval_loader)}")
```

### Step 5: Initialize LSTM Model

```python
# Model configuration
config = {
    'input_dim': sequence_embedder.vector_size,  # 100 (GloVe dimension)
    'hidden_dim': 128,                           # LSTM hidden size
    'num_layers': 2,                             # Number of LSTM layers
    'dropout': 0.3,                              # Dropout rate
    'bidirectional': True,                       # Bidirectional LSTM
    'num_classes': 4                             # Polarity classes (0-3)
}

# Create model
model = LSTMEmpathyNet(config).to(device)

# Print model information
print("\n" + "="*60)
print("LSTM Model Configuration")
print("="*60)
model_info = model.get_model_info()
for key, value in model_info.items():
    print(f"{key:20s}: {value}")
print("="*60)
```

### Step 6: Define Training Functions

```python
def train_one_epoch(model, dataloader, optimizer, loss_fns, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0.0
    
    # Loss weights for multi-task learning
    loss_weights = {'intensity': 0.2, 'empathy': 0.2, 'polarity': 0.6}
    
    for batch in dataloader:
        # Get data
        features = batch['features'].to(device)
        lengths = batch['length'].to(device)  # Important for LSTM!
        labels = {k: v.to(device) for k, v in batch['labels'].items()}
        
        # Forward pass - pass lengths to model
        outputs = model(features, lengths)
        
        # Calculate losses
        loss_intensity = loss_fns['regression'](outputs['intensity'], labels['intensity'])
        loss_empathy = loss_fns['regression'](outputs['empathy'], labels['empathy'])
        loss_polarity = loss_fns['classification'](outputs['polarity'], labels['polarity'])
        
        # Combined weighted loss
        combined_loss = (loss_weights['intensity'] * loss_intensity +
                         loss_weights['empathy'] * loss_empathy +
                         loss_weights['polarity'] * loss_polarity)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        
        total_loss += combined_loss.item()
    
    return total_loss / len(dataloader)


def evaluate_performance(model, dataloader, loss_fns, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0.0
    loss_weights = {'intensity': 0.2, 'empathy': 0.2, 'polarity': 0.6}
    
    all_intensity_preds, all_intensity_labels = [], []
    all_empathy_preds, all_empathy_labels = [], []
    all_polarity_preds, all_polarity_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            lengths = batch['length'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            # Forward pass
            outputs = model(features, lengths)
            
            # Calculate losses
            loss_intensity = loss_fns['regression'](outputs['intensity'], labels['intensity'])
            loss_empathy = loss_fns['regression'](outputs['empathy'], labels['empathy'])
            loss_polarity = loss_fns['classification'](outputs['polarity'], labels['polarity'])
            combined_loss = (loss_weights['intensity'] * loss_intensity +
                             loss_weights['empathy'] * loss_empathy +
                             loss_weights['polarity'] * loss_polarity)
            total_loss += combined_loss.item()
            
            # Store predictions
            all_intensity_preds.append(outputs['intensity'].cpu())
            all_intensity_labels.append(labels['intensity'].cpu())
            all_empathy_preds.append(outputs['empathy'].cpu())
            all_empathy_labels.append(labels['empathy'].cpu())
            
            polarity_preds = torch.argmax(outputs['polarity'], dim=1)
            all_polarity_preds.append(polarity_preds.cpu())
            all_polarity_labels.append(labels['polarity'].cpu())
    
    # Concatenate results
    all_intensity_preds = torch.cat(all_intensity_preds)
    all_intensity_labels = torch.cat(all_intensity_labels)
    all_empathy_preds = torch.cat(all_empathy_preds)
    all_empathy_labels = torch.cat(all_empathy_labels)
    all_polarity_preds = torch.cat(all_polarity_preds)
    all_polarity_labels = torch.cat(all_polarity_labels)
    
    # Calculate metrics
    mae_intensity = nn.functional.l1_loss(all_intensity_preds, all_intensity_labels).item()
    mae_empathy = nn.functional.l1_loss(all_empathy_preds, all_empathy_labels).item()
    
    accuracy_polarity = accuracy_score(all_polarity_labels, all_polarity_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_polarity_labels, all_polarity_preds, average='weighted', zero_division=0
    )
    
    metrics = {
        "val_loss": total_loss / len(dataloader),
        "intensity_mae": mae_intensity,
        "empathy_mae": mae_empathy,
        "polarity_accuracy": accuracy_polarity,
        "polarity_precision": precision,
        "polarity_recall": recall,
        "polarity_f1": f1
    }
    
    return metrics
```

### Step 7: Train the Model

```python
# Training setup
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_functions = {
    'regression': nn.MSELoss(),
    'classification': nn.CrossEntropyLoss()
}

NUM_EPOCHS = 10
best_val_loss = float('inf')
best_model_path = f"{MODELS_SAVE_PATH}/lstm_bidirectional_best.pth"

print("\n" + "="*60)
print("Starting Training")
print("="*60)

for epoch in range(NUM_EPOCHS):
    # Train
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_functions, device)
    
    # Evaluate
    val_metrics = evaluate_performance(model, eval_loader, loss_functions, device)
    
    # Print progress
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
    print(f"  Polarity F1: {val_metrics['polarity_f1']:.4f}")
    print(f"  Polarity Acc: {val_metrics['polarity_accuracy']:.4f}")
    print(f"  Intensity MAE: {val_metrics['intensity_mae']:.4f}")
    print(f"  Empathy MAE: {val_metrics['empathy_mae']:.4f}")
    
    # Save best model
    if val_metrics['val_loss'] < best_val_loss:
        best_val_loss = val_metrics['val_loss']
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✅ New best model saved!")
    print()

print("="*60)
print("Training Complete!")
print(f"Best model saved at: {best_model_path}")
print(f"Best validation loss: {best_val_loss:.4f}")
print("="*60)
```

### Step 8: Test Set Inference

```python
# Load best model
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Create test dataset
test_dataset = RNNInferenceDataset(
    csv_path=test_csv_path,
    sequence_embedder=sequence_embedder,
    id_column='id',
    text_column='text'
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Make predictions
all_ids = []
all_emotion_preds = []
all_empathy_preds = []
all_polarity_preds = []

print("\n--- Running Inference on Test Set ---")
with torch.no_grad():
    for batch in test_loader:
        ids = batch['id']
        features = batch['features'].to(device)
        lengths = batch['length'].to(device)
        
        outputs = model(features, lengths)
        
        emotion_preds = outputs['intensity'].squeeze().cpu().numpy()
        empathy_preds = outputs['empathy'].squeeze().cpu().numpy()
        polarity_preds = torch.argmax(outputs['polarity'], dim=1).cpu().numpy()
        
        all_ids.extend(ids.numpy())
        all_emotion_preds.extend(emotion_preds)
        all_empathy_preds.extend(empathy_preds)
        all_polarity_preds.extend(polarity_preds)

# Create submission file
import pandas as pd
submission_df = pd.DataFrame({
    'id': all_ids,
    'Emotion': all_emotion_preds,
    'EmotionalPolarity': all_polarity_preds,
    'Empathy': all_empathy_preds
})
submission_df['EmotionalPolarity'] = submission_df['EmotionalPolarity'].astype(int)

submission_path = "Report/lstm_submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"\n✅ Predictions saved to: {submission_path}")
print(f"   Total predictions: {len(submission_df)}")
```

---

## 📊 Configuration Options Explained

### `input_dim` (100)
- Dimension of GloVe embeddings
- Fixed by the embedder you choose
- Don't change unless using different embeddings

### `hidden_dim` (128)
- Size of LSTM hidden state
- **64-128**: Small, fast, may underfit
- **128-256**: Good balance (recommended)
- **256-512**: Large, more capacity, slower

### `num_layers` (2)
- Number of stacked LSTM layers
- **1**: Simple tasks, faster
- **2**: Most common (recommended)
- **3+**: Complex tasks, may overfit

### `dropout` (0.3)
- Regularization to prevent overfitting
- **0.2**: Light regularization
- **0.3**: Standard (recommended)
- **0.4-0.5**: Heavy regularization

### `bidirectional` (True/False)
- **False**: Processes sequence left-to-right only
- **True**: Processes both directions (recommended for better performance)
- Note: 2x parameters when bidirectional

### `num_classes` (4)
- Number of polarity classes in your data
- Fixed by your dataset (0, 1, 2, 3)

---

## 🎯 Key Points

1. **Always pass `lengths` to the model** - enables efficient sequence packing
2. **Use `RNNEmpathyDataset`** - not `EmpathyDataset` (which is for ANNs)
3. **Use `GloveSequenceEmbedder`** - not `GloveEmbedder` (which averages)
4. **Sequences are padded** to `max_length` (default: 50)
5. **Three output heads** for multi-task learning

---

## ✅ Summary

Your LSTM implementation is **complete** and includes:

- ✅ **Embeddings**: Pre-computed GloVe (100-dim)
- ✅ **Recurrent Layers**: LSTM with bidirectional option
- ✅ **Task-Specific Heads**: 3 separate outputs
- ✅ **Preprocessing**: Tokenization, padding, batching
- ✅ **Efficient Processing**: Sequence packing/unpacking
- ✅ **Ready to Train**: Complete training pipeline

Just copy the code above and run it! 🚀
