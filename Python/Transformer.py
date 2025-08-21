import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerConfigModel(nn.Module):
    def __init__(self, input_dim, num_prune_classes, num_quant_classes, num_domain_classes, 
                 d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, 
                 dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Task-specific heads
        self.prune_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_prune_classes)
        )
        self.quant_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_quant_classes)
        )
        self.domain_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_domain_classes)
        )

    def forward(self, src):
        # src: [batch_size, seq_len, input_dim]
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        # Take the last token output for classification
        last_output = output[:, -1, :]
        
        prune_logits = self.prune_head(last_output)
        quant_logits = self.quant_head(last_output)
        domain_logits = self.domain_head(last_output)
        
        return prune_logits, quant_logits, domain_logits

class LayerConfigDataset(Dataset):
    def __init__(self, features, prune_labels, quant_labels, domain_labels):
        self.features = features
        self.prune_labels = prune_labels
        self.quant_labels = quant_labels
        self.domain_labels = domain_labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.prune_labels[idx],
            self.quant_labels[idx],
            self.domain_labels[idx]
        )

def prepare_data(file_path):
    # Load data
    df = pd.read_csv(file_path, skiprows=[1], skipfooter=1, engine='python')
    
    # Feature selection
    features = [
        'Sparsity (%)', 'Parameters', 'FLOPs', 
        'Weight L1 Norm', 'Weight L2 Norm', 
        'Activation Norm', 'Activation Kurtosis'
    ]
    
    # Extract features
    X = df[features].values
    
    # Encode targets
    le_prune = LabelEncoder()
    le_quant = LabelEncoder()
    le_domain = LabelEncoder()
    
    # Create synthetic labels for demonstration
    # In practice, these would come from your configuration data
    prune_strategies = np.random.choice(['none', 'magnitude', 'adaptive'], size=len(df))
    quant_levels = np.random.choice([2, 4, 8], size=len(df))
    domains = np.random.choice(['analog', 'digital'], size=len(df))
    
    y_prune = le_prune.fit_transform(prune_strategies)
    y_quant = le_quant.fit_transform(quant_levels)
    y_domain = le_domain.fit_transform(domains)
    
    return X, y_prune, y_quant, y_domain, le_prune, le_quant, le_domain

def train_transformer_model(X, y_prune, y_quant, y_domain):
    # Split data
    X_train, X_test, y_prune_train, y_prune_test, y_quant_train, y_quant_test, y_domain_train, y_domain_test = train_test_split(
        X, y_prune, y_quant, y_domain, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_prune_train = torch.tensor(y_prune_train, dtype=torch.long)
    y_prune_test = torch.tensor(y_prune_test, dtype=torch.long)
    y_quant_train = torch.tensor(y_quant_train, dtype=torch.long)
    y_quant_test = torch.tensor(y_quant_test, dtype=torch.long)
    y_domain_train = torch.tensor(y_domain_train, dtype=torch.long)
    y_domain_test = torch.tensor(y_domain_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(
        X_train, y_prune_train, y_quant_train, y_domain_train
    )
    test_dataset = TensorDataset(
        X_test, y_prune_test, y_quant_test, y_domain_test
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    input_dim = X_train.shape[1]
    num_prune_classes = len(np.unique(y_prune))
    num_quant_classes = len(np.unique(y_quant))
    num_domain_classes = len(np.unique(y_domain))
    
    model = TransformerConfigModel(
        input_dim, num_prune_classes, num_quant_classes, num_domain_classes
    ).to(DEVICE)
    
    # Loss functions and optimizer
    criterion_prune = nn.CrossEntropyLoss()
    criterion_quant = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        prune_correct = 0
        quant_correct = 0
        domain_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            features, prune_labels, quant_labels, domain_labels = batch
            features = features.to(DEVICE)
            prune_labels = prune_labels.to(DEVICE)
            quant_labels = quant_labels.to(DEVICE)
            domain_labels = domain_labels.to(DEVICE)
            
            # Add sequence dimension (batch_size, 1, input_dim)
            features = features.unsqueeze(1)
            
            optimizer.zero_grad()
            
            prune_logits, quant_logits, domain_logits = model(features)
            
            loss_prune = criterion_prune(prune_logits, prune_labels)
            loss_quant = criterion_quant(quant_logits, quant_labels)
            loss_domain = criterion_domain(domain_logits, domain_labels)
            
            # Weighted loss
            loss = 0.4 * loss_prune + 0.4 * loss_quant + 0.2 * loss_domain
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, prune_preds = torch.max(prune_logits, 1)
            _, quant_preds = torch.max(quant_logits, 1)
            _, domain_preds = torch.max(domain_logits, 1)
            
            prune_correct += (prune_preds == prune_labels).sum().item()
            quant_correct += (quant_preds == quant_labels).sum().item()
            domain_correct += (domain_preds == domain_labels).sum().item()
            total_samples += prune_labels.size(0)
        
        train_prune_acc = prune_correct / total_samples
        train_quant_acc = quant_correct / total_samples
        train_domain_acc = domain_correct / total_samples
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        val_prune_acc, val_quant_acc, val_domain_acc, val_loss = evaluate_model(model, test_loader, criterion_prune, criterion_quant, criterion_domain)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Prune Acc: Train {train_prune_acc:.4f} | Val {val_prune_acc:.4f}")
        print(f"  Quant Acc: Train {train_quant_acc:.4f} | Val {val_quant_acc:.4f}")
        print(f"  Domain Acc: Train {train_domain_acc:.4f} | Val {val_domain_acc:.4f}")
        
        # Save best model
        avg_val_acc = (val_prune_acc + val_quant_acc + val_domain_acc) / 3
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), "best_transformer_model.pth")
            print("  Saved best model")
    
    return model

def evaluate_model(model, loader, criterion_prune, criterion_quant, criterion_domain):
    model.eval()
    total_loss = 0.0
    prune_correct = 0
    quant_correct = 0
    domain_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            features, prune_labels, quant_labels, domain_labels = batch
            features = features.to(DEVICE)
            prune_labels = prune_labels.to(DEVICE)
            quant_labels = quant_labels.to(DEVICE)
            domain_labels = domain_labels.to(DEVICE)
            
            # Add sequence dimension
            features = features.unsqueeze(1)
            
            prune_logits, quant_logits, domain_logits = model(features)
            
            loss_prune = criterion_prune(prune_logits, prune_labels)
            loss_quant = criterion_quant(quant_logits, quant_labels)
            loss_domain = criterion_domain(domain_logits, domain_labels)
            
            loss = 0.4 * loss_prune + 0.4 * loss_quant + 0.2 * loss_domain
            total_loss += loss.item()
            
            _, prune_preds = torch.max(prune_logits, 1)
            _, quant_preds = torch.max(quant_logits, 1)
            _, domain_preds = torch.max(domain_logits, 1)
            
            prune_correct += (prune_preds == prune_labels).sum().item()
            quant_correct += (quant_preds == quant_labels).sum().item()
            domain_correct += (domain_preds == domain_labels).sum().item()
            total_samples += prune_labels.size(0)
    
    avg_loss = total_loss / len(loader)
    prune_acc = prune_correct / total_samples
    quant_acc = quant_correct / total_samples
    domain_acc = domain_correct / total_samples
    
    return prune_acc, quant_acc, domain_acc, avg_loss

def predict_layer_config(model, layer_features, le_prune, le_quant, le_domain):
    model.eval()
    with torch.no_grad():
        # Convert to tensor and add dimensions
        features = torch.tensor(layer_features, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(DEVICE)
        
        prune_logits, quant_logits, domain_logits = model(features)
        
        _, prune_pred = torch.max(prune_logits, 1)
        _, quant_pred = torch.max(quant_logits, 1)
        _, domain_pred = torch.max(domain_logits, 1)
        
        prune_strategy = le_prune.inverse_transform(prune_pred.cpu().numpy())[0]
        quant_bitwidth = le_quant.inverse_transform(quant_pred.cpu().numpy())[0]
        domain = le_domain.inverse_transform(domain_pred.cpu().numpy())[0]
        
        return prune_strategy, quant_bitwidth, domain

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    file_path = "ResNet50_Layer_Analysis_20250629_224226.csv"
    X, y_prune, y_quant, y_domain, le_prune, le_quant, le_domain = prepare_data(file_path)
    
    # Train transformer model
    model = train_transformer_model(X, y_prune, y_quant, y_domain)
    
    # Save the entire model with label encoders
    torch.save({
        'model_state_dict': model.state_dict(),
        'le_prune': le_prune,
        'le_quant': le_quant,
        'le_domain': le_domain
    }, "transformer_config_model.pth")
    
    print("\nModel training complete!")
    
    # Example prediction
    sample_layer = X[0]  # Features for first layer
    prune, quant, domain = predict_layer_config(model, sample_layer, le_prune, le_quant, le_domain)
    
    print("\nSample Layer Configuration Prediction:")
    print(f"Pruning Strategy: {prune}")
    print(f"Quantization Bitwidth: {quant}-bit")
    print(f"Hardware Domain: {domain}")
