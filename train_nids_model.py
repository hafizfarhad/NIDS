# pip install scapy pandas numpy scikit-learn matplotlib seaborn joblib requests
# pip install torch torchvision torchaudio

# Import required libraries with error handling
import sys
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_curve, auc,
        precision_recall_curve, average_precision_score, f1_score, 
        accuracy_score, recall_score, precision_score
    )
    import time
    import warnings
    import requests
    import io
    import joblib
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    
    warnings.filterwarnings('ignore')
    print("Libraries imported successfully!")
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please run the pip install command above to install all required packages.")

# Download the NSL-KDD dataset
def download_nsl_kdd():
    try:
        # Try to load from local path first
        try:
            train_data = pd.read_csv('KDDTrain+.txt')
            test_data = pd.read_csv('KDDTest+.txt')
            print("Local dataset files loaded successfully!")
            return train_data, test_data
        except:
            print("Local dataset files not found. Downloading from online source...")

        # If not available locally, download from an online source
        train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
        test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

        train_response = requests.get(train_url)
        test_response = requests.get(test_url)

        # Check if the download was successful
        if train_response.status_code == 200 and test_response.status_code == 200:
            # Define column names according to NSL-KDD dataset documentation
            col_names = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
            ]

            # Convert the content to pandas DataFrames
            train_data = pd.read_csv(io.StringIO(train_response.text), header=None, names=col_names)
            test_data = pd.read_csv(io.StringIO(test_response.text), header=None, names=col_names)

            print("Dataset downloaded successfully!")
            return train_data, test_data
        else:
            print("Failed to download the dataset. Please check your internet connection.")
            return None, None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None, None

def preprocess_data(train_data, test_data):
    # Making a copy of the original data
    train_df = train_data.copy()
    test_df = test_data.copy()

    # Drop the difficulty_level column as it's not needed for classification
    train_df.drop('difficulty_level', axis=1, inplace=True)
    test_df.drop('difficulty_level', axis=1, inplace=True)
    
    # Print information about the dataset
    print("\nBefore preprocessing:")
    print(f"Training data - Normal: {len(train_df[train_df['label'] == 'normal'])}, "
          f"Attack: {len(train_df[train_df['label'] != 'normal'])}")
    print(f"Testing data - Normal: {len(test_df[test_df['label'] == 'normal'])}, "
          f"Attack: {len(test_df[test_df['label'] != 'normal'])}")

    # Convert attack labels to binary classification (normal vs attack)
    train_df['binary_label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['binary_label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Handle categorical features using one-hot encoding
    categorical_cols = ['protocol_type', 'service', 'flag']
    train_categorical = pd.get_dummies(train_df[categorical_cols], drop_first=False)  # Changed to not drop first
    test_categorical = pd.get_dummies(test_df[categorical_cols], drop_first=False)    # Changed to not drop first

    # Ensure test data has the same columns as train data
    # Add missing columns to test set
    missing_cols = set(train_categorical.columns) - set(test_categorical.columns)
    for col in missing_cols:
        test_categorical[col] = 0
    
    # Keep only columns that also appear in the training set
    common_cols = set(train_categorical.columns).intersection(set(test_categorical.columns))
    test_categorical = test_categorical[list(common_cols)]
    train_categorical = train_categorical[list(common_cols)]

    # Drop the original categorical columns and the non-binary label
    train_df = train_df.drop(categorical_cols + ['label'], axis=1)
    test_df = test_df.drop(categorical_cols + ['label'], axis=1)

    # Combine the numerical and one-hot encoded categorical features
    train_df = pd.concat([train_df, train_categorical], axis=1)
    test_df = pd.concat([test_df, test_categorical], axis=1)

    # Separate features and target
    X_train = train_df.drop('binary_label', axis=1)
    y_train = train_df['binary_label']
    X_test = test_df.drop('binary_label', axis=1)
    y_test = test_df['binary_label']

    # Check for missing values
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        print("Warning: Missing values detected, filling with zeros")
        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)

    # Check for infinite values - Fixed handling of non-numeric data
    try:
        # First convert to numeric where possible
        X_train = X_train.apply(pd.to_numeric, errors='ignore')
        X_test = X_test.apply(pd.to_numeric, errors='ignore')
        
        # Get only numeric columns
        numeric_cols_train = X_train.select_dtypes(include=[np.number]).columns
        numeric_cols_test = X_test.select_dtypes(include=[np.number]).columns
        
        # Check for infinite values in numeric columns only
        if (X_train[numeric_cols_train].isin([np.inf, -np.inf])).any().any() or \
           (X_test[numeric_cols_test].isin([np.inf, -np.inf])).any().any():
            print("Warning: Infinite values detected, replacing with large values")
            X_train = X_train.replace([np.inf, -np.inf], np.finfo(np.float32).max)
            X_test = X_test.replace([np.inf, -np.inf], np.finfo(np.float32).max)
    except Exception as e:
        print(f"Warning when checking for infinite values: {e}")
        print("Continuing with preprocessing...")

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create a validation set from the training data (10%)
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # Store feature names in the scaler for later use
    scaler.feature_names_in_ = X_train.columns.tolist()
    
    print(f"\nAfter preprocessing:")
    print(f"X_train shape: {X_train_final.shape}, y_train shape: {y_train_final.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test_scaled.shape}, y_test shape: {y_test.shape}")
    print(f"Class balance - Train: Normal={sum(y_train_final==0)/len(y_train_final):.2f}, Attack={sum(y_train_final==1)/len(y_train_final):.2f}")
    print(f"Class balance - Val: Normal={sum(y_val==0)/len(y_val):.2f}, Attack={sum(y_val==1)/len(y_val):.2f}")
    print(f"Class balance - Test: Normal={sum(y_test==0)/len(y_test):.2f}, Attack={sum(y_test==1)/len(y_test):.2f}")

    return X_train_final, y_train_final, X_val, y_val, X_test_scaled, y_test, scaler

# Improved model architecture with better design principles for this problem
class ImprovedNIDSModel(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(ImprovedNIDSModel, self).__init__()
        
        # Using a more gradual reduction in layer sizes
        # Starting with wider layers for better feature extraction
        self.net = nn.Sequential(
            # First layer - extract features
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            # Second layer - reduce dimensions
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            
            # Third layer - further refinement
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate/2),  # Less dropout near the end
            
            # Output layer - single neuron with sigmoid for binary classification
            nn.Linear(32, 1)
            # Sigmoid moved to loss function for numerical stability
        )
        
    def forward(self, x):
        return self.net(x)

# Improved training function with more careful approach to class imbalance
def train_model(X_train, y_train, X_val, y_val, X_test, y_test, class_weights=None):
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
    
    # Calculate class weights for weighted sampling if not provided
    if class_weights is None:
        # Compute class weights inversely proportional to class frequencies
        class_counts = np.bincount(y_train.astype(int))
        total_samples = len(y_train)
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights)
    
    # Create weighted sampler for imbalanced data
    sample_weights = [class_weights[int(t)] for t in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Create DataLoader with the weighted sampler
    batch_size = 256  # Larger batch size for more stable gradients
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = ImprovedNIDSModel(input_size)
    
    # Define loss function with proper weight balancing
    # Calculate pos_weight based on class imbalance
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Improved optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler for adaptive learning rate - remove verbose param
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training parameters
    num_epochs = 100
    early_stopping_patience = 15
    best_val_loss = float('inf')
    best_val_f1 = 0
    no_improve = 0
    
    # History tracking
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [], 
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }
    
    print("\nStarting model training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            train_loss += loss.item() * inputs.size(0)
            
            # Store predictions and targets for metrics calculation
            train_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().detach().numpy())
            train_targets.extend(targets.cpu().detach().numpy())
        
        # Calculate average training loss and metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_preds = np.array(train_preds).reshape(-1)
        train_targets = np.array(train_targets).reshape(-1)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds)
        train_precision = precision_score(train_targets, train_preds)
        train_recall = recall_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                val_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate average validation loss and metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_preds = np.array(val_preds).reshape(-1)
        val_targets = np.array(val_targets).reshape(-1)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds)
        val_recall = recall_score(val_targets, val_preds)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print the current learning rate when it changes
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
                  f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        # Early stopping based on validation F1 (balances precision and recall)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            no_improve = 0
            # Save best model
            torch.save(model.state_dict(), 'best_nids_model.pth')
            print(f"    New best model saved with F1: {val_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_nids_model.pth'))
    model.eval()
    
    # Evaluate on test set
    test_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            inputs = X_test_tensor[i:i+batch_size]
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            test_probs.extend(probs.cpu().numpy())
    
    test_probs = np.array(test_probs).reshape(-1)
    
    # Calculate test metrics with threshold 0.5
    test_preds = (test_probs > 0.5).astype(int)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds)
    
    print(f"\nTest Results with threshold 0.5:")
    print(f"Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    
    return model, history, test_probs

# Improved analysis function with focus on finding balanced thresholds
def analyze_results(y_true, y_pred_prob, history):
    """
    Comprehensive analysis of model results with threshold optimization
    """
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 16))
    
    # 1. ROC Curve with AUC
    plt.subplot(3, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve with Average Precision
    plt.subplot(3, 2, 2)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    average_precision = average_precision_score(y_true, y_pred_prob)
    plt.plot(recall, precision, label=f'AP = {average_precision:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Training History - Loss
    plt.subplot(3, 2, 3)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Training History - F1 Score
    plt.subplot(3, 2, 4)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Threshold Analysis
    plt.subplot(3, 2, 5)
    thresholds_analysis = np.arange(0.1, 0.91, 0.01)
    metrics = {
        'threshold': [], 'accuracy': [], 'precision': [], 
        'recall': [], 'f1': [], 'fnr': [], 'fpr': []
    }
    
    for threshold in thresholds_analysis:
        y_pred = (y_pred_prob >= threshold).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
        
        # Store metrics
        metrics['threshold'].append(threshold)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        metrics['fnr'].append(fnr)
        metrics['fpr'].append(fpr)
    
    # Plot threshold analysis
    plt.plot(metrics['threshold'], metrics['f1'], label='F1 Score')
    plt.plot(metrics['threshold'], metrics['precision'], label='Precision')
    plt.plot(metrics['threshold'], metrics['recall'], label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Decision Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Confusion Matrix for Optimal F1 Threshold
    plt.subplot(3, 2, 6)
    optimal_f1_idx = np.argmax(metrics['f1'])
    optimal_threshold = metrics['threshold'][optimal_f1_idx]
    y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_optimal)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('nids_improved_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find the best threshold for different scenarios
    best_f1_idx = np.argmax(metrics['f1'])
    best_f1_threshold = metrics['threshold'][best_f1_idx]
    
    # Find a high security threshold (minimize false negatives)
    # But not too extreme - we'll require at least 60% precision
    high_security_candidates = [
        (metrics['threshold'][i], metrics['recall'][i], metrics['precision'][i], metrics['f1'][i])
        for i in range(len(metrics['threshold']))
        if metrics['precision'][i] >= 0.6
    ]
    
    if high_security_candidates:
        high_security_candidates.sort(key=lambda x: -x[1])  # Sort by recall (descending)
        high_security_threshold = high_security_candidates[0][0]
    else:
        high_security_threshold = best_f1_threshold
    
    # Find a balanced threshold - high F1 score with decent precision and recall
    balanced_idx = np.argmax([
        2 * (prec * rec) / (prec + rec) if prec > 0.75 and rec > 0.75 else 0
        for prec, rec in zip(metrics['precision'], metrics['recall'])
    ])
    balanced_threshold = metrics['threshold'][balanced_idx] if balanced_idx > 0 else best_f1_threshold
    
    # Find a low false positive threshold (high precision, but still decent recall)
    low_fp_candidates = [
        (metrics['threshold'][i], metrics['precision'][i], metrics['recall'][i])
        for i in range(len(metrics['threshold']))
        if metrics['recall'][i] >= 0.65  # Still want to catch a good number of attacks
    ]
    
    if low_fp_candidates:
        low_fp_candidates.sort(key=lambda x: -x[1])  # Sort by precision (descending)
        low_fp_threshold = low_fp_candidates[0][0]
    else:
        low_fp_threshold = best_f1_threshold
    
    # Generate report
    print("\n===== NIDS Model Performance Report =====")
    
    # Basic metrics with optimal F1 threshold
    y_pred_opt = (y_pred_prob >= best_f1_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_opt).ravel()
    
    print(f"Optimal F1 Threshold: {best_f1_threshold:.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_opt):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred_opt):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_opt):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_opt):.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix (with optimal F1 threshold):")
    print(f"True Positives: {tp} (Attacks correctly identified)")
    print(f"False Positives: {fp} (Normal traffic incorrectly flagged as attacks)")
    print(f"True Negatives: {tn} (Normal traffic correctly identified)")
    print(f"False Negatives: {fn} (Attacks missed)")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}")
    
    # High security threshold analysis
    y_pred_sec = (y_pred_prob >= high_security_threshold).astype(int)
    tn_sec, fp_sec, fn_sec, tp_sec = confusion_matrix(y_true, y_pred_sec).ravel()
    
    print(f"\nHigh Security Threshold: {high_security_threshold:.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_sec):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_sec):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_sec):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred_sec):.4f}")
    print(f"False Positive Rate: {fp_sec/(fp_sec+tn_sec):.4f}")
    print(f"False Negative Rate: {fn_sec/(fn_sec+tp_sec):.4f}")
    
    # Low false positive threshold analysis
    y_pred_lfp = (y_pred_prob >= low_fp_threshold).astype(int)
    tn_lfp, fp_lfp, fn_lfp, tp_lfp = confusion_matrix(y_true, y_pred_lfp).ravel()
    
    print(f"\nLow False Positive Threshold: {low_fp_threshold:.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_lfp):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_lfp):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_lfp):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred_lfp):.4f}")
    print(f"False Positive Rate: {fp_lfp/(fp_lfp+tn_lfp):.4f}")
    print(f"False Negative Rate: {fn_lfp/(fn_lfp+tp_lfp):.4f}")
    
    # Balanced threshold analysis (balance between precision and recall)
    y_pred_bal = (y_pred_prob >= balanced_threshold).astype(int)
    tn_bal, fp_bal, fn_bal, tp_bal = confusion_matrix(y_true, y_pred_bal).ravel()
    
    print(f"\nBalanced Threshold: {balanced_threshold:.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_bal):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_bal):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_bal):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred_bal):.4f}")
    print(f"False Positive Rate: {fp_bal/(fp_bal+tn_bal):.4f}")
    print(f"False Negative Rate: {fn_bal/(fn_bal+tp_bal):.4f}")
    
    # Recommendation based on results
    print("\n----- Threshold Recommendations -----")
    print(f"1. Balanced (recommended for most scenarios): {balanced_threshold:.4f}")
    print(f"   Good balance between false positives and false negatives")
    print(f"   FPR: {fp_bal/(fp_bal+tn_bal):.4f}, FNR: {fn_bal/(fn_bal+tp_bal):.4f}")
    
    print(f"\n2. High Security: {high_security_threshold:.4f}")
    print(f"   Minimizes missed attacks but may generate more false alarms")
    print(f"   FPR: {fp_sec/(fp_sec+tn_sec):.4f}, FNR: {fn_sec/(fn_sec+tp_sec):.4f}")
    
    print(f"\n3. Low False Positive: {low_fp_threshold:.4f}")
    print(f"   Reduces false alarms but may miss more attacks")
    print(f"   FPR: {fp_lfp/(fp_lfp+tn_lfp):.4f}, FNR: {fn_lfp/(fn_lfp+tp_lfp):.4f}")
    
    print("\n============= End of Report =============")
    
    # Save the report to a file
    with open('nids_model_report.txt', 'w') as f:
        f.write("===== NIDS Model Performance Report =====\n")
        f.write(f"Optimal F1 Threshold: {best_f1_threshold:.4f}\n")
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred_opt):.4f}\n")
        f.write(f"F1 Score: {f1_score(y_true, y_pred_opt):.4f}\n")
        f.write(f"Precision: {precision_score(y_true, y_pred_opt):.4f}\n")
        f.write(f"Recall: {recall_score(y_true, y_pred_opt):.4f}\n")
        f.write(f"AUC-ROC: {roc_auc:.4f}\n")
        
        f.write(f"\nConfusion Matrix (with optimal F1 threshold):\n")
        f.write(f"True Positives: {tp} (Attacks correctly identified)\n")
        f.write(f"False Positives: {fp} (Normal traffic incorrectly flagged as attacks)\n")
        f.write(f"True Negatives: {tn} (Normal traffic correctly identified)\n")
        f.write(f"False Negatives: {fn} (Attacks missed)\n")
        f.write(f"False Positive Rate: {fp/(fp+tn):.4f}\n")
        f.write(f"False Negative Rate: {fn/(fn+tp):.4f}\n")
        
        f.write(f"\nBalanced Threshold: {balanced_threshold:.4f}\n")
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred_bal):.4f}\n")
        f.write(f"Precision: {precision_score(y_true, y_pred_bal):.4f}\n")
        f.write(f"Recall: {recall_score(y_true, y_pred_bal):.4f}\n")
        f.write(f"F1 Score: {f1_score(y_true, y_pred_bal):.4f}\n")
        f.write(f"False Positive Rate: {fp_bal/(fp_bal+tn_bal):.4f}\n")
        f.write(f"False Negative Rate: {fn_bal/(fn_bal+tp_bal):.4f}\n")
        
        f.write("\n----- Threshold Recommendations -----\n")
        f.write(f"1. Balanced (recommended for most scenarios): {balanced_threshold:.4f}\n")
        f.write(f"   Good balance between false positives and false negatives\n")
        f.write(f"   FPR: {fp_bal/(fp_bal+tn_bal):.4f}, FNR: {fn_bal/(fn_bal+tp_bal):.4f}\n")
        
        f.write(f"\n2. High Security: {high_security_threshold:.4f}\n")
        f.write(f"   Minimizes missed attacks but may generate more false alarms\n")
        f.write(f"   FPR: {fp_sec/(fp_sec+tn_sec):.4f}, FNR: {fn_sec/(fn_sec+tp_sec):.4f}\n")
        
        f.write(f"\n3. Low False Positive: {low_fp_threshold:.4f}\n")
        f.write(f"   Reduces false alarms but may miss more attacks\n")
        f.write(f"   FPR: {fp_lfp/(fp_lfp+tn_lfp):.4f}, FNR: {fn_lfp/(fn_lfp+tp_lfp):.4f}\n")
        
        f.write("\n============= End of Report =============\n")

    print("\nReport saved to 'nids_model_report.txt'")
    
    # Return the balanced threshold as the default recommendation
    return balanced_threshold

def main():
    print("=" * 80)
    print("NIDS Model Training Pipeline")
    print("=" * 80)
    
    # Step 1: Download the NSL-KDD dataset
    print("\nStep 1: Downloading and loading the dataset...")
    train_data, test_data = download_nsl_kdd()
    
    if train_data is None or test_data is None:
        print("Failed to load the dataset. Exiting.")
        return
        
    print("\nDataset Statistics:")
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    print("\nDistribution of attack types in training dataset:")
    print(train_data['label'].value_counts())
    
    # Step 2: Preprocess the data
    print("\nStep 2: Preprocessing the data...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_data(train_data, test_data)
    
    # Save the scaler for later use
    joblib.dump(scaler, 'nids_scaler.joblib')
    print("Scaler saved to 'nids_scaler.joblib'")
    
    # Step 3: Train the model
    print("\nStep 3: Training the model with improved architecture...")
    model, history, y_pred_prob = train_model(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Save the trained model
    torch.save(model.state_dict(), 'nids_model.pth')
    print("Model saved to 'nids_model.pth'")
    
    # Step 4: Evaluate the model and generate a report
    print("\nStep 4: Evaluating the model and generating comprehensive analysis...")
    
    # Generate comprehensive analysis and get the optimal threshold
    balanced_threshold = analyze_results(y_test, y_pred_prob, history)
    
    # Save the balanced threshold
    with open('nids_threshold.txt', 'w') as f:
        f.write(str(balanced_threshold))
    print(f"Balanced threshold ({balanced_threshold:.4f}) saved to 'nids_threshold.txt'")
    
    # Final evaluation with the balanced threshold
    final_y_pred = (y_pred_prob >= balanced_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, final_y_pred).ravel()
    
    print("\nFinal performance with balanced threshold:")
    print(f"True Positives: {tp} (Attacks correctly identified)")
    print(f"False Positives: {fp} (Normal traffic incorrectly flagged as attacks)")
    print(f"True Negatives: {tn} (Normal traffic correctly identified)")
    print(f"False Negatives: {fn} (Attacks missed)")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}")
    
    print("\nTraining pipeline completed successfully!")
    print("You can now use 'run_nids_detector.py' for real-time intrusion detection.")

if __name__ == "__main__":
    main()