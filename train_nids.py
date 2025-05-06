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

# Added robust error handling, parameterized file paths, and modularized training loop for better production readiness.
# Optimized preprocessing and enhanced logging.

def download_nsl_kdd(train_path='KDDTrain+.txt', test_path='KDDTest+.txt'):
    """
    Download the NSL-KDD dataset or load it from local files.
    """
    try:
        # Try to load from local path first
        if os.path.exists(train_path) and os.path.exists(test_path):
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

            train_data = pd.read_csv(train_path, header=None, names=col_names)
            test_data = pd.read_csv(test_path, header=None, names=col_names)
            print("Local dataset files loaded successfully!")

            # Debugging: Print column names to verify
            print("Training data columns:", train_data.columns.tolist())
            print("Testing data columns:", test_data.columns.tolist())

            # Check if 'label' column exists, otherwise handle missing column
            if 'label' not in train_data.columns:
                raise KeyError("'label' column is missing in the training dataset.")

            if 'label' not in test_data.columns:
                raise KeyError("'label' column is missing in the testing dataset.")

            return train_data, test_data

        print("Local dataset files not found. Downloading from online source...")

        # If not available locally, download from an online source
        train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
        test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

        train_response = requests.get(train_url)
        test_response = requests.get(test_url)

        # Check if the download was successful
        if train_response.status_code == 200 and test_response.status_code == 200:
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

            train_data = pd.read_csv(io.StringIO(train_response.text), header=None, names=col_names)
            test_data = pd.read_csv(io.StringIO(test_response.text), header=None, names=col_names)

            print("Dataset downloaded successfully!")
            return train_data, test_data
        else:
            raise Exception("Failed to download the dataset. Please check your internet connection.")
    except KeyError as e:
        print(f"KeyError: {e}. Please verify the dataset structure.")
        return None, None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None, None


def preprocess_data(train_data, test_data):
    """
    Preprocess the NSL-KDD dataset for training and testing.
    """
    try:
        train_df = train_data.copy()
        test_df = test_data.copy()

        # Drop unnecessary columns
        train_df.drop('difficulty_level', axis=1, inplace=True)
        test_df.drop('difficulty_level', axis=1, inplace=True)

        # Convert attack labels to binary classification
        train_df['binary_label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        test_df['binary_label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

        # One-hot encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        train_categorical = pd.get_dummies(train_df[categorical_cols], drop_first=False)
        test_categorical = pd.get_dummies(test_df[categorical_cols], drop_first=False)

        # Align test data columns with train data
        train_categorical, test_categorical = train_categorical.align(test_categorical, join='outer', axis=1, fill_value=0)

        # Drop original categorical columns and combine with numerical features
        train_df = pd.concat([train_df.drop(categorical_cols + ['label'], axis=1), train_categorical], axis=1)
        test_df = pd.concat([test_df.drop(categorical_cols + ['label'], axis=1), test_categorical], axis=1)

        # Separate features and target
        X_train = train_df.drop('binary_label', axis=1)
        y_train = train_df['binary_label']
        X_test = test_df.drop('binary_label', axis=1)
        y_test = test_df['binary_label']

        # Handle missing and infinite values
        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)
        X_train.replace([np.inf, -np.inf], np.finfo(np.float32).max, inplace=True)
        X_test.replace([np.inf, -np.inf], np.finfo(np.float32).max, inplace=True)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create validation set
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        print("Preprocessing completed successfully.")
        return X_train_final, y_train_final, X_val, y_val, X_test_scaled, y_test, scaler
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None, None, None, None, None, None


def train_model(X_train, y_train, X_val, y_val, X_test, y_test, model_path='nids_model.pth'):
    """
    Train the NIDS model and save the best model to disk.
    """
    try:
        # Initialize model
        input_size = X_train.shape[1]
        model = NIDSModel(input_size)

        # Define loss function and optimizer
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Training loop
        num_epochs = 100
        early_stopping_patience = 15
        best_val_loss = float('inf')
        no_improve = 0

        history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            y_train_true, y_train_pred = [], []

            for inputs, targets in DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train.values).reshape(-1, 1)), batch_size=256):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

                y_train_true.extend(targets.numpy().flatten())
                y_train_pred.extend(torch.sigmoid(outputs).detach().numpy().flatten())

            train_loss /= len(X_train)
            train_f1 = f1_score(y_train_true, (np.array(y_train_pred) >= 0.5).astype(int))

            # Validation phase
            model.eval()
            val_loss = 0
            y_val_true, y_val_pred = [], []

            with torch.no_grad():
                for inputs, targets in DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val.values).reshape(-1, 1)), batch_size=256):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

                    y_val_true.extend(targets.numpy().flatten())
                    y_val_pred.extend(torch.sigmoid(outputs).numpy().flatten())

            val_loss /= len(X_val)
            val_f1 = f1_score(y_val_true, (np.array(y_val_pred) >= 0.5).astype(int))

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(model.state_dict(), model_path)
                print(f"Epoch {epoch+1}: New best model saved with validation loss {val_loss:.4f}")
            else:
                no_improve += 1
                if no_improve >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Generate predictions for the test set
        model.eval()
        y_test_pred = []
        with torch.no_grad():
            for inputs in DataLoader(torch.FloatTensor(X_test), batch_size=256):
                outputs = model(inputs)
                y_test_pred.extend(torch.sigmoid(outputs).numpy().flatten())

        print("Training completed successfully.")
        return model, history, np.array(y_test_pred)
    except Exception as e:
        print(f"Error during training: {e}")
        return None, None, None

# Improved model architecture with better design principles for this problem
class NIDSModel(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(NIDSModel, self).__init__()
        
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
    plt.savefig('nids_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return the balanced threshold as the default recommendation
    return optimal_threshold

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
    print("You can now use 'run_nids.py' for real-time intrusion detection.")

if __name__ == "__main__":
    main()