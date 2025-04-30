# pip install scapy pandas numpy scikit-learn matplotlib seaborn joblib netifaces requests
# pip install torch torchvision torchaudio



# step 01

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
        precision_recall_curve, average_precision_score, f1_score
    )
    from scapy.all import *
    import time
    import warnings
    import requests
    import io
    import joblib
    import os
    warnings.filterwarnings('ignore')

    print("Libraries imported successfully!")
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please run the pip install command above to install all required packages.")



# step 02

# Download the NSL-KDD dataset
# Uncomment the lines below if you need to download the dataset
# !wget https://www.unb.ca/cic/datasets/nsl.html -O nsl-kdd.zip
# !unzip nsl-kdd.zip

# For this implementation, we'll use a direct download link to the CSV file
# You can replace this with the actual file path if you have the dataset locally
import requests
import io

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

# Download the NSL-KDD dataset
train_data, test_data = download_nsl_kdd()

# Display the first few rows of the training dataset
if train_data is not None:
    print("\nTraining dataset shape:", train_data.shape)
    print("\nSample of training data:")
    print(train_data.head())

    # Count of attack types in the training dataset
    print("\nDistribution of attack types in training dataset:")
    print(train_data['label'].value_counts())




# step 03

def preprocess_data(train_data, test_data):
    # Making a copy of the original data
    train_df = train_data.copy()
    test_df = test_data.copy()

    # Drop the difficulty_level column as it's not needed for classification
    train_df.drop('difficulty_level', axis=1, inplace=True)
    test_df.drop('difficulty_level', axis=1, inplace=True)

    # Convert attack labels to binary classification (normal vs attack)
    train_df['binary_label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['binary_label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Handle categorical features using one-hot encoding
    categorical_cols = ['protocol_type', 'service', 'flag']
    train_categorical = pd.get_dummies(train_df[categorical_cols], drop_first=True)
    test_categorical = pd.get_dummies(test_df[categorical_cols], drop_first=True)

    # Ensure test data has the same columns as train data
    missing_cols = set(train_categorical.columns) - set(test_categorical.columns)
    for col in missing_cols:
        test_categorical[col] = 0
    test_categorical = test_categorical[train_categorical.columns]

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

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler

# Preprocess the data
if train_data is not None and test_data is not None:
    X_train, y_train, X_test, y_test, scaler = preprocess_data(train_data, test_data)
    print("Data preprocessing completed!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Check class distribution
    print("\nClass distribution in training set:")
    print(pd.Series(y_train).value_counts(normalize=True) * 100)

    # Check class distribution in test set
    print("\nClass distribution in test set:")
    print(pd.Series(y_test).value_counts(normalize=True) * 100)




# step 04

import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

# PyTorch model
class NIDSModel(nn.Module):
    def __init__(self, input_size):
        super(NIDSModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def build_model(input_size):
    model = NIDSModel(input_size)
    return model

# PyTorch training function
def train_model(X_train, y_train, X_test, y_test):
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    model = build_model(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    best_loss = float('inf')
    patience = 10
    no_improve = 0
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor).squeeze()
            val_loss = criterion(val_outputs, y_test_tensor)
            val_acc = accuracy_score(y_test, (val_outputs > 0.5).float().numpy())
            
        # Store history
        history['loss'].append(epoch_loss/len(train_loader))
        history['val_loss'].append(val_loss.item())
        history['accuracy'].append(accuracy_score(y_train, (model(X_train_tensor).squeeze() > 0.5).float().numpy()))
        history['val_accuracy'].append(val_acc)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Evaluation
    with torch.no_grad():
        test_outputs = model(X_test_tensor).squeeze()
        test_loss = criterion(test_outputs, y_test_tensor)
        test_acc = accuracy_score(y_test, (test_outputs > 0.5).float().numpy())
        
    print(f"\nTest Loss: {test_loss.item():.4f} — Test Accuracy: {test_acc:.4f}")
    
    # Return model and history
    return model, history

# Train the model
if X_train is not None and y_train is not None and X_test is not None and y_test is not None:
    print("Training the NIDS model...")
    nids_model, history = train_model(X_train, y_train, X_test, y_test)
    
    # Save the model
    torch.save(nids_model.state_dict(), 'nids_model.pth')
    print("Model saved to 'nids_model.pth'")




# step 05

class NIDSPacketCapture:
    def __init__(self, model_path='nids_model.pth', scaler=None, interface='eth0'):
        """
        Initialize the NIDS packet capture system

        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        scaler : StandardScaler
            Scaler used to normalize features
        interface : str
            Network interface to capture packets from
        """
        # Load PyTorch model
        input_size = 122  # Default size - should match your feature size
        if scaler:
            input_size = len(scaler.feature_names_in_)
        self.model = build_model(input_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set model to evaluation mode
        
        self.scaler = scaler
        self.interface = interface
        self.packet_buffer = []
        self.buffer_size = 100  # Number of packets to buffer before analysis
        self.alert_threshold = 0.7  # Probability threshold for alerts

    def extract_features_from_packet(self, packet):
        """
        Extract relevant features from a captured packet

        Parameters:
        -----------
        packet : Scapy packet
            Captured network packet

        Returns:
        --------
        dict : Dictionary containing extracted features
        """
        features = {
            # Initialize with default values
            'duration': 0,
            'protocol_type': 'tcp',  # Default protocol
            'service': 'http',       # Default service
            'flag': 'SF',            # Default flag
            'src_bytes': 0,
            'dst_bytes': 0,
            'land': 0,
            'wrong_fragment': 0,
            'urgent': 0,
            # Additional features with default values
            'count': 1,
            'srv_count': 1,
            'serror_rate': 0,
            'srv_serror_rate': 0,
            'rerror_rate': 0,
            'srv_rerror_rate': 0,
            'same_srv_rate': 1,
            'diff_srv_rate': 0,
            'srv_diff_host_rate': 0
        }

        # Extract IP layer features if present
        if IP in packet:
            features['src_bytes'] = len(packet[IP])

            # Check for TCP protocol
            if TCP in packet:
                features['protocol_type'] = 'tcp'
                # TCP flags can be used to identify connection state
                if packet[TCP].flags & 0x02:  # SYN flag
                    features['flag'] = 'S'
                elif packet[TCP].flags & 0x10:  # ACK flag
                    features['flag'] = 'A'

                # Identify service based on port
                dst_port = packet[TCP].dport
                if dst_port == 80:
                    features['service'] = 'http'
                elif dst_port == 443:
                    features['service'] = 'https'
                elif dst_port == 22:
                    features['service'] = 'ssh'
                elif dst_port == 21:
                    features['service'] = 'ftp'
                else:
                    features['service'] = 'other'

                # Check for urgent pointer
                features['urgent'] = 1 if packet[TCP].urgptr > 0 else 0

            # Check for UDP protocol
            elif UDP in packet:
                features['protocol_type'] = 'udp'
                features['flag'] = 'SF'  # No flags in UDP

                # Identify service based on port
                dst_port = packet[UDP].dport
                if dst_port == 53:
                    features['service'] = 'domain'
                else:
                    features['service'] = 'other'

            # Check for ICMP protocol
            elif ICMP in packet:
                features['protocol_type'] = 'icmp'
                features['flag'] = 'SF'  # No flags in ICMP
                features['service'] = 'ecr_i'  # echo reply

            # Check for fragment
            if packet[IP].flags & 0x01 or packet[IP].frag > 0:  # MF flag or fragment offset
                features['wrong_fragment'] = 1

            # Check for land attack (same source and destination)
            if packet[IP].src == packet[IP].dst:
                features['land'] = 1

            # Set destination bytes
            features['dst_bytes'] = features['src_bytes']  # Simplified assumption

        return features

    def preprocess_packet_features(self, packet_features):
        """
        Preprocess packet features to match the format expected by the model

        Parameters:
        -----------
        packet_features : dict
            Dictionary containing packet features

        Returns:
        --------
        np.array : Preprocessed features ready for model input
        """
        # Convert categorical features to one-hot encoding
        # Create a DataFrame with the features
        feature_df = pd.DataFrame([packet_features])

        # One-hot encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        categorical_data = pd.get_dummies(feature_df[categorical_cols], drop_first=True)

        # Combine with numerical features
        numerical_cols = [col for col in feature_df.columns if col not in categorical_cols]
        processed_df = pd.concat([feature_df[numerical_cols], categorical_data], axis=1)

        # Make sure we have all required columns (match training data)
        # If missing columns, add them with zeros
        # This is a simplified approach - in a production system, you'd ensure all columns match exactly

        # Apply scaling if scaler is available
        if self.scaler:
            # Ensure all columns in the scaler are present in the DataFrame
            missing_cols = set(self.scaler.feature_names_in_) - set(processed_df.columns)
            for col in missing_cols:
                processed_df[col] = 0

            # Reorder columns to match the scaler's expected order
            processed_df = processed_df[self.scaler.feature_names_in_]

            # Scale the features
            processed_features = self.scaler.transform(processed_df)
        else:
            # If no scaler provided, return as is (not recommended for production)
            processed_features = processed_df.values

        return processed_features

    def packet_callback(self, packet):
        """
        Callback function for packet processing

        Parameters:
        -----------
        packet : Scapy packet
            Captured network packet
        """
        # Check if it's an IP packet
        if IP in packet:
            # Extract features from the packet
            features = self.extract_features_from_packet(packet)

            # Add to buffer
            self.packet_buffer.append(features)

            # Process buffer if it reaches the threshold
            if len(self.packet_buffer) >= self.buffer_size:
                self.process_buffer()

    def process_buffer(self):
        """
        Process the buffered packets and detect anomalies
        """
        print(f"Processing {len(self.packet_buffer)} packets...")

        # Preprocess each packet in the buffer
        processed_packets = []
        for packet_features in self.packet_buffer:
            processed_features = self.preprocess_packet_features(packet_features)
            processed_packets.append(processed_features)

        if not processed_packets:
            print("No valid packets to process.")
            self.packet_buffer = []
            return

        # Stack the processed features
        X = np.vstack(processed_packets)

        # Make predictions with PyTorch
        with torch.no_grad():
            inputs = torch.FloatTensor(X)
            outputs = self.model(inputs).squeeze()
            y_pred_prob = outputs.cpu().numpy()

        # Count anomalous packets
        anomalous_packets = np.sum(y_pred_prob > self.alert_threshold)

        # Generate alert if needed
        if anomalous_packets > 0:
            alert_percentage = (anomalous_packets / len(self.packet_buffer)) * 100
            print(f"\n🚨 ALERT: Detected {anomalous_packets} potential intrusions "
                  f"({alert_percentage:.2f}% of traffic)!")

            # In a real system, you might want to log these alerts or trigger notifications

        # Clear the buffer
        self.packet_buffer = []

    def start_capture(self, count=None, timeout=None):
        """
        Start capturing packets

        Parameters:
        -----------
        count : int, optional
            Number of packets to capture (None for indefinite)
        timeout : int, optional
            Timeout in seconds (None for indefinite)
        """
        print(f"Starting packet capture on interface '{self.interface}'...")
        print("Press Ctrl+C to stop the capture.")

        try:
            # Start sniffing packets
            sniff(iface=self.interface, prn=self.packet_callback, count=count, timeout=timeout)
        except KeyboardInterrupt:
            print("\nPacket capture stopped by user.")
        finally:
            # Process any remaining packets in the buffer
            if self.packet_buffer:
                self.process_buffer()
            print("Packet capture complete.")

# Example usage of the NIDS packet capture system
def run_nids():
    try:
        # Load the saved model
        model_path = 'nids_model.pth'  # Changed from .h5 to .pth
        print(f"Loading model from {model_path}...")

        # Determine the network interface to use
        # In a real implementation, this would be configurable
        import netifaces
        interfaces = netifaces.interfaces()
        # Remove loopback interfaces
        interfaces = [i for i in interfaces if not i.startswith('lo')]

        if not interfaces:
            print("No network interfaces found!")
            return

        # Use the first available interface
        interface = interfaces[0]
        print(f"Available interfaces: {interfaces}")
        print(f"Using interface: {interface}")

        # Create and start the NIDS system
        # In a real implementation, you should also pass the trained scaler
        nids = NIDSPacketCapture(model_path=model_path, interface=interface)

        # Start the capture
        nids.start_capture(timeout=60)  # Capture for 60 seconds

    except ImportError:
        print("Could not import netifaces. Please install it with: pip install netifaces")
    except Exception as e:
        print(f"Error running NIDS: {e}")

# Run the NIDS if the model has been trained
if 'nids_model' in locals():
    print("\nReady to run real-time NIDS packet capture.")
    print("NOTE: Running this in a Jupyter notebook might cause issues with packet capture.")
    print("For best results, export this code to a Python script and run it with administrator/root privileges.")
    print("\nTo run the NIDS, execute the following command:")
    print("run_nids()")





# step 06

def analyze_results(y_true, y_pred, y_pred_prob):
    """
    Analyze model results and create visualizations

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_prob : array-like
        Predicted probabilities
    """
    # Create a figure with multiple subplots
    plt.figure(figsize=(18, 12))

    # 1. Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 2. ROC Curve
    plt.subplot(2, 3, 2)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # 3. Precision-Recall Curve
    plt.subplot(2, 3, 3)
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)
    plt.step(recall, precision, where='post', label=f'AP = {ap:.4f}')
    plt.fill_between(recall, precision, step='post', alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    # 4. Decision Threshold Analysis
    plt.subplot(2, 3, 4)
    thresholds = np.arange(0, 1.01, 0.05)
    f1_scores = []
    for threshold in thresholds:
        y_pred_t = (y_pred_prob > threshold).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred_t)
        f1_scores.append(f1)

    plt.plot(thresholds, f1_scores, '-o')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Decision Threshold')
    plt.grid(True)

    # 5. Probability Distribution
    plt.subplot(2, 3, 5)
    normal_probs = y_pred_prob[y_true == 0]
    attack_probs = y_pred_prob[y_true == 1]

    plt.hist(normal_probs, bins=20, alpha=0.5, label='Normal Traffic', color='green')
    plt.hist(attack_probs, bins=20, alpha=0.5, label='Attack Traffic', color='red')
    plt.xlabel('Predicted Probability of Attack')
    plt.ylabel('Count')
    plt.title('Probability Distribution')
    plt.legend(loc='best')  # Automatically position the legend optimally

    # 6. Feature Importance (if available)
    plt.subplot(2, 3, 6)
    # For neural network models, we could consider adding permutation importance
    # or other model-agnostic feature importance methods in the future
    plt.text(0.5, 0.5, "Feature Importance\n(Not available for neural networks)\n\nConsider implementing permutation importance\nor integrated gradients for future versions",
             ha='center', va='center', fontsize=10, multialignment='center')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('nids_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create a summary report
    print("\n===== NIDS Model Performance Report =====")

    # Basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    # Confusion matrix breakdown
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Positives: {tp} (Attacks correctly identified)")
    print(f"False Positives: {fp} (Normal traffic incorrectly flagged as attacks)")
    print(f"True Negatives: {tn} (Normal traffic correctly identified)")
    print(f"False Negatives: {fn} (Attacks missed)")

    # Additional metrics
    fpr = fp / (fp + tn)  # False positive rate
    fnr = fn / (tp + fn)  # False negative rate

    print(f"\nFalse Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")

    # Optimal threshold
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    print(f"\nOptimal decision threshold: {best_threshold:.2f}")
    print(f"F1 Score at optimal threshold: {f1_scores[best_threshold_idx]:.4f}")

    print("\n============= End of Report =============")

    # Save the report to a file
    with open('nids_model_report.txt', 'w') as f:
        f.write("===== NIDS Model Performance Report =====\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC-ROC: {roc_auc:.4f}\n")
        f.write(f"\nTrue Positives: {tp} (Attacks correctly identified)\n")
        f.write(f"False Positives: {fp} (Normal traffic incorrectly flagged as attacks)\n")
        f.write(f"True Negatives: {tn} (Normal traffic correctly identified)\n")
        f.write(f"False Negatives: {fn} (Attacks missed)\n")
        f.write(f"\nFalse Positive Rate: {fpr:.4f}\n")
        f.write(f"False Negative Rate: {fnr:.4f}\n")
        f.write(f"\nOptimal decision threshold: {best_threshold:.2f}\n")
        f.write(f"F1 Score at optimal threshold: {f1_scores[best_threshold_idx]:.4f}\n")
        f.write("\n============= End of Report =============\n")

    print("Report saved to 'nids_model_report.txt'")

# Generate analysis and report if model has been trained
if 'X_test' in locals() and 'y_test' in locals() and 'nids_model' in locals():
    # Get predictions on test data
    # Function to generate an analytical report after training the model
    def generate_analytical_report(model, X_test, y_test):
        print("Generating analytical report...")

        # Make predictions with PyTorch
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            y_pred_prob = model(X_test_tensor).squeeze().cpu().numpy()
            y_pred = (y_pred_prob > 0.5).astype(int)

        # Analyze results
        analyze_results(y_test, y_pred, y_pred_prob)

        return y_pred, y_pred_prob

# Run the analysis if model has been trained
if 'X_test' in locals() and 'y_test' in locals() and 'nids_model' in locals():
    print("\nGenerating comprehensive analysis and visualization report...")
    y_pred, y_pred_prob = generate_analytical_report(nids_model, X_test, y_test)
    print("Analysis and visualization completed!")



# step 07

class NIDSSystem:
    """
    Network Intrusion Detection System - Complete Integration

    This class integrates all components of the NIDS system:
    - Data preprocessing
    - Model training
    - Real-time detection
    - Visualization and reporting
    """

    def __init__(self):
        """Initialize the NIDS system"""
        self.scaler = None
        self.model = None
        self.training_history = None
        self.dataset_loaded = False
        self.model_trained = False

    def load_dataset(self, train_path=None, test_path=None):
        """
        Load and preprocess the dataset

        Parameters:
        -----------
        train_path : str, optional
            Path to the training dataset
        test_path : str, optional
            Path to the testing dataset

        Returns:
        --------
        bool : Whether dataset loading was successful
        """
        try:
            # Download or load the dataset
            self.train_data, self.test_data = download_nsl_kdd()

            if self.train_data is None or self.test_data is None:
                print("Failed to load dataset.")
                return False

            # Preprocess the data
            self.X_train, self.y_train, self.X_test, self.y_test, self.scaler = preprocess_data(
                self.train_data, self.test_data
            )

            self.dataset_loaded = True
            print("Dataset loaded and preprocessed successfully!")
            return True

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def train_model(self, save_path='nids_model.pth'):
        """
        Train the NIDS machine learning model

        Parameters:
        -----------
        save_path : str, optional
            Path to save the trained model

        Returns:
        --------
        bool : Whether model training was successful
        """
        try:
            if not self.dataset_loaded:
                print("Dataset not loaded. Please load the dataset first.")
                return False

            print("Training NIDS model...")

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(self.X_train)
            y_train_tensor = torch.FloatTensor(self.y_train.values)
            X_test_tensor = torch.FloatTensor(self.X_test)
            y_test_tensor = torch.FloatTensor(self.y_test.values)
            
            # Create DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            
            # Initialize model and training components
            self.model = build_model(self.X_train.shape[1])
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters())
            
            # Training loop
            best_loss = float('inf')
            patience = 10
            no_improve = 0
            self.training_history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
            
            for epoch in range(20):  # Reduced for faster training
                self.model.train()
                epoch_loss = 0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_test_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_test_tensor)
                    val_acc = accuracy_score(self.y_test, (val_outputs > 0.5).float().numpy())
                    
                # Store history
                self.training_history['loss'].append(epoch_loss/len(train_loader))
                self.training_history['val_loss'].append(val_loss.item())
                self.training_history['accuracy'].append(accuracy_score(self.y_train, (self.model(X_train_tensor).squeeze() > 0.5).float().numpy()))
                self.training_history['val_accuracy'].append(val_acc)
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improve = 0
                    torch.save(self.model.state_dict(), save_path)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                        
            # Load best model
            self.model.load_state_dict(torch.load(save_path))
            self.model.eval()
            
            # Evaluate the model
            with torch.no_grad():
                test_outputs = self.model(X_test_tensor).squeeze()
                test_loss = criterion(test_outputs, y_test_tensor)
                test_acc = accuracy_score(self.y_test, (test_outputs > 0.5).float().numpy())
                
            print(f"Test Accuracy: {test_acc:.4f}")

            self.model_trained = True
            return True

        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def evaluate_model(self):
        """
        Evaluate the trained model and generate reports

        Returns:
        --------
        bool : Whether evaluation was successful
        """
        try:
            if not self.model_trained:
                print("Model not trained. Please train the model first.")
                return False

            # Make predictions with PyTorch
            self.model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(self.X_test)
                y_pred_prob = self.model(X_test_tensor).squeeze().cpu().numpy()
                y_pred = (y_pred_prob > 0.5).astype(int)

            # Display classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))

            # Generate comprehensive analysis
            analyze_results(self.y_test, y_pred, y_pred_prob)

            return True

        except Exception as e:
            print(f"Error evaluating model: {e}")
            return False

    def start_real_time_detection(self, interface=None, timeout=60):
        """
        Start real-time network traffic monitoring and intrusion detection

        Parameters:
        -----------
        interface : str, optional
            Network interface to monitor
        timeout : int, optional
            Timeout in seconds for packet capture

        Returns:
        --------
        bool : Whether real-time detection was started successfully
        """
        try:
            if not self.model_trained:
                print("Model not trained. Please train the model first.")
                return False

            # Determine network interface if not specified
            if interface is None:
                try:
                    import netifaces
                    interfaces = netifaces.interfaces()
                    # Remove loopback interfaces
                    interfaces = [i for i in interfaces if not i.startswith('lo')]

                    if not interfaces:
                        print("No network interfaces found!")
                        return False

                    interface = interfaces[0]
                    print(f"Available interfaces: {interfaces}")
                    print(f"Using interface: {interface}")
                except ImportError:
                    print("Could not import netifaces. Please install it with: pip install netifaces")
                    print("Defaulting to 'eth0' interface.")
                    interface = 'eth0'

            # Create and start the NIDS
            nids = NIDSPacketCapture(
                model_path='nids_model.pth',
                scaler=self.scaler,
                interface=interface
            )

            # Start the capture
            nids.start_capture(timeout=timeout)
            return True

        except Exception as e:
            print(f"Error starting real-time detection: {e}")
            return False

    def run(self):
        """
        Run the complete NIDS pipeline with user interaction
        """
        print("=" * 80)
        print("Network Intrusion Detection System (NIDS) using Machine Learning")
        print("=" * 80)

        # Step 1: Load dataset
        print("\nStep 1: Loading dataset...")
        if not self.load_dataset():
            print("Failed to load dataset. Exiting.")
            return

        # Step 2: Train model
        print("\nStep 2: Training model...")
        if not self.train_model():
            print("Failed to train model. Exiting.")
            return

        # Step 3: Evaluate model
        print("\nStep 3: Evaluating model...")
        if not self.evaluate_model():
            print("Failed to evaluate model. Continuing anyway.")

        # Step 4: Real-time detection
        print("\nStep 4: Starting real-time detection...")
        print("NOTE: This step requires administrator/root privileges in a real environment.")
        print("In a Jupyter notebook, packet capture functionality might be limited.")

        try:
            user_input = input("Do you want to proceed with real-time detection? (y/n): ")
            if user_input.lower() == 'y':
                timeout = 60
                try:
                    timeout_input = input("Enter capture duration in seconds (default: 60): ")
                    if timeout_input.strip():
                        timeout = int(timeout_input)
                except ValueError:
                    print("Invalid input. Using default timeout of 60 seconds.")

                self.start_real_time_detection(timeout=timeout)
            else:
                print("Skipping real-time detection.")
        except Exception as e:
            print(f"Error during user interaction: {e}")

        print("\nNIDS pipeline completed!")

# Function to run the complete NIDS system
def run_complete_nids():
    """
    Run the complete NIDS system pipeline
    """
    nids_system = NIDSSystem()
    nids_system.run()

# Provide guidance on how to run the system
print("\n" + "=" * 80)
print("NIDS Project Implementation Complete!")
print("=" * 80)
print("\nTo run the complete NIDS pipeline, execute the following command:")
print("run_complete_nids()")
print("\nAlternatively, you can run individual components as follows:")
print("1. Load and preprocess data:")
print("   nids = NIDSSystem()")
print("   nids.load_dataset()")
print("\n2. Train the model:")
print("   nids.train_model()")
print("\n3. Evaluate the model:")
print("   nids.evaluate_model()")
print("\n4. Start real-time detection:")
print("   nids.start_real_time_detection()")
print("\nNote: For real-time packet capture, you may need to run the script outside")
print("of Jupyter notebook with administrator/root privileges.")
