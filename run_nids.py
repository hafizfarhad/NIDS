# pip install scapy pandas numpy scikit-learn torch joblib netifaces

# Import required libraries with error handling
import sys
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from scapy.all import *
    import joblib
    import time
    import warnings
    import os
    import netifaces
    import datetime
    import threading
    warnings.filterwarnings('ignore')
    
    print("Libraries imported successfully!")
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please run the pip install command above to install all required packages.")

# PyTorch model definition (must match the one used in training)
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

class NIDSPacketCapture:
    def __init__(self, model_path='nids_model.pth', scaler_path='nids_scaler.joblib',
                 threshold_path='nids_threshold.txt', interface='eth0', log_file=None):
        """
        Initialize the NIDS packet capture system

        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        scaler_path : str
            Path to the saved scaler file
        threshold_path : str
            Path to the optimal threshold file
        interface : str
            Network interface to capture packets from
        log_file : str
            Path to the log file
        """
        print(f"Initializing NIDS Packet Capture on interface '{interface}'...")
        
        # Initialize log file
        self.log_file = log_file
        if self.log_file:
            # Create log file with header
            with open(self.log_file, 'w') as f:
                f.write("=== NIDS Log File ===\n")
                f.write(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Interface: {interface}\n")
                f.write("=" * 50 + "\n\n")
        
        # Load the scaler
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
            input_size = len(self.scaler.feature_names_in_)
        except Exception as e:
            print(f"Error loading scaler: {e}")
            print("Using default feature size of 122")
            self.scaler = None
            input_size = 122
        
        # Load PyTorch model
        try:
            self.model = build_model(input_size)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set model to evaluation mode
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Load the optimal threshold
        try:
            with open(threshold_path, 'r') as f:
                self.alert_threshold = float(f.read().strip())
            print(f"Using optimal threshold: {self.alert_threshold}")
        except Exception as e:
            print(f"Error loading threshold: {e}. Using default threshold of 0.5.")
            self.alert_threshold = 0.5
        
        self.interface = interface
        self.packet_buffer = []
        self.buffer_size = 100  # Number of packets to buffer before analysis
        self.start_time = time.time()
        self.packet_count = 0
        self.attack_count = 0
        self.total_packets = 0
        self.total_attacks = 0
        self.running = False
        self.end_time = None  # Will be set when start_capture is called with timeout
        
        # Initialize counters for connection tracking
        self.connections = {}  # Dictionary to track connections
        
        print("NIDS initialization complete!")

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
            'protocol_type': 'unknown',  # Default protocol
            'service': 'other',          # Default service
            'flag': 'OTH',               # Default flag
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

        # Check if packet is valid and contains the necessary layers
        try:
            # Extract IP layer features if present
            if IP in packet:
                features['src_bytes'] = len(packet[IP])

                # Check for TCP protocol
                if TCP in packet:
                    features['protocol_type'] = 'tcp'
                    # TCP flags can be used to identify connection state
                    tcp_flags = packet[TCP].flags if hasattr(packet[TCP], 'flags') else 0
                    if tcp_flags & 0x02:  # SYN flag
                        features['flag'] = 'S'
                    elif tcp_flags & 0x10:  # ACK flag
                        features['flag'] = 'A'

                    # Identify service based on port
                    dst_port = packet[TCP].dport if hasattr(packet[TCP], 'dport') else 0
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
                    features['urgent'] = 1 if hasattr(packet[TCP], 'urgptr') and packet[TCP].urgptr > 0 else 0
                    
                    # Track connections for better feature extraction
                    try:
                        conn_key = f"{packet[IP].src}:{packet[TCP].sport}-{packet[IP].dst}:{packet[TCP].dport}"
                        if conn_key in self.connections:
                            # Update existing connection
                            self.connections[conn_key]['count'] += 1
                            self.connections[conn_key]['bytes'] += len(packet)
                            features['count'] = self.connections[conn_key]['count']
                        else:
                            # New connection
                            self.connections[conn_key] = {
                                'start_time': time.time(),
                                'count': 1,
                                'bytes': len(packet),
                                'service': features['service']
                            }
                    except (AttributeError, IndexError) as e:
                        # Handle cases where TCP header might be malformed
                        pass

                # Check for UDP protocol
                elif UDP in packet:
                    features['protocol_type'] = 'udp'
                    features['flag'] = 'SF'  # No flags in UDP

                    # Identify service based on port
                    dst_port = packet[UDP].dport if hasattr(packet[UDP], 'dport') else 0
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
                if hasattr(packet[IP], 'flags') and hasattr(packet[IP], 'frag'):
                    if packet[IP].flags & 0x01 or packet[IP].frag > 0:  # MF flag or fragment offset
                        features['wrong_fragment'] = 1

                # Check for land attack (same source and destination)
                if hasattr(packet[IP], 'src') and hasattr(packet[IP], 'dst'):
                    if packet[IP].src == packet[IP].dst:
                        features['land'] = 1

                # Set destination bytes
                features['dst_bytes'] = features['src_bytes']  # Simplified assumption
        except (AttributeError, IndexError, TypeError) as e:
            # Log error and return default features
            print(f"Error extracting features: {e}")
        
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
        try:
            # Convert categorical features to one-hot encoding
            # Create a DataFrame with the features
            feature_df = pd.DataFrame([packet_features])

            # One-hot encode categorical features
            categorical_cols = ['protocol_type', 'service', 'flag']
            categorical_data = pd.get_dummies(feature_df[categorical_cols], drop_first=True)

            # Combine with numerical features
            numerical_cols = [col for col in feature_df.columns if col not in categorical_cols]
            processed_df = pd.concat([feature_df[numerical_cols], categorical_data], axis=1)

            # Apply scaling if scaler is available
            if self.scaler:
                # Ensure all columns in the scaler are present in the DataFrame
                missing_cols = set(self.scaler.feature_names_in_) - set(processed_df.columns)
                for col in missing_cols:
                    processed_df[col] = 0

                # Add columns that are in processed_df but not in scaler
                extra_cols = set(processed_df.columns) - set(self.scaler.feature_names_in_)
                processed_df = processed_df.drop(columns=extra_cols)

                # Reorder columns to match the scaler's expected order
                processed_df = processed_df.reindex(columns=self.scaler.feature_names_in_, fill_value=0)

                # Scale the features
                processed_features = self.scaler.transform(processed_df)
            else:
                # If no scaler provided, return as is (not recommended for production)
                processed_features = processed_df.values

            return processed_features
        except Exception as e:
            print(f"Error preprocessing features: {e}")
            # Return dummy features if preprocessing fails
            if self.scaler:
                dummy_features = np.zeros((1, len(self.scaler.feature_names_in_)))
            else:
                dummy_features = np.zeros((1, 122))  # Default feature size
            return dummy_features

    def extract_packet_details(self, packet):
        """
        Extract detailed information from a packet for logging purposes
        
        Parameters:
        -----------
        packet : Scapy packet
            The packet to extract details from
            
        Returns:
        --------
        dict : Dictionary containing detailed packet information
        """
        details = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'src_ip': None,
            'dst_ip': None,
            'protocol': 'unknown',
            'src_port': None,
            'dst_port': None,
            'flags': None,
            'size': len(packet) if packet else 0,
            'attack_type': 'unknown'
        }
        
        try:
            if IP in packet:
                details['src_ip'] = packet[IP].src
                details['dst_ip'] = packet[IP].dst
                
                # TCP specific details
                if TCP in packet:
                    details['protocol'] = 'TCP'
                    details['src_port'] = packet[TCP].sport
                    details['dst_port'] = packet[TCP].dport
                    
                    # Get TCP flags
                    tcp_flags = packet[TCP].flags
                    flag_str = []
                    if tcp_flags & 0x01: flag_str.append('FIN')
                    if tcp_flags & 0x02: flag_str.append('SYN')
                    if tcp_flags & 0x04: flag_str.append('RST')
                    if tcp_flags & 0x08: flag_str.append('PSH')
                    if tcp_flags & 0x10: flag_str.append('ACK')
                    if tcp_flags & 0x20: flag_str.append('URG')
                    details['flags'] = ','.join(flag_str) if flag_str else 'None'
                    
                    # Detect attack type based on patterns
                    if tcp_flags & 0x02 and not (tcp_flags & 0x10):  # SYN without ACK
                        # Track this connection for potential port scan detection
                        src_ip = packet[IP].src
                        if not hasattr(self, 'scan_tracking'):
                            self.scan_tracking = {}
                        
                        if src_ip not in self.scan_tracking:
                            self.scan_tracking[src_ip] = {
                                'ports': set(),
                                'first_seen': time.time()
                            }
                        
                        self.scan_tracking[src_ip]['ports'].add(packet[TCP].dport)
                        
                        # Check if this looks like a port scan (many ports in short time)
                        if (len(self.scan_tracking[src_ip]['ports']) > 5 and 
                            time.time() - self.scan_tracking[src_ip]['first_seen'] < 30):
                            details['attack_type'] = 'PORT_SCAN'
                    
                    # Check for NULL, FIN, XMAS scans
                    if (tcp_flags == 0 or 
                        tcp_flags == 0x01 or  # FIN
                        (tcp_flags & 0x29) == 0x29):  # FIN,PSH,URG
                        details['attack_type'] = 'STEALTH_SCAN'
                
                # UDP specific details
                elif UDP in packet:
                    details['protocol'] = 'UDP'
                    details['src_port'] = packet[UDP].sport
                    details['dst_port'] = packet[UDP].dport
                    details['flags'] = 'N/A'
                
                # ICMP specific details
                elif ICMP in packet:
                    details['protocol'] = 'ICMP'
                    details['flags'] = 'N/A'
                    if hasattr(packet[ICMP], 'type'):
                        icmp_type = packet[ICMP].type
                        if icmp_type == 8:
                            details['attack_type'] = 'PING_SCAN' if packet[IP].src not in self.trusted_hosts else 'NORMAL'
            
            # Check for large packets which might indicate DoS
            if len(packet) > 1500:
                details['attack_type'] = 'POSSIBLE_DOS'
                
        except Exception as e:
            print(f"Error extracting packet details: {e}")
        
        return details

    def store_packet_for_analysis(self, packet, prediction_score):
        """
        Store packet details for later analysis, particularly for alert logs
        
        Parameters:
        -----------
        packet : Scapy packet
            The packet to store
        prediction_score : float
            The anomaly score from the model
        """
        if not hasattr(self, 'suspicious_packets'):
            self.suspicious_packets = []
            
        # Extract packet details
        details = self.extract_packet_details(packet)
        details['prediction_score'] = prediction_score
        
        # Store the packet details
        self.suspicious_packets.append(details)
        
        # Limit the history to the most recent packets
        max_history = 1000
        if len(self.suspicious_packets) > max_history:
            self.suspicious_packets = self.suspicious_packets[-max_history:]
            
    def log_detailed_alert(self, alert_type, packet_details):
        """
        Create a detailed log entry for an alert
        
        Parameters:
        -----------
        alert_type : str
            Type of alert (e.g., 'PORT_SCAN', 'POSSIBLE_DOS')
        packet_details : dict
            Dictionary with packet details
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create a descriptive alert message based on attack type
        if alert_type == 'PORT_SCAN':
            alert_msg = f"PORT SCAN detected from {packet_details['src_ip']} targeting {packet_details['dst_ip']}"
            severity = "HIGH" 
        elif alert_type == 'STEALTH_SCAN':
            alert_msg = f"STEALTH SCAN detected from {packet_details['src_ip']} using {packet_details['flags']} flags"
            severity = "HIGH"
        elif alert_type == 'PING_SCAN':
            alert_msg = f"PING SCAN detected from {packet_details['src_ip']}"
            severity = "MEDIUM"
        elif alert_type == 'POSSIBLE_DOS':
            alert_msg = f"Possible DoS attack from {packet_details['src_ip']} (packet size: {packet_details['size']} bytes)"
            severity = "HIGH"
        else:
            alert_msg = f"Anomalous traffic from {packet_details['src_ip']} to {packet_details['dst_ip']}"
            severity = "MEDIUM"
        
        # Format the detailed alert
        detailed_alert = f"[{timestamp}] {severity} ALERT: {alert_msg}"
        detailed_alert += f"\n  Source: {packet_details['src_ip']}:{packet_details['src_port']}"
        detailed_alert += f"\n  Destination: {packet_details['dst_ip']}:{packet_details['dst_port']}"
        detailed_alert += f"\n  Protocol: {packet_details['protocol']} Flags: {packet_details['flags']}"
        detailed_alert += f"\n  Size: {packet_details['size']} bytes"
        detailed_alert += f"\n  Anomaly Score: {packet_details['prediction_score']:.4f}"
        
        # Log to console with color
        print(f"\n🚨 {detailed_alert}")
        
        # Log to file
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{detailed_alert}\n\n")

    def packet_callback(self, packet):
        """
        Callback function for packet processing

        Parameters:
        -----------
        packet : Scapy packet
            Captured network packet
        """
        try:
            self.packet_count += 1
            self.total_packets += 1
            
            # Initialize trusted hosts if not already
            if not hasattr(self, 'trusted_hosts'):
                self.trusted_hosts = set()
                # Add localhost and common local addresses
                self.trusted_hosts.add('127.0.0.1')
                
                # Try to get the local IP
                for iface in netifaces.interfaces():
                    if iface != 'lo':
                        addrs = netifaces.ifaddresses(iface)
                        if netifaces.AF_INET in addrs:
                            for addr in addrs[netifaces.AF_INET]:
                                self.trusted_hosts.add(addr['addr'])
            
            # Check if it's an IP packet
            if IP in packet:
                # Store original packet for detailed logging if it's suspicious
                packet_copy = packet.copy()
                
                # Extract features from the packet
                features = self.extract_features_from_packet(packet)

                # Add to buffer with the original packet reference
                self.packet_buffer.append((features, packet_copy))

                # Process buffer if it reaches the threshold or every 10 seconds
                current_time = time.time()
                if len(self.packet_buffer) >= self.buffer_size or (current_time - self.start_time >= 10 and self.packet_buffer):
                    self.process_buffer()
                    
                # Periodically clean up old connections
                if self.packet_count % 1000 == 0:
                    self.cleanup_connections()
        except Exception as e:
            print(f"Error in packet callback: {e}")

    def cleanup_connections(self, max_age=300):
        """
        Clean up old connections from the tracking dictionary
        
        Parameters:
        -----------
        max_age : int
            Maximum age of connections in seconds
        """
        try:
            current_time = time.time()
            keys_to_remove = []
            
            for key, conn in self.connections.items():
                if current_time - conn['start_time'] > max_age:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self.connections[key]
                
            if keys_to_remove:
                print(f"Cleaned up {len(keys_to_remove)} old connections.")
        except Exception as e:
            print(f"Error cleaning up connections: {e}")

    def log_message(self, message):
        """Log a message to both console and log file"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{message}\n")

    def log_alert(self, alert_message, packet_details=None):
        """Log an alert with timestamp and details"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert = f"[{timestamp}] ALERT: {alert_message}"
        
        print("\n🚨 " + alert)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{alert}\n")
                if packet_details:
                    f.write(f"Details: {packet_details}\n\n")

    def process_buffer(self):
        """
        Process the buffered packets and detect anomalies
        """
        if not self.packet_buffer:
            return
            
        try:
            print(f"Processing {len(self.packet_buffer)} packets...")

            # Separate features and packets
            features_list = [item[0] for item in self.packet_buffer]
            packets_list = [item[1] for item in self.packet_buffer]

            # Preprocess each packet in the buffer
            processed_packets = []
            for features in features_list:
                processed_features = self.preprocess_packet_features(features)
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
                # Handle the case where there's only one sample
                if len(outputs.shape) == 0:
                    y_pred_prob = np.array([outputs.item()])
                else:
                    y_pred_prob = outputs.cpu().numpy()

            # Find anomalous packets
            anomalous_indices = np.where(y_pred_prob > self.alert_threshold)[0]
            
            # Count anomalous packets
            anomalous_packets = len(anomalous_indices)
            self.attack_count += anomalous_packets
            self.total_attacks += anomalous_packets

            # Process and log each anomalous packet
            if anomalous_packets > 0:
                alert_percentage = (anomalous_packets / len(self.packet_buffer)) * 100
                
                # General alert message
                print(f"\n🚨 Detected {anomalous_packets} potential intrusions ({alert_percentage:.2f}% of traffic)!")
                
                # Process each anomalous packet for detailed logging
                for idx in anomalous_indices:
                    # Get the original packet and extract details
                    original_packet = packets_list[idx]
                    details = self.extract_packet_details(original_packet)
                    details['prediction_score'] = y_pred_prob[idx]
                    
                    # Create detailed log based on attack type
                    if details['attack_type'] != 'unknown':
                        self.log_detailed_alert(details['attack_type'], details)
                    else:
                        # Generic anomaly
                        self.log_detailed_alert('ANOMALY', details)
            
            # Show periodic statistics and remaining time
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 10:  # Update stats every 10 seconds
                self.show_status_update()
            
            # Clear the buffer
            self.packet_buffer = []
            
        except Exception as e:
            print(f"Error processing buffer: {e}")
            import traceback
            traceback.print_exc()
            self.packet_buffer = []

    def show_status_update(self):
        """Show status update with statistics and remaining time"""
        elapsed_time = time.time() - self.start_time
        packets_per_second = self.packet_count / elapsed_time if elapsed_time > 0 else 0
        attack_percentage = (self.attack_count / self.packet_count) * 100 if self.packet_count > 0 else 0
        
        status_msg = "\n--- NIDS Status Update ---"
        status_msg += f"\nRuntime: {elapsed_time:.1f} seconds"
        status_msg += f"\nPackets analyzed: {self.packet_count} ({packets_per_second:.1f} packets/sec)"
        status_msg += f"\nPotential attacks detected: {self.attack_count} ({attack_percentage:.2f}%)"
        
        # Add remaining time if end_time is set
        if self.end_time:
            remaining = max(0, self.end_time - time.time())
            status_msg += f"\nRemaining time: {remaining:.1f} seconds"
        
        status_msg += "\n--------------------------"
        
        print(status_msg)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(status_msg + "\n")
        
        # Reset counters but keep the start time for next interval
        self.packet_count = 0
        self.attack_count = 0
        self.start_time = time.time()

    def timer_thread(self, timeout):
        """Thread to display remaining time periodically"""
        end_time = time.time() + timeout
        while time.time() < end_time and self.running:
            remaining = max(0, end_time - time.time())
            # Print remaining time without newlines to avoid cluttering
            sys.stdout.write(f"\rRemaining time: {remaining:.1f} seconds    ")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\r" + " " * 40 + "\r")  # Clear the line
        sys.stdout.flush()

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
        
        # Set up end time if timeout is specified
        if timeout:
            self.end_time = time.time() + timeout
            # Start timer display thread
            self.running = True
            timer_thread = threading.Thread(target=self.timer_thread, args=(timeout,))
            timer_thread.daemon = True
            timer_thread.start()
        
        # Create a completed flag to track when sniffing is done
        completed = False
        
        try:
            # Use AsyncSniffer to handle the case where we want to run a set time
            # regardless of whether packets are received
            if timeout:
                # We'll create a custom stop_filter to ensure we run for the full time
                def time_based_filter(packet):
                    return time.time() >= self.end_time
                
                sniffer = AsyncSniffer(
                    iface=self.interface,
                    prn=self.packet_callback,
                    filter="ip",
                    store=False,
                    stop_filter=time_based_filter
                )
                
                sniffer.start()
                
                # Wait for the timeout
                start = time.time()
                while time.time() - start < timeout:
                    time.sleep(0.1)  # Check every 100ms
                    # Process buffer periodically even if we don't get many packets
                    if len(self.packet_buffer) > 0 and time.time() - self.start_time > 5:
                        self.process_buffer()
                
                sniffer.stop()
                completed = True
            else:
                # If no timeout, use regular sniff
                sniff(iface=self.interface, prn=self.packet_callback, count=count, store=0, filter="ip")
                completed = True
                
        except KeyboardInterrupt:
            print("\nPacket capture stopped by user.")
        except socket.error as e:
            print(f"\nSocket error during packet capture: {e}")
            print("This might be due to insufficient permissions or interface issues.")
            print("Trying alternative capture method...")
            
            # Try with a different approach - using a more specific filter
            try:
                sniff(iface=self.interface, prn=self.packet_callback, count=count,
                     timeout=timeout, store=0, filter="ip and not host 127.0.0.1")
                completed = True
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")
        except Exception as e:
            print(f"\nError during packet capture: {e}")
        finally:
            self.running = False
            
            # Process any remaining packets in the buffer
            if self.packet_buffer:
                self.process_buffer()
                
            # Log completion
            if completed:
                completion_msg = "Packet capture completed successfully."
            else:
                completion_msg = "Packet capture ended with errors."
                
            print(completion_msg)
            
            # Final statistics
            total_elapsed = time.time() - (self.end_time - timeout) if timeout else 0
            final_stats = f"\n=== Final Statistics ===\n"
            final_stats += f"Total runtime: {total_elapsed:.1f} seconds\n"
            final_stats += f"Total packets analyzed: {self.total_packets}\n"
            final_stats += f"Total potential attacks detected: {self.total_attacks}\n"
            
            if self.total_packets > 0:
                attack_percentage = (self.total_attacks / self.total_packets) * 100
                final_stats += f"Overall attack percentage: {attack_percentage:.2f}%\n"
                
            print(final_stats)
            
            # Log to file
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write("\n" + completion_msg + "\n")
                    f.write(final_stats)
                    f.write(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n")
                print(f"Log file saved to {self.log_file}")

def get_available_interfaces():
    """Get a list of available network interfaces"""
    try:
        interfaces = netifaces.interfaces()
        # Filter out loopback interfaces
        interfaces = [i for i in interfaces if not i.startswith('lo')]
        return interfaces
    except Exception as e:
        print(f"Error getting network interfaces: {e}")
        return ['eth0']  # Default fallback

def check_interface_status(interface):
    """Check if an interface is up and running"""
    try:
        # Check if interface exists
        if interface not in netifaces.interfaces():
            print(f"Interface {interface} does not exist.")
            return False
        
        # Check if interface has an IP address
        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET not in addrs:
            print(f"Interface {interface} does not have an IPv4 address.")
            return False
            
        return True
    except Exception as e:
        print(f"Error checking interface status: {e}")
        return False

def generate_log_filename():
    """Generate a log filename with current timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"nids_log_{timestamp}.txt"

def main():
    print("=" * 80)
    print("NIDS Real-time Network Intrusion Detection System")
    print("=" * 80)
    
    # Import socket here to use it in error handling
    import socket
    
    # Check if model and scaler files exist
    model_path = 'nids_model.pth'
    scaler_path = 'nids_scaler.joblib'
    threshold_path = 'nids_threshold.txt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run 'train_nids_model.py' first to train and save the model.")
        return
        
    if not os.path.exists(scaler_path):
        print(f"Warning: Scaler file '{scaler_path}' not found.")
        print("Feature scaling may not be applied correctly.")
    
    if not os.path.exists(threshold_path):
        print(f"Warning: Threshold file '{threshold_path}' not found.")
        print("Using default threshold of 0.5.")
    
    # Get available network interfaces
    interfaces = get_available_interfaces()
    
    if not interfaces:
        print("No network interfaces found!")
        return
    
    # Display available interfaces and prompt for selection
    print("\nAvailable network interfaces:")
    for i, interface in enumerate(interfaces):
        status = "UP" if check_interface_status(interface) else "DOWN"
        print(f"{i+1}. {interface} ({status})")
    
    # Prompt for interface selection
    selected_interface = None
    while selected_interface is None:
        try:
            selection = input(f"\nSelect interface (1-{len(interfaces)}) or press Enter for default [{interfaces[0]}]: ")
            if selection.strip():
                idx = int(selection) - 1
                if 0 <= idx < len(interfaces):
                    selected_interface = interfaces[idx]
                    # Check if interface is up
                    if not check_interface_status(selected_interface):
                        print(f"Warning: Interface {selected_interface} may not be ready for packet capture.")
                        confirm = input("Continue anyway? (y/n): ")
                        if confirm.lower() != 'y':
                            selected_interface = None
                            continue
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(interfaces)}.")
                    continue
            else:
                selected_interface = interfaces[0]
                if not check_interface_status(selected_interface):
                    print(f"Warning: Interface {selected_interface} may not be ready for packet capture.")
                    confirm = input("Continue anyway? (y/n): ")
                    if confirm.lower() != 'y':
                        selected_interface = None
                        continue
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Prompt for capture duration
    timeout = None
    while timeout is None:
        try:
            timeout_input = input("\nEnter capture duration in seconds (press Enter for continuous capture): ")
            if timeout_input.strip():
                timeout = int(timeout_input)
                if timeout <= 0:
                    print("Duration must be a positive number.")
                    timeout = None
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    # Create log file
    log_file = generate_log_filename()
    print(f"Log will be saved to: {log_file}")
    
    # Start NIDS
    try:
        nids = NIDSPacketCapture(
            model_path=model_path,
            scaler_path=scaler_path,
            threshold_path=threshold_path,
            interface=selected_interface,
            log_file=log_file
        )
        
        # Add option to configure trusted hosts
        add_trusted = input("\nWould you like to add trusted hosts? (y/n): ")
        if add_trusted.lower() == 'y':
            while True:
                host = input("Enter trusted host IP (or press Enter to finish): ")
                if not host.strip():
                    break
                if not hasattr(nids, 'trusted_hosts'):
                    nids.trusted_hosts = set()
                nids.trusted_hosts.add(host)
                print(f"Added {host} to trusted hosts.")
        
        print("\nStarting NIDS packet capture...")
        print("NOTE: This requires administrator/root privileges in a real environment.")
        print("Press Ctrl+C to stop the capture at any time.")
        
        # Give the user a moment to read the info
        time.sleep(2)
        
        # Start the capture
        nids.start_capture(timeout=timeout)
    
    except Exception as e:
        print(f"Error running NIDS: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()