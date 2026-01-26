"""
Temporal Motion Model: LSTM-based Motion Abnormality Detector
Complements rule-based analysis with deep learning temporal patterns.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

from config.config import MOTION_CONFIG


class MotionLSTM(nn.Module):
    """LSTM model for temporal motion abnormality detection."""
    
    def __init__(
        self, 
        input_size: int = 18,  # 9 joints × 2 coordinates
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features (joint coordinates)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(MotionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
        
        Returns:
            Abnormality score (0-1)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class TemporalMotionAnalyzer:
    """
    Temporal motion analyzer using LSTM for abnormality detection.
    Can work in two modes:
    1. Pretrained mode: Use trained LSTM model
    2. Fallback mode: Use rule-based analysis if model not available
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize temporal motion analyzer.
        
        Args:
            model_path: Path to pretrained LSTM model (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.use_lstm = False
        
        # Try to load pretrained model
        if model_path and Path(model_path).exists():
            try:
                self.model = MotionLSTM(
                    input_size=18,
                    hidden_size=MOTION_CONFIG["lstm_hidden_size"],
                    num_layers=MOTION_CONFIG["lstm_num_layers"],
                    dropout=MOTION_CONFIG["lstm_dropout"]
                )
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.use_lstm = True
                logger.info(f"LSTM model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load LSTM model: {e}. Using rule-based fallback.")
                self.use_lstm = False
        else:
            # Initialize untrained model for future use
            self.model = MotionLSTM(
                input_size=18,
                hidden_size=MOTION_CONFIG["lstm_hidden_size"],
                num_layers=MOTION_CONFIG["lstm_num_layers"],
                dropout=MOTION_CONFIG["lstm_dropout"]
            )
            self.model.to(self.device)
            logger.info("LSTM model initialized (untrained). Using rule-based analysis.")
        
        # Temporal buffer for sequence
        self.sequence_buffer = []
        self.sequence_length = MOTION_CONFIG["window_size"]
    
    def joints_to_vector(self, joints: Dict) -> Optional[np.ndarray]:
        """
        Convert joint dictionary to feature vector.
        
        Args:
            joints: Dictionary of joint positions
        
        Returns:
            Flattened numpy array of joint coordinates
        """
        logger.debug(f"[TEMPORAL LSTM] Converting joints to vector")
        if not joints:
            logger.debug(f"[TEMPORAL LSTM] No joints provided")
            return None
        
        # Expected joints (9 key joints)
        joint_names = [
            "nose", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]
        
        vector = []
        for joint_name in joint_names:
            if joint_name in joints:
                pos = joints[joint_name]["position"]
                vector.extend([pos[0], pos[1]])
            else:
                vector.extend([0.0, 0.0])  # Missing joint
        
        return np.array(vector, dtype=np.float32)
    
    def update_sequence(self, joints: Dict):
        """Update temporal sequence buffer."""
        vector = self.joints_to_vector(joints)
        if vector is not None:
            self.sequence_buffer.append(vector)
            
            # Keep only recent frames
            if len(self.sequence_buffer) > self.sequence_length:
                self.sequence_buffer.pop(0)
    
    def predict_abnormality(self, joints: Dict) -> float:
        """
        Predict motion abnormality score using LSTM.
        
        Args:
            joints: Current joint positions
        
        Returns:
            Abnormality score (0-100)
        """
        if not self.use_lstm or self.model is None:
            return 0.0  # Fallback to rule-based
        
        # Update sequence
        self.update_sequence(joints)
        
        # Need minimum sequence length
        if len(self.sequence_buffer) < 10:
            return 0.0
        
        try:
            # Prepare input tensor
            sequence = np.array(self.sequence_buffer[-self.sequence_length:])
            
            # Pad if needed
            if len(sequence) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(sequence), sequence.shape[1]))
                sequence = np.vstack([padding, sequence])
            
            # Convert to tensor
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(x)
                score = output.item() * 100  # Scale to 0-100
            
            return score
        
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return 0.0
    
    def reset(self):
        """Reset sequence buffer."""
        self.sequence_buffer.clear()


def train_motion_lstm(
    training_data: List[Tuple[np.ndarray, float]],
    model_save_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """
    Train LSTM model on labeled motion sequences.
    
    Args:
        training_data: List of (sequence, abnormality_label) tuples
        model_save_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    logger.info(f"Training LSTM model on {len(training_data)} sequences...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = MotionLSTM(
        input_size=18,
        hidden_size=MOTION_CONFIG["lstm_hidden_size"],
        num_layers=MOTION_CONFIG["lstm_num_layers"],
        dropout=MOTION_CONFIG["lstm_dropout"]
    )
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Shuffle data
        np.random.shuffle(training_data)
        
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            
            # Prepare batch
            sequences = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            
            # Convert to tensors
            x = torch.FloatTensor(np.array(sequences)).to(device)
            y = torch.FloatTensor(labels).unsqueeze(1).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(training_data) / batch_size)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    # Example usage
    logger.info("Temporal Motion LSTM module loaded")
    
    # Create dummy model for testing
    model = MotionLSTM()
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
