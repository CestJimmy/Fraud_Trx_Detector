"""Fraud detection model wrapper for real-time inference."""
import os
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from config import Config

logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """Wrapper for fraud detection ML model."""
    
    def __init__(self):
        """Initialize the model wrapper."""
        self.model = None
        self.scaler = None
        self.model_path = Config.MODEL_PATH
        self.scaler_path = Config.SCALER_PATH
        self.is_loaded = False
        logger.info("Fraud detection model wrapper initialized")
    
    def load_model(self):
        """Load the trained model and scaler from disk."""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}. Train the model first.")
                return False
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_loaded = True
            logger.info("Model and scaler loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def preprocess_transaction(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess transaction data for model prediction.
        
        Args:
            transaction: Raw transaction dict
            
        Returns:
            Preprocessed DataFrame ready for model
        """
        # Extract features
        features = {
            'step': transaction.get('step', 0),
            'amount': transaction.get('amount', 0.0),
            'oldbalanceOrg': transaction.get('oldbalanceOrg', 0.0),
            'newbalanceOrig': transaction.get('newbalanceOrig', 0.0),
            'oldbalanceDest': transaction.get('oldbalanceDest', 0.0),
            'newbalanceDest': transaction.get('newbalanceDest', 0.0),
        }
        
        # Transaction type one-hot encoding
        tx_type = transaction.get('type', 'PAYMENT')
        type_features = {
            'type_CASH_IN': 1 if tx_type == 'CASH_IN' else 0,
            'type_CASH_OUT': 1 if tx_type == 'CASH_OUT' else 0,
            'type_DEBIT': 1 if tx_type == 'DEBIT' else 0,
            'type_PAYMENT': 1 if tx_type == 'PAYMENT' else 0,
            'type_TRANSFER': 1 if tx_type == 'TRANSFER' else 0,
        }
        features.update(type_features)
        
        # Customer type indicators
        name_orig = transaction.get('nameOrig', '')
        name_dest = transaction.get('nameDest', '')
        features['is_merchant_orig'] = 1 if name_orig.startswith('M') else 0
        features['is_merchant_dest'] = 1 if name_dest.startswith('M') else 0
        
        # Derived features
        features['balance_change_orig'] = features['oldbalanceOrg'] - features['newbalanceOrig']
        features['balance_change_dest'] = features['newbalanceDest'] - features['oldbalanceDest']
        
        # Amount to balance ratio (avoid division by zero)
        if features['oldbalanceOrg'] > 0:
            features['amount_to_balance_ratio'] = features['amount'] / features['oldbalanceOrg']
        else:
            features['amount_to_balance_ratio'] = 0.0
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        return df
    
    def predict(self, transaction: Dict[str, Any]) -> Tuple[float, int]:
        """
        Predict fraud probability for a transaction.
        
        Args:
            transaction: Raw transaction dict
            
        Returns:
            Tuple of (fraud_probability, is_fraud_prediction)
        """
        if not self.is_loaded:
            if not self.load_model():
                logger.warning("Model not loaded, returning default prediction")
                return 0.0, 0
        
        try:
            # Preprocess
            X = self.preprocess_transaction(transaction)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            fraud_prob = self.model.predict_proba(X_scaled)[0][1]  # Probability of class 1 (fraud)
            is_fraud = 1 if fraud_prob >= 0.5 else 0
            
            logger.debug(f"Prediction - Probability: {fraud_prob:.4f}, Is Fraud: {is_fraud}")
            
            return float(fraud_prob), int(is_fraud)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0, 0
    
    def batch_predict(self, transactions: list) -> list:
        """
        Predict fraud for multiple transactions.
        
        Args:
            transactions: List of transaction dicts
            
        Returns:
            List of (fraud_probability, is_fraud_prediction) tuples
        """
        if not self.is_loaded:
            if not self.load_model():
                return [(0.0, 0) for _ in transactions]
        
        try:
            # Preprocess all transactions
            dfs = [self.preprocess_transaction(tx) for tx in transactions]
            X = pd.concat(dfs, ignore_index=True)
            
            # Scale and predict
            X_scaled = self.scaler.transform(X)
            fraud_probs = self.model.predict_proba(X_scaled)[:, 1]
            is_frauds = (fraud_probs >= 0.5).astype(int)
            
            return list(zip(fraud_probs.tolist(), is_frauds.tolist()))
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [(0.0, 0) for _ in transactions]
