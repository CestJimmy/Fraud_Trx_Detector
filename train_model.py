"""Train fraud detection model on transaction dataset."""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load transaction data from CSV file."""
    logger.info(f"Loading data from {filepath}")
    
    try:
        # Limit to 1 million rows to avoid OOM in dev environment
        # The user requested downsampling/subsetting for better performance
        chunk_size = 1000000
        logger.info(f"Reading first {chunk_size} rows for training...")
        df = pd.read_csv(filepath, nrows=chunk_size)
        logger.info(f"Loaded {len(df)} transactions")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw transaction data."""
    logger.info("Engineering features...")
    
    # Make a copy
    df = df.copy()
    
    # One-hot encode transaction type
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, type_dummies], axis=1)
    
    # Customer type indicators (M = Merchant, C = Customer)
    df['is_merchant_orig'] = df['nameOrig'].str.startswith('M').astype(int)
    df['is_merchant_dest'] = df['nameDest'].str.startswith('M').astype(int)
    
    # Derived features
    df['balance_change_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Amount to balance ratio (handle division by zero)
    df['amount_to_balance_ratio'] = np.where(
        df['oldbalanceOrg'] > 0,
        df['amount'] / df['oldbalanceOrg'],
        0
    )
    
    # Drop non-numeric and identifier columns
    columns_to_drop = ['type', 'nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    logger.info(f"Engineered features. Final shape: {df.shape}")
    logger.info(f"Features: {[col for col in df.columns if col != 'isFraud']}")
    
    return df


def train_model(X_train, y_train, X_test, y_test, model_type='random_forest'):
    """Train and evaluate a fraud detection model."""
    logger.info(f"Training {model_type} model...")
    
    # Initialize model based on type
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"\nModel: {model_type}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        logger.info(f"\nTop 10 Feature Importances:\n{feature_importance.head(10)}")
    
    return model, {'f1': f1, 'auc': auc}


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("Starting Fraud Detection Model Training Pipeline")
    logger.info("="*60)
    
    # Paths
    data_path = os.path.join('data', 'transactions.csv')
    model_output_path = os.path.join('model', 'fraud_detector.pkl')
    scaler_output_path = os.path.join('model', 'scaler.pkl')
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    
    # Check data balance
    fraud_count = df['isFraud'].sum()
    total_count = len(df)
    fraud_rate = fraud_count / total_count * 100
    logger.info(f"\nData Balance:")
    logger.info(f"Total transactions: {total_count}")
    logger.info(f"Fraud transactions: {fraud_count} ({fraud_rate:.2f}%)")
    logger.info(f"Legitimate transactions: {total_count - fraud_count} ({100-fraud_rate:.2f}%)")
    
    # Engineer features
    df_processed = engineer_features(df)
    
    # Separate features and target
    X = df_processed.drop('isFraud', axis=1)
    y = df_processed['isFraud']
    
    # Split data
    logger.info("\nSplitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Handle class imbalance with undersampling
    if fraud_rate < 10 or len(X_train) > 100000:  
        logger.info("\nApplying RandomUnderSampler to handle class imbalance (memory efficient)...")
        from imblearn.under_sampling import RandomUnderSampler
        
        # Keep all fraud samples, undersample legitimate to 10:1 ratio
        rus = RandomUnderSampler(random_state=42, sampling_strategy=0.1)
        X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
        logger.info(f"After undersampling - Train set: {len(X_train_balanced)} samples")
        logger.info(f"Fraud samples: {y_train_balanced.sum()}")
        logger.info(f"Legitimate samples: {len(y_train_balanced) - y_train_balanced.sum()}")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Scale features
    logger.info("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Train multiple models and compare
    logger.info("\n" + "="*60)
    logger.info("Training and comparing multiple models")
    logger.info("="*60)
    
    models = {}
    metrics = {}
    
    for model_type in ['logistic', 'random_forest', 'gradient_boosting']:
        model, model_metrics = train_model(
            X_train_scaled, y_train_balanced, 
            X_test_scaled, y_test, 
            model_type=model_type
        )
        models[model_type] = model
        metrics[model_type] = model_metrics
        logger.info("-"*60)
    
    # Select best model based on F1-score
    best_model_type = max(metrics, key=lambda k: metrics[k]['f1'])
    best_model = models[best_model_type]
    best_metrics = metrics[best_model_type]
    
    logger.info("\n" + "="*60)
    logger.info(f"Best Model: {best_model_type}")
    logger.info(f"F1-Score: {best_metrics['f1']:.4f}")
    logger.info(f"AUC-ROC: {best_metrics['auc']:.4f}")
    logger.info("="*60)

    # Save metrics to text file for comparison
    metrics_output_path = os.path.join('model', 'performance_metrics.txt')
    logger.info(f"Saving performance metrics to {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Fraud Detection Model Performance Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Model: {best_model_type}\n")
        f.write(f"F1-Score: {best_metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC: {best_metrics['auc']:.4f}\n\n")
        
        f.write("All Models Comparison:\n")
        f.write("-" * 40 + "\n")
        for m_type, m_metrics in metrics.items():
            f.write(f"Model: {m_type}\n")
            f.write(f"  F1-Score: {m_metrics['f1']:.4f}\n")
            f.write(f"  AUC-ROC:  {m_metrics['auc']:.4f}\n")
            f.write("-" * 40 + "\n")

    # Save model and scaler
    logger.info(f"\nSaving model to {model_output_path}")
    with open(model_output_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    logger.info(f"Saving scaler to {scaler_output_path}")
    with open(scaler_output_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info("\n" + "="*60)
    logger.info("Training pipeline completed successfully!")
    logger.info("="*60)
    
    return best_model, scaler


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
