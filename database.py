"""Database operations for fraud detection system."""
import pyodbc
import logging
from typing import List, Dict, Any
from contextlib import contextmanager
from config import Config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self):
        """Initialize database manager."""
        self.connection_string = Config.get_db_connection_string()
        self._connection = None
        logger.info("Database manager initialized")
    
    def connect(self):
        """Establish database connection."""
        try:
            self._connection = pyodbc.connect(self.connection_string)
            logger.info("Database connection established")
            return self._connection
        except pyodbc.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = pyodbc.connect(self.connection_string)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def insert_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Insert a single transaction into the database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                    INSERT INTO transactions (
                        transaction_id, step, type, amount, name_orig,
                        oldbalance_org, newbalance_orig, name_dest,
                        oldbalance_dest, newbalance_dest, is_fraud, is_flagged_fraud
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (transaction_id) DO NOTHING
                """
                
                cursor.execute(sql, (
                    transaction['transaction_id'],
                    transaction['step'],
                    transaction['type'],
                    transaction['amount'],
                    transaction['nameOrig'],
                    transaction['oldbalanceOrg'],
                    transaction['newbalanceOrig'],
                    transaction['nameDest'],
                    transaction['oldbalanceDest'],
                    transaction['newbalanceDest'],
                    transaction.get('isFraud', 0),
                    transaction.get('isFlaggedFraud', 0)
                ))
                
                return True
        except Exception as e:
            logger.error(f"Failed to insert transaction: {e}")
            return False
    
    def batch_insert_fraud_scores(self, scores: List[Dict[str, Any]]) -> int:
        """Batch insert fraud scores into the database."""
        if not scores:
            return 0
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                    INSERT INTO fraud_scores (
                        transaction_id, fraud_probability, is_fraud_prediction, model_version
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT (transaction_id) 
                    DO UPDATE SET 
                        fraud_probability = EXCLUDED.fraud_probability,
                        is_fraud_prediction = EXCLUDED.is_fraud_prediction,
                        scored_at = CURRENT_TIMESTAMP
                """
                
                # Prepare batch data
                batch_data = [
                    (
                        score['transaction_id'],
                        score['fraud_probability'],
                        score['is_fraud_prediction'],
                        score.get('model_version', '1.0')
                    )
                    for score in scores
                ]
                
                cursor.executemany(sql, batch_data)
                inserted_count = cursor.rowcount
                
                logger.info(f"Batch inserted {inserted_count} fraud scores")
                return inserted_count
                
        except Exception as e:
            logger.error(f"Failed to batch insert fraud scores: {e}")
            raise
    
    def get_transaction_stats(self) -> Dict[str, Any]:
        """Get statistics about processed transactions."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_transactions,
                        COUNT(CASE WHEN is_fraud_prediction = 1 THEN 1 END) as predicted_frauds,
                        AVG(fraud_probability) as avg_fraud_probability
                    FROM fraud_scores
                """)
                
                row = cursor.fetchone()
                return {
                    'total_transactions': row[0],
                    'predicted_frauds': row[1],
                    'avg_fraud_probability': float(row[2]) if row[2] else 0.0
                }
        except Exception as e:
            logger.error(f"Failed to get transaction stats: {e}")
            return {}
