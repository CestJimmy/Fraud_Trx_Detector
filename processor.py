import logging
import signal
import sys
import time
from typing import List, Dict, Any
from kafka_consumer import KafkaConsumerClient
from fraud_model import FraudDetectionModel
from database import DatabaseManager
from config import Config

# Configure logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionProcessor:
    
    def __init__(self):
        self.kafka_consumer = KafkaConsumerClient()
        self.fraud_model = FraudDetectionModel()
        self.db_manager = DatabaseManager()
        self.running = False
        
        # Batch processing
        self.batch = []
        self.last_commit_time = time.time()
        
        logger.info("Fraud detection processor initialized")
    
    def setup(self):
        try:
            # Start Kafka consumer
            self.kafka_consumer.start()
            
            # Load fraud detection model
            if not self.fraud_model.load_model():
                logger.error("Failed to load fraud detection model")
                logger.info("The processor will continue, but predictions will be unavailable")
            
            logger.info("Setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    def process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single transaction.
        
        Args:
            transaction: Raw transaction from Kafka
            
        Returns:
            Fraud score dict
        """
        try:
            # Predict fraud
            fraud_prob, is_fraud = self.fraud_model.predict(transaction)
            
            # Create fraud score record
            score = {
                'transaction_id': transaction['transaction_id'],
                'fraud_probability': fraud_prob,
                'is_fraud_prediction': is_fraud,
                'model_version': '1.0'
            }
            
            # Store transaction in database
            self.db_manager.insert_transaction(transaction)
            
            return score
            
        except Exception as e:
            logger.error(f"Failed to process transaction {transaction.get('transaction_id')}: {e}")
            return None
    
    def flush_batch(self):
        if not self.batch:
            return
        
        try:
            inserted = self.db_manager.batch_insert_fraud_scores(self.batch)
            logger.info(f"Flushed batch of {inserted} fraud scores to database")
            
            self.kafka_consumer.commit()
            
            self.batch = []
            self.last_commit_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")
            # Don't clear batch on error - will retry
    
    def should_flush(self) -> bool:
        if len(self.batch) >= Config.BATCH_SIZE:
            return True
        
        if time.time() - self.last_commit_time >= Config.COMMIT_INTERVAL_SECONDS:
            return True
        
        return False
    
    def run(self):
        self.running = True
        logger.info("Starting fraud detection processing loop")
        
        messages_processed = 0
        
        while self.running:
            try:
                # Consume message from Kafka
                transaction = self.kafka_consumer.consume_message(timeout=1.0)
                
                if transaction:
                    # Process transaction
                    score = self.process_transaction(transaction)
                    
                    if score:
                        self.batch.append(score)
                        messages_processed += 1
                        
                        if messages_processed % 100 == 0:
                            logger.info(f"Processed {messages_processed} transactions")
                    
                    # Flush batch if needed
                    if self.should_flush():
                        self.flush_batch()
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1)  # Brief pause before continuing
        
        # Final flush before shutdown
        self.flush_batch()
        
        logger.info(f"Processing loop ended. Total messages processed: {messages_processed}")
    
    def shutdown(self):
        logger.info("Shutting down fraud detection processor")
        self.running = False
        self.flush_batch()
        self.kafka_consumer.stop()
        self.db_manager.disconnect()
        
        logger.info("Shutdown complete")


def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    if processor:
        processor.shutdown()
    sys.exit(0)


# Global processor instance for signal handler
processor = None


def main():
    global processor
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run processor
    processor = FraudDetectionProcessor()
    
    if not processor.setup():
        logger.error("Failed to set up processor, exiting")
        sys.exit(1)
    
    try:
        processor.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        processor.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
