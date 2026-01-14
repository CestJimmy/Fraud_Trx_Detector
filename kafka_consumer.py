"""Kafka consumer for payment transactions."""
import json
import logging
from typing import Optional, Dict, Any
from confluent_kafka import Consumer, KafkaError, KafkaException
from config import Config

logger = logging.getLogger(__name__)


class KafkaConsumerClient:
    """Kafka consumer for reading payment transactions."""
    
    def __init__(self):
        """Initialize Kafka consumer."""
        self.config = Config.get_kafka_config()
        self.topic = Config.KAFKA_TOPIC
        self.consumer = None
        self._running = False
        logger.info(f"Kafka consumer initialized for topic: {self.topic}")
    
    def start(self):
        """Start the Kafka consumer."""
        try:
            self.consumer = Consumer(self.config)
            self.consumer.subscribe([self.topic])
            self._running = True
            logger.info(f"Kafka consumer started, subscribed to topic: {self.topic}")
        except KafkaException as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise
    
    def consume_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Consume a single message from Kafka.
        
        Args:
            timeout: Timeout in seconds for polling
            
        Returns:
            Parsed transaction dict or None if no message available
        """
        if not self.consumer:
            raise RuntimeError("Consumer not started. Call start() first.")
        
        try:
            msg = self.consumer.poll(timeout=timeout)
            
            if msg is None:
                return None
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug(f"Reached end of partition: {msg.partition()}")
                else:
                    logger.error(f"Kafka error: {msg.error()}")
                return None
            
            # Parse message value
            try:
                transaction = json.loads(msg.value().decode('utf-8'))
                
                # Add transaction ID if not present (use offset as ID)
                if 'transaction_id' not in transaction:
                    transaction['transaction_id'] = f"{msg.topic()}-{msg.partition()}-{msg.offset()}"
                
                logger.debug(f"Consumed transaction: {transaction['transaction_id']}")
                return transaction
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message as JSON: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error consuming message: {e}")
            return None
    
    def commit(self):
        """Manually commit current offsets."""
        if self.consumer:
            try:
                self.consumer.commit(asynchronous=False)
                logger.debug("Committed Kafka offsets")
            except KafkaException as e:
                logger.error(f"Failed to commit offsets: {e}")
    
    def stop(self):
        """Stop the Kafka consumer and close connection."""
        self._running = False
        if self.consumer:
            try:
                self.consumer.close()
                logger.info("Kafka consumer stopped")
            except Exception as e:
                logger.error(f"Error stopping consumer: {e}")
    
    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running
