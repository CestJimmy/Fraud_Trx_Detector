import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
    KAFKA_TOPIC = os.getenv('KAFKA_TOPIC')
    KAFKA_GROUP_ID = os.getenv('KAFKA_GROUP_ID')
    KAFKA_AUTO_OFFSET_RESET = os.getenv('KAFKA_AUTO_OFFSET_RESET')
    
    # PostgreSQL Configuration
    DB_DRIVER = os.getenv('DB_DRIVER')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH')
    SCALER_PATH = os.getenv('SCALER_PATH')
    
    # Processing Configuration
    BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
    COMMIT_INTERVAL_SECONDS = int(os.getenv('COMMIT_INTERVAL_SECONDS'))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL')
    
    @classmethod
    def get_db_connection_string(cls):
        """Build PostgreSQL ODBC connection string."""
        return (
            f"DRIVER={{{cls.DB_DRIVER}}};"
            f"SERVER={cls.DB_HOST};"
            f"PORT={cls.DB_PORT};"
            f"DATABASE={cls.DB_NAME};"
            f"UID={cls.DB_USER};"
            f"PWD={cls.DB_PASSWORD};"
        )
    
    @classmethod
    def get_kafka_config(cls):
        """Get Kafka consumer configuration."""
        return {
            'bootstrap.servers': cls.KAFKA_BOOTSTRAP_SERVERS,
            'group.id': cls.KAFKA_GROUP_ID,
            'auto.offset.reset': cls.KAFKA_AUTO_OFFSET_RESET,
            'enable.auto.commit': False,  # Manual commit for exactly-once semantics
        }
