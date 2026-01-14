
# IMPORTANT DISCLAIMER:
LLM assistance has only been used for unit testing, documenting the code and exception handling.

## 1. SQL Schema

This is the database schema used for determining fraud scores and storing transactions.

```sql
-- Transactions table: stores raw transaction data
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE NOT NULL,
    step INTEGER NOT NULL,
    type VARCHAR(20) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    name_orig VARCHAR(50) NOT NULL,
    oldbalance_org DECIMAL(15, 2),
    newbalance_orig DECIMAL(15, 2),
    name_dest VARCHAR(50) NOT NULL,
    oldbalance_dest DECIMAL(15, 2),
    newbalance_dest DECIMAL(15, 2),
    is_fraud INTEGER,
    is_flagged_fraud INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_transaction_id (transaction_id),
    INDEX idx_created_at (created_at),
    INDEX idx_type (type)
);

-- Fraud scores table: stores ML model predictions
CREATE TABLE fraud_scores (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE NOT NULL,
    fraud_probability DECIMAL(5, 4) NOT NULL,
    is_fraud_prediction INTEGER NOT NULL,
    model_version VARCHAR(20) DEFAULT '1.0',
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key to transactions table
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    
    INDEX idx_transaction_id (transaction_id),
    INDEX idx_fraud_prediction (is_fraud_prediction),
    INDEX idx_fraud_probability (fraud_probability),
    INDEX idx_scored_at (scored_at)
);

-- Create view for easy access to transactions with their scores
CREATE OR REPLACE VIEW transaction_scores AS
SELECT 
    t.transaction_id,
    t.step,
    t.type,
    t.amount,
    t.name_orig,
    t.name_dest,
    t.is_fraud AS actual_fraud,
    fs.fraud_probability,
    fs.is_fraud_prediction AS predicted_fraud,
    fs.scored_at,
    t.created_at
FROM transactions t
LEFT JOIN fraud_scores fs ON t.transaction_id = fs.transaction_id;
```

## 2. Sample Kafka Messages

These are simulated JSON messages that would be produced by ShadowTraffic to the `payment-transactions` topic.

**Sample 1: Standard Payment (No Fraud)**
```json
{
  "step": 1,
  "type": "PAYMENT",
  "amount": 9839.64,
  "nameOrig": "C123456789",
  "oldbalanceOrg": 170136.00,
  "newbalanceOrig": 160296.36,
  "nameDest": "M987654321",
  "oldbalanceDest": 0.00,
  "newbalanceDest": 0.00,
  "isFraud": 0,
  "isFlaggedFraud": 0
}
```

**Sample 2: Large Transfer (Flagged as Fraud)**
```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 250000.00,
  "nameOrig": "C555555555",
  "oldbalanceOrg": 250000.00,
  "newbalanceOrig": 0.00,
  "nameDest": "C666666666",
  "oldbalanceDest": 0.00,
  "newbalanceDest": 250000.00,
  "isFraud": 1,
  "isFlaggedFraud": 1
}
```

**Sample 3: Cash Out (Suspicious but not flagged)**
```json
{
  "step": 2,
  "type": "CASH_OUT",
  "amount": 80000.50,
  "nameOrig": "C777777777",
  "oldbalanceOrg": 80000.50,
  "newbalanceOrig": 0.00,
  "nameDest": "C888888888",
  "oldbalanceDest": 0.00,
  "newbalanceDest": 80000.50,
  "isFraud": 0,
  "isFlaggedFraud": 0
}
```

## All Models Comparison:
```
**Best Model**: random_forest
F1-Score: 0.4846
AUC-ROC: 0.9948
----------------------------------------**
Model: logistic
  F1-Score: 0.0090
  AUC-ROC:  0.9716
----------------------------------------**
Model: random_forest
  F1-Score: 0.4846
  AUC-ROC:  0.9948
----------------------------------------**
Model: gradient_boosting
  F1-Score: 0.4280
  AUC-ROC:  0.9948
----------------------------------------**

```
## 3. Project README

# Real-Time Transaction Fraud Detection System

A production-ready fraud detection system that processes payment transactions in real-time using Apache Kafka, machine learning, and PostgreSQL.

## Architecture

```
ShadowTraffic (Simulate real-time data streaming) → Kafka → Fraud Detector App → PostgreSQL
                  ↓            ↓
              Topic:      ML Model
          payment-
         transactions
```

**Components:**
- **ShadowTraffic**: Generates realistic transaction data
- **Apache Kafka**: Message broker for real-time streaming
- **Fraud Detector App**: Python application with ML model
- **PostgreSQL**: Database for transactions and fraud scores

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.10+ (for local development)

### 1. Train the Fraud Detection Model

```bash
# Install dependencies
pip install -r requirements.txt

# Train model on existing dataset
python model/train_model.py
```

This will:
- Load the transaction dataset (`data/transactions.csv`)
- Engineer features and handle class imbalance with SMOTE
- Train and compare multiple models (Logistic, Random Forest, Gradient Boosting)
- Save the best model to `model/fraud_detector.pkl`

### 2. Start All Services

```bash
# Start infrastructure (Kafka, PostgreSQL, ShadowTraffic, Fraud Detector)
docker-compose up -d

# View logs
docker-compose logs -f fraud-detector
```

### 3. Monitor Processing

```bash
# Check fraud detector logs
docker-compose logs -f fraud-detector

# Query PostgreSQL for results
docker exec -it fraud_trx_detector_postgres psql -U postgres -d fraud_detection -c "SELECT * FROM transaction_scores LIMIT 10;"

# Get statistics
docker exec -it fraud_trx_detector_postgres psql -U postgres -d fraud_detection -c "SELECT COUNT(*), AVG(fraud_probability), SUM(CASE WHEN is_fraud_prediction=1 THEN 1 ELSE 0 END) as frauds FROM fraud_scores;"
```

## Dataset Structure

The system expects transaction data with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Time unit (1 step = 1 hour) |
| `type` | string | Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER) |
| `amount` | float | Transaction amount |
| `nameOrig` | string | Customer who initiated transaction |
| `oldbalanceOrg` | float | Balance before transaction |
| `newbalanceOrig` | float | Balance after transaction |
| `nameDest` | string | Transaction recipient |
| `oldbalanceDest` | float | Recipient balance before |
| `newbalanceDest` | float | Recipient balance after |
| `isFlaggedFraud` | int | Flagged if amount > 200,000 |

## Feature Engineering

The model uses the following feature engineering techniques to transform raw transaction data into model-ready features:

1.  **One-Hot Encoding**: The `type` categorical field is converted into binary columns:
    *   `type_CASH_IN`
    *   `type_CASH_OUT`
    *   `type_DEBIT`
    *   `type_PAYMENT`
    *   `type_TRANSFER`

2.  **Merchant Identifiers**: Extracted from `nameOrig` and `nameDest` to indicate if the party is a merchant (starts with 'M'):
    *   `is_merchant_orig`: 1 if originator is a Merchant, else 0
    *   `is_merchant_dest`: 1 if recipient is a Merchant, else 0

3.  **Balance Changes**: Calculated to capture the net effect of the transaction:
    *   `balance_change_orig` = `oldbalanceOrg` - `newbalanceOrig`
    *   `balance_change_dest` = `newbalanceDest` - `oldbalanceDest`

4.  **Transaction Ratios**:
    *   `amount_to_balance_ratio` = `amount` / `oldbalanceOrg` (Set to 0 if old balance is 0)

5.  **Data Cleaning**:
    *   Original `type`, `nameOrig`, and `nameDest` columns are dropped after feature extraction.
    *   `isFlaggedFraud` is removed as it's a rule-based flag, not a predictive feature.

## Configuration

Environment variables can be set in `.env` file (File is not uploaded to the public server as it contains the actual license id and parameters for shadow traffic and kafka):

```bash
# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=payment-transactions
KAFKA_GROUP_ID=fraud-detector-group

# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fraud_detection
DB_USER=postgres
DB_PASSWORD=password

# Processing
BATCH_SIZE=100
COMMIT_INTERVAL_SECONDS=5
```
# Logging
LOG_LEVEL=INFO


# Shadowtraffic license info (It is real-time data streaming simulator)
LICENSE_ID= [ENCRYPTION_KEY]
LICENSE_EMAIL= [EMAIL_ADDRESS]
LICENSE_ORGANIZATION= [Organization_name]
LICENSE_EDITION=[ShadowTraffic_Free_Trial]
LICENSE_EXPIRATION=[2026-02-14]
LICENSE_SIGNATURE=[provided_by_shadowtraffic]

## Project Structure

```
Fraud_Trx_Detector/
├── src/                    # Application source code
│   ├── config.py          # Configuration management
│   ├── database.py        # PostgreSQL operations
│   ├── kafka_consumer.py  # Kafka consumer
│   ├── fraud_model.py     # ML model wrapper
│   └── processor.py       # Main processing pipeline
├── model/                 # ML model artifacts
│   ├── train_model.py    # Training script
│   ├── fraud_detector.pkl # Trained model
│   └── scaler.pkl        # Feature scaler
├── data/                  # Data files
│   └── transactions.csv  # Training dataset
├── shadowtraffic/         # Data simulation config
│   └── config.json       # ShadowTraffic configuration
├── sql/                   # Database schemas
│   └── schema.sql        # PostgreSQL table definitions runs only once
├── docker-compose.yml     # Multi-service orchestration
├── Dockerfile            # Application container
└── requirements.txt      # Python dependencies
```

## Testing

### Unit Testing
```bash
python -m pytest tests/
```

### End-to-End Testing
```bash
# Start all services
docker-compose up -d

# Wait for services to be healthy
sleep 10

# Verify ShadowTraffic is generating data
docker-compose logs shadowtraffic | grep "Sent"

# Check database has records
docker exec -it fraud_trx_detector_postgres psql -U postgres -d fraud_detection -c "SELECT COUNT(*) FROM fraud_scores;"
```

## Performance Considerations

- **Throughput**: Processes ~100 transactions/second
- **Latency**: <200ms per transaction (includes DB write)
- **Batching**: Configurable batch size (default: 100) for optimal throughput

- **Scaling**: Can run multiple fraud-detector instances with same consumer group

