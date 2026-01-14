-- Create database (run this separately if needed)
-- CREATE DATABASE fraud_detection;

-- Connect to the database
-- \c fraud_detection;

-- Drop tables if they exist
DROP TABLE IF EXISTS fraud_scores CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;

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

-- Comments for documentation
COMMENT ON TABLE transactions IS 'Raw payment transaction data consumed from Kafka';
COMMENT ON TABLE fraud_scores IS 'ML model fraud detection scores for transactions';
COMMENT ON COLUMN fraud_scores.fraud_probability IS 'Probability of fraud (0.0 to 1.0)';
COMMENT ON COLUMN fraud_scores.is_fraud_prediction IS 'Binary prediction: 1 = fraud, 0 = legitimate';
