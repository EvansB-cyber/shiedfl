# edge/sms_pipeline.py
import torch
from models.sms_fraud_cnn import SMSFraudCNN

RISK_THRESHOLD = 0.75

def score_sms(sms_text: str, model: SMSFraudCNN, tokenizer) -> float:
    tokens = tokenizer(sms_text)
    with torch.no_grad():
        logits = model(tokens)
        risk_score = torch.sigmoid(logits).item()
    return risk_score