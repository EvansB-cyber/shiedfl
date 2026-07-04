# Fraud Schemes Mapping

This document details how various mobile-money fraud schemes in Ghana map to the outputs and feature inputs of the 3-Tier Federated Learning models.

## 1. Fake Reversal / Accidental Transfer Scams
**Description**: Scammer sends an SMS claiming money was accidentally sent to the victim, then calls the victim to "reverse" the transaction, often guiding them through USSD codes to authorize a transfer to the scammer.
**Model Mapping**: 
- **SMSFraudCNN**: Detects urgency and reversal-related keywords in the initial text.
- **CallDetectionMLP**: High `times_called_today` combined with an unsaved contact and a suspicious `source_phone_risk_score`.

## 2. Voice Cloning (AI-based Social Engineering)
**Description**: Scammer uses AI to clone the voice of a trusted contact to request emergency funds.
**Model Mapping**:
- **CallDetectionMLP**: While it cannot process the audio, it flags anomalies such as unusual call `hour_of_day`, and irregular transfer amounts initiated immediately after the call.
- **Heuristics / Escrow**: The system captures `amount_risk_score` (unusually large transfers) which pushes the total risk score above the escrow threshold (0.65).

## 3. Fake Loan Scams
**Description**: Scammer offers "no collateral" loans but demands a processing fee upfront via MoMo.
**Model Mapping**:
- **SMSFraudCNN**: Trained to identify terms like "claim", "prize", "loan approval", and "processing fee". Highly effective at blocking the initial lure.

## 4. SMS Phishing (Smishing)
**Description**: Scammer sends texts pretending to be MTN, Telecel, or AirtelTigo warning of account suspension or requesting credential updates.
**Model Mapping**:
- **SMSFraudCNN**: Directly flags keywords like "verify", "suspend", "action required", "secure your account". 

## 5. SIM Swap Scams
**Description**: Scammer ports the victim's number and intercepts OTAs to empty wallets.
**Model Mapping**:
- **Heuristics**: If the system detects an uncharacteristic spike in transaction volume or a sudden change in trusted contacts, it elevates the `contact_risk_score` and flags the transfer to the Provider Escrow queue.

## 6. Data Bundle Scams
**Description**: Fraudulent texts claiming the user has won massive data bundles, requesting a small activation fee.
**Model Mapping**:
- **SMSFraudCNN**: Detects "congratulations", "winner", "claim", and data bundle-related keywords.
