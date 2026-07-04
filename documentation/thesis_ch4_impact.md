# Chapter 4: False Positives & Social Impact

## 1. The Cost of False Positives in Mobile Money

In the context of Ghanaian mobile money (MoMo), a "False Positive" occurs when a legitimate transaction is flagged as fraudulent. While maximizing the detection of true fraud (True Positives) is critical, an excessively aggressive fraud detection model can have devastating social and economic impacts.

### 1.1 Social and Economic Ramifications
Mobile money in Ghana is not merely an alternative to banking; for many individuals in rural and underserved areas, it is the *primary* financial lifeline. A blocked MoMo wallet or an escrowed transaction can result in:
- **Medical Emergencies:** Inability to pay for critical healthcare or medicine at hospitals that do not accept credit cards.
- **Business Disruption:** Market traders rely on MoMo for daily inventory purchases. A false positive can freeze their operating capital, leading to lost revenue.
- **Loss of Trust:** Repeated false flags may drive users back to physical cash, undermining the Bank of Ghana's financial inclusion and cashless society initiatives.

## 2. Escrow Authority as a Mitigation Strategy

To balance the need for high-security fraud detection with the socio-economic risks of false positives, our architecture introduces a **Provider-Level Escrow Authority**. 

Rather than executing a hard "BLOCK" on suspicious transactions, the model assigns a continuous risk score. If the score breaches a configured threshold (e.g., 0.65), the transaction is placed into a `HELD_IN_ESCROW` status.

### 2.1 The Escrow Workflow
1. **Edge Detection:** The sender's local device evaluates the transaction using its federated SMS and Call models, producing `sms_risk_score` and `contact_risk_score`.
2. **Provider Evaluation:** The Provider node calculates the `total_risk_score`. If > 0.65, the funds are debited from the sender but *not* credited to the receiver.
3. **Human-in-the-loop Validation:** An authorized agent at the telecom provider (MTN, Telecel, AirtelTigo) investigates the escrowed transfer via the Escrow Center portal. They can contact the sender to verify the transaction's legitimacy.
4. **Resolution:** The agent manually clicks **Release** (completing the transfer) or **Block** (refunding the sender and flagging the receiver).

This human-in-the-loop mechanism acts as a safety net. It allows the deep learning models to maintain a high sensitivity (catching more fraud) without subjecting users to irreversible fund freezes.

## 3. Threshold Tuning and Performance Tracking

The system dashboard provides a dynamic **Risk Escrow Threshold** slider. 
- **Lowering the threshold** (e.g., 0.40) increases security but floods the Escrow queue, requiring more human operators and increasing the False Positive Rate.
- **Raising the threshold** (e.g., 0.85) reduces the burden on operators and minimizes False Positives but allows sophisticated scams to slip through.

The global metrics logger tracks the False Positive (FP) rate per federated round, allowing network administrators to empirically tune this threshold based on live traffic behavior.
