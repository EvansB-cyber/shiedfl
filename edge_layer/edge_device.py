import torch
import torch.nn as nn
import torch.optim as optim
from .models import SMSFraudCNN, CallDetectionMLP
from .data import get_dataloaders, tokenize_message
from .contact_db import ContactDatabase
from .message_db import MessageDatabase
import os
import sys
import time
import tracemalloc
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.crypto import encrypt_weights, decrypt_weights

class EdgeDevice:
    """
    Tier 3: Local Edge Client.
    Manages local models, local databases, trains locally, and runs real-time risk assessment.
    """
    def __init__(self, device_id, db_folder="C:/Users/Appau Robert Atsu/.gemini/antigravity/scratch/project/edge_layer/databases"):
        self.device_id = device_id
        self.db_path = os.path.join(db_folder, f"contacts_{device_id}.db")
        self.stakeholder_type = random.choice(["Subscriber", "Agent"])
        
        # Initialize databases
        self.contact_db = ContactDatabase(self.db_path)
        self.message_db = MessageDatabase(self.db_path)
        
        # Instantiate local models
        self.sms_model = SMSFraudCNN()
        self.call_model = CallDetectionMLP()
        
        # Set up loss and optimizers
        self.criterion = nn.CrossEntropyLoss()
        
        # Seed initial data for demonstration if empty
        self.seed_local_dbs()

    def seed_local_dbs(self):
        """
        Seeds contact list and message history to simulate user activity.
        """
        # Contacts Seed List
        contacts_seed = [
            ("Akosua Boateng", "+233541234567", 1, 0.0),
            ("Kwame Asante", "+233552345678", 1, 0.0),
            ("Ama Mensah", "+233553456789", 1, 0.0)
        ]
        # S1-0, S2-0, S3-0 get a flagged/scam number added
        if self.device_id.endswith("-0"):
            contacts_seed.append(("Unknown Fraud", "+233200000000", 0, 0.95))
        elif self.device_id.endswith("-1"):
            contacts_seed.append(("Suspicious Sender", "+233500000000", 0, 0.60))
            
        self.contact_db.seed_initial_contacts(contacts_seed)

        # Messages Seed List
        messages_seed = [
            ("+233541234567", "Hi, are we still meeting at the market tomorrow morning?", "2026-06-13 08:00:00", 0),
            ("+233552345678", "Thanks for the payment update, see you at the office tomorrow.", "2026-06-13 08:15:00", 0)
        ]
        if self.device_id.endswith("-0"):
            messages_seed.append(("+233200000000", "URGENT: Your MTN wallet is suspended. Click the link to verify your account now.", "2026-06-13 08:30:00", 1))
        elif self.device_id.endswith("-1"):
            messages_seed.append(("+233500000000", "Official alert: Your AirtelTigo bill is due today. Login to secure your account link.", "2026-06-13 08:45:00", 1))

        self.message_db.seed_initial_messages(messages_seed)

    def set_model_weights(self, sms_encrypted, call_encrypted):
        """
        Loads the encrypted weights distributed by the server (Global/Provider).
        """
        sms_state_dict = decrypt_weights(sms_encrypted)
        call_state_dict = decrypt_weights(call_encrypted)
        self.sms_model.load_state_dict(sms_state_dict)
        self.call_model.load_state_dict(call_state_dict)

    @staticmethod
    def _proximal_term(model, global_params, mu):
        """
        FedProx proximal regularization term: (mu/2) * sum_i ||w_i - w_global_i||^2

        Penalizes local model drift away from the global model received at the
        start of the round. This stabilizes training under non-IID client data
        distributions by discouraging any single client's local update from
        moving too far from the shared global model, which is the failure mode
        plain FedAvg suffers from on heterogeneous data (Li et al., MLSys 2020).
        """
        prox_term = 0.0
        for name, param in model.named_parameters():
            if name in global_params:
                prox_term = prox_term + torch.sum((param - global_params[name]) ** 2)
        return (mu / 2.0) * prox_term

# ════════════════════════════════════════════════════════════════════
#  PATCH: Replace the train_local method in edge_layer/edge_device.py
#  Changes:
#   1. Adds class-weighted CrossEntropyLoss for the call model so the
#      minority fraud class is not overwhelmed by legitimate calls.
#   2. Uses a lower learning rate for the call model (0.005 vs 0.01)
#      to avoid overshooting on a small imbalanced dataset.
#   3. Adds more local epochs for the call model (3 vs 1) so it
#      has enough passes to learn the minority pattern.
# ════════════════════════════════════════════════════════════════════

    def train_local(self, epochs=1, lr=0.01, batch_size=8,
                    dp_enabled=False, dp_noise_multiplier=0.1, dp_clip_norm=1.0,
                    fedprox_mu=0.0, fl_algorithm=None,
                    global_sms_state=None, global_call_state=None):
        import copy
        start_time = time.time()
        tracemalloc.start()
        
        old_sms_state = copy.deepcopy(self.sms_model.state_dict())
        old_call_state = copy.deepcopy(self.call_model.state_dict())

        if global_sms_state is not None:
            global_sms_params = {k: v.clone().detach() for k, v in global_sms_state.items() if v.dtype.is_floating_point}
        else:
            global_sms_params = {k: v.clone().detach() for k, v in old_sms_state.items() if v.dtype.is_floating_point}

        if global_call_state is not None:
            global_call_params = {k: v.clone().detach() for k, v in global_call_state.items() if v.dtype.is_floating_point}
        else:
            global_call_params = {k: v.clone().detach() for k, v in old_call_state.items() if v.dtype.is_floating_point}

        sms_loader, call_loader = get_dataloaders(self.device_id, batch_size=batch_size, num_samples=80)

        # ── 1. Train SMS Model (unchanged) ───────────────────────────────────
        self.sms_model.train()
        sms_criterion = nn.CrossEntropyLoss()
        sms_optimizer = optim.Adam(self.sms_model.parameters(), lr=lr)
        sms_loss_total = 0.0
        sms_steps = 0
        for epoch in range(epochs):
            for texts, labels in sms_loader:
                sms_optimizer.zero_grad()
                outputs = self.sms_model(texts)
                loss = sms_criterion(outputs, labels)
                if fedprox_mu > 0.0:
                    loss = loss + self._proximal_term(self.sms_model, global_sms_params, fedprox_mu)
                loss.backward()
                sms_optimizer.step()
                sms_loss_total += loss.item()
                sms_steps += 1
        avg_sms_loss = sms_loss_total / max(1, sms_steps)

        # ── 2. Train Call Model (class-weighted loss) ─────────────────────────
        # WHY: Call fraud is rare (~13% of samples). Standard CrossEntropyLoss
        # lets the model cheat by predicting "legitimate" for everything and
        # still get ~87% accuracy. Weighting makes fraud mistakes 6× more costly,
        # forcing the model to actually learn the fraud pattern.
        #
        # Weight formula: w_class = total_samples / (num_classes * count_of_class)
        # e.g. if 80 samples: 69 legit (class 0), 11 fraud (class 1)
        #   w0 = 80 / (2 * 69) ≈ 0.58   w1 = 80 / (2 * 11) ≈ 3.64
        self.call_model.train()

        # Count class frequencies from this loader pass
        all_call_labels = []
        for _, labels in call_loader:
            all_call_labels.extend(labels.tolist())

        n_total = len(all_call_labels)
        n_fraud = sum(1 for l in all_call_labels if l == 1)
        n_legit = n_total - n_fraud

        # Avoid division by zero if a class is completely absent
        w_legit = n_total / (2.0 * n_legit) if n_legit > 0 else 1.0
        w_fraud = n_total / (2.0 * n_fraud) if n_fraud > 0 else 1.0

        import torch
        call_weights = torch.tensor([w_legit, w_fraud], dtype=torch.float)
        call_criterion = nn.CrossEntropyLoss(weight=call_weights)

        # Use slightly lower lr + more epochs for the call model
        call_lr = lr * 0.5
        call_epochs = max(epochs, 3)   # at least 3 passes regardless of round setting
        call_optimizer = optim.Adam(self.call_model.parameters(), lr=call_lr)

        call_loss_total = 0.0
        call_steps = 0
        for epoch in range(call_epochs):
            for features, labels in call_loader:
                call_optimizer.zero_grad()
                outputs = self.call_model(features)
                loss = call_criterion(outputs, labels)
                if fedprox_mu > 0.0:
                    loss = loss + self._proximal_term(self.call_model, global_call_params, fedprox_mu)
                loss.backward()
                call_optimizer.step()
                call_loss_total += loss.item()
                call_steps += 1
        avg_call_loss = call_loss_total / max(1, call_steps)

        sms_w = self.sms_model.state_dict()
        call_w = self.call_model.state_dict()

        if dp_enabled:
            for k in sms_w.keys():
                delta = sms_w[k] - old_sms_state[k]
                clipped_delta = torch.clamp(delta, -dp_clip_norm, dp_clip_norm)
                if delta.dtype.is_floating_point:
                    noise = torch.randn_like(delta) * (dp_noise_multiplier * dp_clip_norm)
                    sms_w[k] = old_sms_state[k] + clipped_delta + noise
                else:
                    sms_w[k] = old_sms_state[k] + clipped_delta

            for k in call_w.keys():
                delta = call_w[k] - old_call_state[k]
                clipped_delta = torch.clamp(delta, -dp_clip_norm, dp_clip_norm)
                if delta.dtype.is_floating_point:
                    noise = torch.randn_like(delta) * (dp_noise_multiplier * dp_clip_norm)
                    call_w[k] = old_call_state[k] + clipped_delta + noise
                else:
                    call_w[k] = old_call_state[k] + clipped_delta

            self.sms_model.load_state_dict(sms_w)
            self.call_model.load_state_dict(call_w)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        training_time = time.time() - start_time
        memory_mb = peak / 1024 / 1024

        return {
            "sms_weights_encrypted": encrypt_weights(sms_w),
            "call_weights_encrypted": encrypt_weights(call_w),
            "sms_loss": avg_sms_loss,
            "call_loss": avg_call_loss,
            "fedprox_mu": fedprox_mu,
            "training_time": training_time,
            "memory_mb": memory_mb
        }

    def train_baselines(self):
        """
        Trains baseline non-DL models (Random Forest, XGBoost) on the local call dataset
        for comparative false positive analysis.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            import xgboost as xgb
            import numpy as np
        except ImportError:
            return {"error": "scikit-learn or xgboost not installed."}

        _, call_loader = get_dataloaders(self.device_id, batch_size=80, num_samples=80)
        
        X_list, y_list = [], []
        for features, labels in call_loader:
            X_list.append(features.numpy())
            y_list.append(labels.numpy())
        
        if not X_list:
            return {"error": "No data"}
            
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # Skip if only one class is present locally
        if len(np.unique(y)) < 2:
            return {"status": "skipped", "reason": "Only one class present in local data"}

        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        rf_preds = rf.predict(X)
        rf_metrics = {
            "accuracy": accuracy_score(y, rf_preds),
            "f1": f1_score(y, rf_preds, zero_division=0)
        }

        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X, y)
        xgb_preds = xgb_model.predict(X)
        xgb_metrics = {
            "accuracy": accuracy_score(y, xgb_preds),
            "f1": f1_score(y, xgb_preds, zero_division=0)
        }

        return {"rf": rf_metrics, "xgb": xgb_metrics}

    def assess_transfer_risk(self, receiver_phone, amount, message_text=""):
        """
        Runs real-time local model evaluation to predict transaction risk.
        Computes scores from:
          1. Local database contact status
          2. SMS body phishing analysis (via SMSFraudCNN)
          3. Transaction volume risk heuristics (amount)
        """
        self.sms_model.eval()
        
        # 1. Contact database check
        contact = self.contact_db.get_contact_by_phone(receiver_phone)
        if contact:
            if contact["is_trusted"]:
                contact_risk = 0.0
            else:
                contact_risk = contact["risk_score"]
        else:
            # Unknown sender/receiver gets neutral-high default risk
            contact_risk = 0.40

        # 2. Message risk check (Phishing text analyzer)
        sms_risk = 0.0
        if message_text:
            tokens = tokenize_message(message_text)
            tokens_tensor = torch.tensor([tokens], dtype=torch.long) # Add batch dimension
            with torch.no_grad():
                logits = self.sms_model(tokens_tensor)
                probs = torch.softmax(logits, dim=1)
                sms_risk = probs[0, 1].item() # probability of class 1 (fraud)

        # 3. Transaction size risk check (heuristic)
        # Small transfers (under 50) are low risk, large transfers (above 1000) are high risk
        if amount <= 50:
            amount_risk = 0.0
        elif amount >= 1000:
            amount_risk = 0.8
        else:
            amount_risk = (amount - 50) / 950 * 0.8

        # Weighted combination: message text takes precedence if it contains strong phishing flags
        if sms_risk > 0.7:
            total_risk = max(sms_risk, 0.4 * contact_risk + 0.4 * sms_risk + 0.2 * amount_risk)
        else:
            total_risk = 0.3 * contact_risk + 0.5 * sms_risk + 0.2 * amount_risk
            
        return {
            "device_id": self.device_id,
            "receiver_phone": receiver_phone,
            "amount": amount,
            "message_text": message_text,
            "contact_risk_score": round(contact_risk, 3),
            "sms_risk_score": round(sms_risk, 3),
            "amount_risk_score": round(amount_risk, 3),
            "total_risk_score": round(total_risk, 3)
        }