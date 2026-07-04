import sys
import os
import uuid
import datetime
import random
import time
import threading
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from global_layer.global_server import GlobalServer
from provider_layer.provider_server import ProviderServer
from edge_layer.edge_device import EdgeDevice
from utils.auth_db import get_user, verify_password, get_password_hash, update_user_credentials
from byzantine_aggregators import TrimmedMeanAggregator, KrumAggregator, ByzantineEscrowMonitor

# JWT Configuration
SECRET_KEY = "your-super-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

security = HTTPBearer()

app = FastAPI(title="3-Tier Federated Learning Escrow Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory singletons
GLOBAL_SERVER = GlobalServer()
PROVIDERS = {
    "S1": ProviderServer("S1"),
    "S2": ProviderServer("S2"),
    "S3": ProviderServer("S3")
}
DEVICES = {}
TRANSFERS_LOG = []

# Differential Privacy Config State
DP_ENABLED = False
DP_NOISE = 0.1
DP_CLIP = 1.0

# FedProx / FedOpt Strategy Config State
FEDPROX_ENABLED = False
FEDPROX_MU = 0.01
SERVER_OPTIMIZER = "fedavg"
SERVER_LR = 0.1
SERVER_BETA1 = 0.9
SERVER_BETA2 = 0.99
SERVER_TAU = 1e-3

# Risk Threshold Config State
RISK_THRESHOLD = 0.65

# Byzantine-Robust Aggregation State
BYZANTINE_ENABLED = True

tier1_aggregator = TrimmedMeanAggregator(trim_fraction=0.15)
tier2_aggregator = KrumAggregator(num_byzantine=1, multi_k=2)

PROVIDER_NAMES = {"S1": "MTN", "S2": "Telecel", "S3": "AirtelTigo"}

def _escrow_hold_callback(provider_id: str, reason: str):
    print(f"[ESCROW HOLD] {provider_id}: {reason}")

byzantine_monitor = ByzantineEscrowMonitor(
    threshold=3,
    db_conn=None,
    escrow_callback=_escrow_hold_callback,
)

# Background Simulation State
SIMULATION_RUNNING = False
SIMULATION_THREAD = None

# Initialize devices and hook them to providers
for prov_id, provider in PROVIDERS.items():
    for client_idx in range(3):
        dev_id = f"{prov_id}-{client_idx}"
        device = EdgeDevice(dev_id)
        provider.add_edge_device(device)
        DEVICES[dev_id] = device

# Sync initial weights
g_sms, g_call = GLOBAL_SERVER.get_global_weights()
for dev in DEVICES.values():
    dev.set_model_weights(g_sms, g_call)

# Request Models
class ContactCreate(BaseModel):
    name: str
    phone: str
    is_trusted: bool = True
    risk_score: float = 0.0

class TransferRequest(BaseModel):
    sender_id: str
    receiver_phone: str
    amount: float
    message: str
    is_spam_simulation: Optional[bool] = None

class ThresholdConfigRequest(BaseModel):
    risk_threshold: float

class EscrowActionRequest(BaseModel):
    action: str

class DPConfigRequest(BaseModel):
    dp_enabled: bool
    dp_noise: float
    dp_clip: float

class FedStrategyConfigRequest(BaseModel):
    fedprox_enabled: bool = False
    fedprox_mu: float = 0.01
    server_optimizer: str = "fedavg"
    server_lr: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    tau: float = 1e-3

class LoginRequest(BaseModel):
    username: str
    password: str

class UpdateCredentialsRequest(BaseModel):
    current_password: str
    new_username: Optional[str] = None
    new_password: Optional[str] = None

# JWT Helpers
def create_access_token(username: str):
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials) -> str:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("username")
        if not isinstance(username, str):
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    return verify_token(credentials)

# Background Simulation
def run_background_transfers():
    global SIMULATION_RUNNING

    normal_msgs = [
        "Hi, are we still meeting at the market tomorrow morning?",
        "I have sent the invoice; please confirm receipt.",
        "Thanks for the update, see you at the office tomorrow.",
        "Can you send me the payment receipt for the shop?",
        "I am running a bit late, start the meeting without me.",
        "That works well for me. See you at the station later.",
        "Thanks for the dinner last night; it was lovely.",
        "I will be home in about 20 minutes.",
        "Let me know if you need any help with the project."
    ]
    spam_msgs = [
        "URGENT: Your MTN wallet is suspended. Click the link to verify your account details.",
        "Congratulations! You won a GH₵1000 prize. Click here to claim now.",
        "Official alert: Your Telecel bill is due immediately. Login to secure payment link.",
        "Action required: Verify your AirtelTigo credentials to avoid suspension.",
        "Security alert: Suspicious login attempt. Secure your account now link.",
        "Your parcel is at the depot. Please click the link to schedule delivery.",
        "IMPORTANT: Account verification required. Update your login profile now."
    ]

    while SIMULATION_RUNNING:
        try:
            sender_id = random.choice(list(DEVICES.keys()))
            device = DEVICES[sender_id]
            is_spam = random.random() < 0.35

            if is_spam:
                receiver_phone = random.choice(["+233200000000", "+233500000000", f"+233{random.randint(200000000, 999999999)}"])
                message = random.choice(spam_msgs)
                amount = random.uniform(150.0, 1500.0)
            else:
                contacts = device.contact_db.get_all_contacts()
                if contacts:
                    contact = random.choice(contacts)
                    receiver_phone = contact["phone"]
                else:
                    receiver_phone = "+15550199"
                message = random.choice(normal_msgs)
                amount = random.uniform(5.0, 200.0)

            req = TransferRequest(
                sender_id=sender_id,
                receiver_phone=receiver_phone,
                amount=round(amount, 2),
                message=message,
                is_spam_simulation=is_spam
            )
            _internal_transfer(req)

        except Exception as e:
            print(f"Error in background simulation loop: {e}")

        time.sleep(4.0)

def _internal_transfer(request: TransferRequest):
    if request.sender_id not in DEVICES:
        raise HTTPException(status_code=404, detail="Sender edge device not found")

    sender_device = DEVICES[request.sender_id]
    provider_id = request.sender_id.split("-")[0]
    provider = PROVIDERS[provider_id]

    risk_report = sender_device.assess_transfer_risk(
        request.receiver_phone, request.amount, request.message
    )

    transfer_id = str(uuid.uuid4())[:8]
    initiation_time = time.time()

    escrow_decision = provider.evaluate_escrow(
        transfer_id, request.sender_id, request.receiver_phone, request.amount, risk_report, message=request.message
    )
    
    # Apply dynamic threshold if heuristic agent doesn't auto-resolve
    if escrow_decision.get("status") == "APPROVED" and escrow_decision.get("reason", "") == "Passed security filters.":
        if risk_report["total_risk_score"] >= RISK_THRESHOLD:
            escrow_decision["status"] = "HELD_IN_ESCROW"
            escrow_decision["reason"] = f"Risk score {risk_report['total_risk_score']:.2f} exceeds threshold {RISK_THRESHOLD}"

    settlement_latency = 0.0 if escrow_decision["status"] != "HELD_IN_ESCROW" else None

    transaction = {
        **escrow_decision,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "initiation_time_sys": initiation_time,
        "settlement_latency": settlement_latency,
        "is_spam_simulation": request.is_spam_simulation,
        "stakeholder_type": getattr(sender_device, "stakeholder_type", "Subscriber")
    }
    TRANSFERS_LOG.insert(0, transaction)

    if len(TRANSFERS_LOG) > 500:
        TRANSFERS_LOG.pop()

    is_phishing = risk_report["sms_risk_score"] > RISK_THRESHOLD
    sender_device.message_db.add_message(
        request.receiver_phone, request.message, transaction["timestamp"], is_phishing
    )

    return transaction

# Auth endpoints
@app.post("/api/login")
def login(request: LoginRequest):
    user = get_user(request.username)
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(request.username)
    return {"access_token": access_token, "token_type": "bearer", "username": request.username}

@app.get("/api/users/me")
def get_current_user_info(user: str = Depends(get_current_user)):
    user_record = get_user(user)
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": user_record["username"]}

@app.put("/api/users/me")
def update_credentials(request: UpdateCredentialsRequest, user: str = Depends(get_current_user)):
    user_record = get_user(user)
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(request.current_password, user_record["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    new_username = request.new_username or user
    new_password = request.new_password or request.current_password
    success = update_user_credentials(user, new_username, new_password)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update credentials. Username may already exist.")
    access_token = create_access_token(new_username)
    return {"access_token": access_token, "token_type": "bearer", "username": new_username, "message": "Credentials updated successfully"}

# Device endpoints
@app.get("/api/devices")
def get_devices(user: str = Depends(get_current_user)):
    result = []
    for dev_id, dev in DEVICES.items():
        contacts = dev.contact_db.get_all_contacts()
        msg_counts = dev.message_db.get_message_counts()
        result.append({
            "device_id": dev_id,
            "provider_id": dev_id.split("-")[0],
            "contacts_count": len(contacts),
            "total_messages": msg_counts["total"],
            "spam_messages": msg_counts["spam"]
        })
    return result

@app.get("/api/devices/{device_id}/contacts")
def get_device_contacts(device_id: str, user: str = Depends(get_current_user)):
    if device_id not in DEVICES:
        raise HTTPException(status_code=404, detail="Device not found")
    return DEVICES[device_id].contact_db.get_all_contacts()

@app.post("/api/devices/{device_id}/contacts")
def add_device_contact(device_id: str, contact: ContactCreate, user: str = Depends(get_current_user)):
    if device_id not in DEVICES:
        raise HTTPException(status_code=404, detail="Device not found")
    success = DEVICES[device_id].contact_db.add_contact(
        contact.name, contact.phone, contact.is_trusted, contact.risk_score
    )
    if not success:
        raise HTTPException(status_code=400, detail="Phone number already exists in contact DB")
    return {"message": "Contact added successfully"}

@app.delete("/api/devices/{device_id}/contacts/{contact_id}")
def delete_device_contact(device_id: str, contact_id: int, user: str = Depends(get_current_user)):
    if device_id not in DEVICES:
        raise HTTPException(status_code=404, detail="Device not found")
    deleted = DEVICES[device_id].contact_db.delete_contact(contact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Contact not found")
    return {"message": "Contact deleted successfully"}

@app.post("/api/transfer")
def initiate_transfer(request: TransferRequest, user: str = Depends(get_current_user)):
    return _internal_transfer(request)

@app.get("/api/transfers")
def get_transfers(user: str = Depends(get_current_user)):
    return TRANSFERS_LOG

@app.get("/api/escrow")
def get_escrow(user: str = Depends(get_current_user)):
    escrow_list = []
    for provider in PROVIDERS.values():
        for record in provider.escrow_records.values():
            if record["status"] == "HELD_IN_ESCROW":
                escrow_list.append(record)
    return escrow_list

@app.post("/api/escrow/{transfer_id}/action")
def manage_escrow(transfer_id: str, request: EscrowActionRequest, user: str = Depends(get_current_user)):
    record = None
    target_provider = None
    for provider in PROVIDERS.values():
        if transfer_id in provider.escrow_records:
            record = provider.escrow_records[transfer_id]
            target_provider = provider
            break
    if not record:
        raise HTTPException(status_code=404, detail="Escrow record not found")
    if request.action not in ["RELEASE", "BLOCK"]:
        raise HTTPException(status_code=400, detail="Invalid escrow action. Choose 'RELEASE' or 'BLOCK'")
    resolved_record = target_provider.resolve_escrow(transfer_id, request.action)
    for idx, tx in enumerate(TRANSFERS_LOG):
        if tx["transfer_id"] == transfer_id:
            TRANSFERS_LOG[idx]["status"] = resolved_record["status"]
            if TRANSFERS_LOG[idx].get("initiation_time_sys"):
                TRANSFERS_LOG[idx]["settlement_latency"] = time.time() - TRANSFERS_LOG[idx]["initiation_time_sys"]
            break
    return resolved_record

@app.post("/api/federated/round")
def run_federated_round(user: str = Depends(get_current_user)):
    next_round = len(GLOBAL_SERVER.metrics_history)
    provider_updates = []
    byzantine_report = {}
    mu = FEDPROX_MU if FEDPROX_ENABLED else 0.0

    total_training_time = 0.0
    total_memory_mb = 0.0
    num_clients = 0

    # 1. Edge devices train locally
    for prov_id, provider in PROVIDERS.items():
        client_results = []
        for dev in provider.edge_devices:
            res = dev.train_local(
                epochs=1,
                lr=0.01,
                dp_enabled=DP_ENABLED,
                dp_noise_multiplier=DP_NOISE,
                dp_clip_norm=DP_CLIP,
                fedprox_mu=mu
            )
            client_results.append(res)
            total_training_time += res.get("training_time", 0.0)
            total_memory_mb += res.get("memory_mb", 0.0)
            num_clients += 1

        # 2. Provider aggregates locally
        p_sms_weights, p_call_weights = provider.aggregate_local_updates(client_results)

        # INSERT #3 — Tier 1→2 Trimmed Mean (inside for loop, before append)
        if BYZANTINE_ENABLED and len(client_results) >= 3:
            sms_weight_list = [r["sms_weights"] for r in client_results if "sms_weights" in r]
            if len(sms_weight_list) >= 3:
                robust_sms, trimmed_idx = tier1_aggregator.aggregate(
                    sms_weight_list,
                    provider_id=PROVIDER_NAMES.get(prov_id, prov_id),
                )
                if trimmed_idx:
                    flagged_ids = [f"{prov_id}-device-{i}" for i in trimmed_idx]
                    byzantine_monitor.record_flags(
                        flagged_ids, round_num=next_round,
                        reason="trimmed_mean_edge_rejection"
                    )
                    byzantine_report[prov_id] = {"trimmed_edge_devices": trimmed_idx}
                p_sms_weights = robust_sms

        provider_updates.append({
            "sms_weights": p_sms_weights,
            "call_weights": p_call_weights
        })

    # INSERT #4 — Tier 2→3 Multi-Krum (outside for loop, before global aggregation)
    if BYZANTINE_ENABLED and len(provider_updates) >= 2:
        provider_id_list = list(PROVIDERS.keys())
        sms_updates = [upd["sms_weights"] for upd in provider_updates]
        _, selected_idx, rejected_idx = tier2_aggregator.aggregate(
            sms_updates,
            provider_ids=[PROVIDER_NAMES.get(p, p) for p in provider_id_list],
        )
        if rejected_idx:
            rejected_names = [PROVIDER_NAMES.get(provider_id_list[i], provider_id_list[i])
                              for i in rejected_idx]
            byzantine_monitor.record_flags(
                rejected_names, round_num=next_round,
                reason="krum_provider_rejection"
            )
            byzantine_report["global"] = {"krum_rejected_providers": rejected_names}
        provider_updates = [provider_updates[i] for i in selected_idx]

    # 3. Global aggregation
    GLOBAL_SERVER.aggregate_provider_updates(provider_updates)

    # 4. Log metrics
    new_metrics = GLOBAL_SERVER.log_metrics(round_id=next_round)
    
    if num_clients > 0:
        new_metrics["avg_training_time_s"] = total_training_time / num_clients
        new_metrics["avg_memory_mb"] = total_memory_mb / num_clients
            
    # Run baselines on one device
    baseline_device = list(DEVICES.values())[0]
    baselines = baseline_device.train_baselines()
    if "error" not in baselines and "skipped" not in baselines:
        new_metrics["baseline_rf_acc"] = baselines.get("rf", {}).get("accuracy", 0.0)
        new_metrics["baseline_rf_f1"] = baselines.get("rf", {}).get("f1", 0.0)
        new_metrics["baseline_xgb_acc"] = baselines.get("xgb", {}).get("accuracy", 0.0)
        new_metrics["baseline_xgb_f1"] = baselines.get("xgb", {}).get("f1", 0.0)
    g_sms, g_call = GLOBAL_SERVER.get_global_weights()
    for dev in DEVICES.values():
        dev.set_model_weights(g_sms, g_call)

    # INSERT #5 — attach Byzantine report
    new_metrics["byzantine_report"] = byzantine_report
    
    # Calculate FP rate from recent transfers
    # False Positive: is_spam_simulation == False, but status is HELD_IN_ESCROW or BLOCKED
    recent_transfers = [t for t in TRANSFERS_LOG if t.get("is_spam_simulation") is not None]
    if recent_transfers:
        legit_transfers = [t for t in recent_transfers if t["is_spam_simulation"] is False]
        if legit_transfers:
            fps = sum(1 for t in legit_transfers if t["status"] in ("HELD_IN_ESCROW", "BLOCKED"))
            new_metrics["false_positive_rate"] = fps / len(legit_transfers)
        else:
            new_metrics["false_positive_rate"] = 0.0
    else:
        new_metrics["false_positive_rate"] = 0.0

    return new_metrics

@app.get("/api/federated/metrics")
def get_federated_metrics(user: str = Depends(get_current_user)):
    return GLOBAL_SERVER.metrics_history

@app.get("/api/federated/config")
def get_federated_config(user: str = Depends(get_current_user)):
    return {"dp_enabled": DP_ENABLED, "dp_noise": DP_NOISE, "dp_clip": DP_CLIP}

@app.post("/api/federated/config")
def update_federated_config(config: DPConfigRequest, user: str = Depends(get_current_user)):
    global DP_ENABLED, DP_NOISE, DP_CLIP
    DP_ENABLED = config.dp_enabled
    DP_NOISE = config.dp_noise
    DP_CLIP = config.dp_clip
    return {"message": "Federated DP configurations updated successfully."}

@app.get("/api/federated/strategy")
def get_fed_strategy(user: str = Depends(get_current_user)):
    return {
        "fedprox_enabled": FEDPROX_ENABLED,
        "fedprox_mu": FEDPROX_MU,
        "server_optimizer": SERVER_OPTIMIZER,
        "server_lr": SERVER_LR,
        "beta1": SERVER_BETA1,
        "beta2": SERVER_BETA2,
        "tau": SERVER_TAU
    }

@app.post("/api/federated/strategy")
def update_fed_strategy(config: FedStrategyConfigRequest, user: str = Depends(get_current_user)):
    global FEDPROX_ENABLED, FEDPROX_MU, SERVER_OPTIMIZER, SERVER_LR, SERVER_BETA1, SERVER_BETA2, SERVER_TAU
    FEDPROX_ENABLED = config.fedprox_enabled
    FEDPROX_MU = config.fedprox_mu
    SERVER_OPTIMIZER = config.server_optimizer.lower()
    SERVER_LR = config.server_lr
    SERVER_BETA1 = config.beta1
    SERVER_BETA2 = config.beta2
    SERVER_TAU = config.tau
    try:
        GLOBAL_SERVER.set_server_strategy(SERVER_OPTIMIZER, SERVER_LR, SERVER_BETA1, SERVER_BETA2, SERVER_TAU)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": f"Strategy updated: FedProx={'on' if FEDPROX_ENABLED else 'off'} (mu={FEDPROX_MU}), server_optimizer={SERVER_OPTIMIZER}"}

@app.post("/api/simulation/start")
def start_simulation(user: str = Depends(get_current_user)):
    global SIMULATION_RUNNING, SIMULATION_THREAD
    if SIMULATION_RUNNING:
        return {"status": "already running"}
    SIMULATION_RUNNING = True
    SIMULATION_THREAD = threading.Thread(target=run_background_transfers, daemon=True)
    SIMULATION_THREAD.start()
    return {"status": "started"}

@app.get("/api/config/threshold")
def get_threshold(user: str = Depends(get_current_user)):
    return {"risk_threshold": RISK_THRESHOLD}

@app.post("/api/config/threshold")
def update_threshold(config: ThresholdConfigRequest, user: str = Depends(get_current_user)):
    global RISK_THRESHOLD
    RISK_THRESHOLD = config.risk_threshold
    return {"message": f"Risk threshold updated to {RISK_THRESHOLD}"}

import csv
import io
from fastapi.responses import StreamingResponse

@app.get("/api/reports/export")
def export_reports(user: str = Depends(get_current_user)):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Timestamp", "Transfer ID", "Sender ID", "Receiver Phone", 
        "Amount", "Status", "Total Risk", "SMS Risk", "Contact Risk",
        "Decision By", "Reason", "Settlement Latency (s)", "Ground Truth Spam", "Stakeholder Type"
    ])
    for tx in TRANSFERS_LOG:
        writer.writerow([
            tx.get("timestamp", ""),
            tx.get("transfer_id", ""),
            tx.get("sender_id", ""),
            tx.get("receiver_phone", ""),
            tx.get("amount", ""),
            tx.get("status", ""),
            tx.get("risk_report", {}).get("total_risk_score", ""),
            tx.get("risk_report", {}).get("sms_risk_score", ""),
            tx.get("risk_report", {}).get("contact_risk_score", ""),
            tx.get("decision_by", ""),
            tx.get("reason", ""),
            f"{tx.get('settlement_latency', 0):.2f}" if tx.get("settlement_latency") is not None else "",
            tx.get("is_spam_simulation", ""),
            tx.get("stakeholder_type", "")
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=transfers_audit_report.csv"}
    )

@app.post("/api/simulation/stop")
def stop_simulation(user: str = Depends(get_current_user)):
    global SIMULATION_RUNNING
    SIMULATION_RUNNING = False
    return {"status": "stopped"}

@app.get("/api/simulation/status")
def get_simulation_status(user: str = Depends(get_current_user)):
    return {"running": SIMULATION_RUNNING}

# INSERT #6 — Byzantine audit endpoints (MUST be before app.mount)
@app.get("/api/byzantine/audit")
def get_byzantine_audit(provider_id: str = None, user: str = Depends(get_current_user)):
    return {
        "audit_log": byzantine_monitor.get_audit_log(provider_id),
        "flag_counts": byzantine_monitor._flag_counts,
    }

@app.post("/api/byzantine/clear/{provider_id}")
def clear_byzantine_flags(provider_id: str, user: str = Depends(get_current_user)):
    byzantine_monitor.clear_flags(provider_id)
    return {"status": "cleared", "provider_id": provider_id}

# Static mount LAST — must come after all API routes
ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")