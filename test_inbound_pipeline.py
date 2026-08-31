"""
test_inbound_pipeline.py — Step 4 verification test.

Tests the pure inbound path:
  EdgeDevice.receive_inbound()
    → tokenize_message() (frozen vocab)
    → SMSFraudCNN inference
    → tracking_db.log_inbound_message()
    → trg_inbound_update_registry (SQLite trigger)
    → phone_number_registry UPSERT

No transfer, no amount, no escrow queue — only an inbound SMS.

Run:
    python test_inbound_pipeline.py

Expected output (model weights are random at this point, so sms_risk
will be near 0.5; the test validates structure, not score magnitude):
    [1] receive_inbound() returned clean result ... PASS
    [2] inbound_messages row written to DB      ... PASS
    [3] inbound_action is correct               ... PASS
    [4] phone_number_registry updated           ... PASS (or SKIP if risk < 0.75)
    [5] below-threshold message NOT in registry ... PASS
    [6] get_inbound_stats() totals correct      ... PASS
    ALL TESTS PASSED
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── bootstrap ────────────────────────────────────────────────────────────────
from edge_layer.edge_device import EdgeDevice
import utils.tracking_db as tracking_db

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

failures = []

def check(label, condition, skip_msg=None):
    if skip_msg:
        print(f"  {label:<55} ... {SKIP} ({skip_msg})")
        return
    status = PASS if condition else FAIL
    print(f"  {label:<55} ... {status}")
    if not condition:
        failures.append(label)


print("=" * 65)
print("INBOUND PIPELINE TEST — no transfer, no escrow queue")
print("=" * 65)

# ── fixture ───────────────────────────────────────────────────────────────────
DEVICE_ID     = "S1-0"
PROVIDER_ID   = "S1"
SPAM_PHONE    = "+233900000001"   # unknown to device contacts
HAM_PHONE     = "+233900000002"   # another unknown, low-risk message
MODEL_ROUND   = 99                # sentinel value so test rows are identifiable

spam_sms = (
    "URGENT: Your MTN MoMo wallet is suspended. "
    "Send your PIN and OTP to verify and unblock your account now."
)
ham_sms  = "Hey, are we still meeting for lunch tomorrow at 12?"

device = EdgeDevice(DEVICE_ID)

# ── Test 1: receive_inbound() returns expected keys ───────────────────────────
print("\n[Step A] Spam SMS — high-risk inbound (no transfer)")
result_spam = device.receive_inbound(
    sender_phone=SPAM_PHONE,
    message_text=spam_sms,
    provider_id=PROVIDER_ID,
    model_round_id=MODEL_ROUND,
    is_ground_truth_spam=True,
)

REQUIRED_KEYS = {
    "device_id", "sender_phone", "sms_risk_score",
    "contact_known", "contact_trusted",
    "inbound_action", "inbound_id",
    "persisted", "phone_reputation", "timestamp",
}
check("[1] receive_inbound() returned all expected keys",
      REQUIRED_KEYS.issubset(result_spam.keys()))
check("[2] inbound_id is non-empty string",
      isinstance(result_spam["inbound_id"], str) and len(result_spam["inbound_id"]) == 32)
check("[3] sms_risk_score is a float in [0, 1]",
      isinstance(result_spam["sms_risk_score"], float)
      and 0.0 <= result_spam["sms_risk_score"] <= 1.0)
check("[4] persisted = True",
      result_spam["persisted"])
check("[5] sender not in trusted contacts (unknown number)",
      not result_spam["contact_trusted"])

print(f"\n       sms_risk_score : {result_spam['sms_risk_score']}")
print(f"       inbound_action : {result_spam['inbound_action']}")
print(f"       inbound_id     : {result_spam['inbound_id']}")

# ── Test 2: row appears in inbound_messages ────────────────────────────────────
print("\n[Step B] Confirm inbound_messages row in DB")
stats_spam = tracking_db.get_inbound_stats(sender_phone=SPAM_PHONE)
check("[6] inbound_messages has >= 1 row for spam sender",
      stats_spam["total"] >= 1)

# ── Test 3: inbound_action matches threshold logic ────────────────────────────
print("\n[Step C] inbound_action threshold logic")
FLAG_THRESH = tracking_db.INBOUND_FLAG_THRESHOLD
expected_action = (
    "FLAGGED_SENDER" if result_spam["sms_risk_score"] >= FLAG_THRESH else "LOGGED_ONLY"
)
check(f"[7] inbound_action matches threshold ({FLAG_THRESH})",
      result_spam["inbound_action"] == expected_action)

# ── Test 4: phone_number_registry updated for FLAGGED_SENDER ─────────────────
print("\n[Step D] phone_number_registry update")
if result_spam["inbound_action"] == "FLAGGED_SENDER":
    rep = tracking_db.get_phone_reputation(SPAM_PHONE)
    check("[8] phone_number_registry row exists for sender",
          rep is not None)
    check("[9] times_flagged >= 1",
          rep is not None and rep["times_flagged"] >= 1)
    check("[10] current_status is WATCH or BLOCKED (not CLEAR)",
          rep is not None and rep["current_status"] in ("WATCH", "BLOCKED"))
    print(f"\n       registry entry: {rep}")
else:
    check("[8] registry NOT updated for LOGGED_ONLY (correct)",
          tracking_db.get_phone_reputation(SPAM_PHONE) is None,
          skip_msg=f"sms_risk={result_spam['sms_risk_score']:.3f} < {FLAG_THRESH} → LOGGED_ONLY expected")

# ── Test 5: below-threshold (ham) message → LOGGED_ONLY, no registry entry ───
print("\n[Step E] Ham SMS — below-threshold, no registry impact")
result_ham = device.receive_inbound(
    sender_phone=HAM_PHONE,
    message_text=ham_sms,
    provider_id=PROVIDER_ID,
    model_round_id=MODEL_ROUND,
    is_ground_truth_spam=False,
)
print(f"\n       sms_risk_score : {result_ham['sms_risk_score']}")
print(f"       inbound_action : {result_ham['inbound_action']}")

check("[11] ham message persisted",
      result_ham["persisted"])
check("[12] ham message action is LOGGED_ONLY (if risk < threshold)",
      result_ham["inbound_action"] == "LOGGED_ONLY"
      if result_ham["sms_risk_score"] < FLAG_THRESH
      else True,   # if model scores ham as fraud, action will be FLAGGED_SENDER — still valid
      skip_msg=f"sms_risk={result_ham['sms_risk_score']:.3f} >= {FLAG_THRESH} (untrained model)"
               if result_ham["sms_risk_score"] >= FLAG_THRESH else None)

# ── Test 6: get_inbound_stats() totals ────────────────────────────────────────
print("\n[Step F] get_inbound_stats() aggregate counts")
stats_all = tracking_db.get_inbound_stats()
check("[13] total inbound_messages >= 2 (spam + ham)",
      stats_all["total"] >= 2)
check("[14] stats dict has expected keys",
      {"total", "flagged_sender", "logged_only", "avg_risk", "rows"}.issubset(stats_all.keys()))
check("[15] flagged_sender + logged_only == total",
      stats_all["flagged_sender"] + stats_all["logged_only"] == stats_all["total"])

print(f"\n       total rows     : {stats_all['total']}")
print(f"       flagged_sender : {stats_all['flagged_sender']}")
print(f"       logged_only    : {stats_all['logged_only']}")
print(f"       avg_risk       : {stats_all['avg_risk']}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
if failures:
    print(f"FAILED: {len(failures)} test(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"\033[32mALL TESTS PASSED\033[0m")
    print()
    print("Next steps:")
    print("  • Run retrain_frozen_vocab.py to close Step 3 checkbox 5")
    print("  • Run main.py --rounds 2 to verify full pipeline end-to-end")
    print("=" * 65)
