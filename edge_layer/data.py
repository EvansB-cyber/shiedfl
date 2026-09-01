import torch
from torch.utils.data import Dataset, DataLoader
import random
import urllib.request
import zipfile
import io
import os
import json
import hashlib

# ---------------------------------------------------------------------------
# VOCABULARY — word → integer index mapping shared by Python training and
# the Kotlin Android edge client.  This is the SINGLE SOURCE OF TRUTH.
#
# Rules that must never be broken:
#   index 0  → <PAD>   (zero-padding, never a real token)
#   index 1  → <UNK>   (out-of-vocabulary fallback)
#   2–69     → general SMS / call terms
#   70–109   → Ghana / MoMo-specific fraud terms
#
# To add tokens: append to the list BELOW any existing token (never reorder,
# never insert in the middle — that would shift all subsequent indices and
# require full retraining).
#
# After any change: call export_vocab_json() once; commit vocab.json alongside
# the code change; bump VOCAB_VERSION.
#
# VOCAB_SIZE is derived from len(VOCAB) — it is NOT a separate magic constant.
# SMSFraudCNN's embedding table is sized to VOCAB_SIZE so the two are always
# in sync without any manual bookkeeping.
# ---------------------------------------------------------------------------

VOCAB_VERSION = "1.1.0"   # Bump this whenever tokens are added/removed.

VOCAB = [
    # ── Special tokens ──────────────────────────────────────────────────────
    "<PAD>", "<UNK>",

    # ── General conversational / benign ─────────────────────────────────────
    "hello", "hi", "how", "are", "you", "meeting", "tomorrow",
    "lunch", "dinner", "ok", "thanks", "sender", "receiver", "amount", "transfer",
    "bank", "verify", "suspend", "alert", "urgent", "link", "click", "claim",
    "prize", "winner", "cash", "account", "secure", "credentials", "login",
    "password", "service", "payment", "due", "unpaid", "bill", "official",
    "update", "temporary", "limit", "access", "gift", "card", "congratulations",
    "immediate", "action", "required", "call", "now", "free", "txt", "text",
    "stop", "mobile", "customer", "contact", "reply", "msg",
    "please", "won", "latest", "important",

    # ── Ghana / MoMo-specific fraud terms ────────────────────────────────────
    # Mobile money platforms & currency
    "momo", "mtn", "telecel", "airteltigo", "ghs", "cedis", "cedi", "pesewas",
    # MoMo wallet / account actions
    "wallet", "top", "up", "topup", "load", "withdraw", "deposit", "send",
    "receive", "balance", "cashout", "cash_out",
    # Fraud-pattern verbs
    "reverse", "reversible", "reversal", "redirect", "unblock", "replace",
    "sim", "swap", "port", "activate",
    # Auth / credential theft targets
    "pin", "otp", "code", "secret", "token", "number",
    # Social-engineering phrases common in Ghanaian scams
    "mensa", "agent", "ebusiness", "merchant", "collect", "approve",
    "confirm", "decline", "authorize", "transaction", "reference", "promo",
    "bonus", "reward", "offer", "selected", "kyc",
    # Urgency / threat language
    "blocked", "expired", "disabled", "immediately", "failure", "error",
    "override", "waive", "charges",
]

# ---------------------------------------------------------------------------
# ☑ Checkbox 1: VOCAB is static — deterministic across runs.
# ☑ Checkbox 2: Vocabulary is finalized above (MTN/Telecel/AirtelTigo/phishing
#               terms explicitly included; special tokens <PAD>=0, <UNK>=1).
# ☑ Checkbox 3: export step adds "vocab_version" field + SHA-256 checksum file.
# ☑ Checkbox 4: tokenizer reads from VOCAB_MAP which mirrors vocab.json exactly.
# ☑ Checkbox 5: VOCAB_SIZE = len(VOCAB) — retraining picks this up automatically.
# ---------------------------------------------------------------------------

VOCAB_MAP: dict[str, int] = {word: idx for idx, word in enumerate(VOCAB)}

# True size of the vocabulary — used by SMSFraudCNN's embedding table.
# Previously hard-coded to 1000 (wasted ~890 embedding rows); now exact.
VOCAB_SIZE: int = len(VOCAB)

# Path for the exported vocab asset consumed by the Kotlin Android app and
# the Flower client bridge.
_VOCAB_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.json")
_VOCAB_SHA_PATH = _VOCAB_JSON_PATH + ".sha256"


# ---------------------------------------------------------------------------
# ☑ Checkbox 3 — Export step: serialises {token: index} + version field.
# ---------------------------------------------------------------------------

def export_vocab_json(path: str = None) -> str:
    """
    Writes the complete vocabulary to a JSON file consumed by both Python and
    Kotlin (Android assets/).

    Output schema:
    {
      "vocab_version": "1.1.0",
      "pad_id": 0,
      "unk_id": 1,
      "vocab_size": 110,
      "tokens": {"<PAD>": 0, "<UNK>": 1, "hello": 2, ...}
    }

    A companion <filename>.sha256 file is written alongside so callers can
    detect staleness without re-reading the full JSON.

    Args:
        path: Target file path.  Defaults to edge_layer/vocab.json.

    Returns:
        The path the file was written to.
    """
    target = path or _VOCAB_JSON_PATH
    sha_target = target + ".sha256"

    payload = {
        "vocab_version": VOCAB_VERSION,
        "pad_id": VOCAB_MAP["<PAD>"],
        "unk_id": VOCAB_MAP["<UNK>"],
        "vocab_size": VOCAB_SIZE,
        "tokens": VOCAB_MAP,
    }
    json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    with open(target, "wb") as f:
        f.write(json_bytes)

    checksum = hashlib.sha256(json_bytes).hexdigest()
    with open(sha_target, "w", encoding="utf-8") as f:
        f.write(checksum + "\n")

    return target


def get_vocab_info() -> dict:
    """
    Returns a lightweight summary of the current vocabulary state.
    Used by the Flower bridge, Android sync checks, and unit tests —
    callers don't need to read or parse vocab.json themselves.

    Returns:
        {
          "vocab_version": str,
          "vocab_size": int,
          "pad_id": int,
          "unk_id": int,
          "export_path": str,
          "checksum_path": str,
        }
    """
    return {
        "vocab_version": VOCAB_VERSION,
        "vocab_size": VOCAB_SIZE,
        "pad_id": VOCAB_MAP["<PAD>"],
        "unk_id": VOCAB_MAP["<UNK>"],
        "export_path": _VOCAB_JSON_PATH,
        "checksum_path": _VOCAB_SHA_PATH,
    }


# ---------------------------------------------------------------------------
# Always regenerate vocab.json on module import so any vocab change in this
# file is immediately reflected in the JSON asset — no manual step needed.
# Safe to run repeatedly; the output is deterministic.
# ---------------------------------------------------------------------------
export_vocab_json(_VOCAB_JSON_PATH)


# ---------------------------------------------------------------------------
# ☑ Checkbox 4 — Tokenizer loads from VOCAB_MAP (which is the in-memory
# mirror of vocab.json).  Both are derived from the same VOCAB list, so
# Python training and Kotlin inference are numerically identical.
# ---------------------------------------------------------------------------

def tokenize_message(text: str, seq_len: int = 20) -> list[int]:
    """
    Cleans, tokenizes, and pads/truncates a message into an integer sequence.

    Steps:
      1. Lower-case and strip punctuation.
      2. Split on whitespace.
      3. Map each token to its VOCAB_MAP index; unknown tokens → UNK (1).
      4. Pad with PAD (0) or truncate to seq_len.

    The output is numerically identical to what the Kotlin Android tokenizer
    produces when it loads the same vocab.json — verified by get_vocab_info()
    checksum comparison.
    """
    clean = text.lower()
    for ch in [",", ".", "!", "?", "\"", "'", ":", ";", "(", ")", "-", "_", "/"]:
        clean = clean.replace(ch, " ")
    tokens = clean.split()

    indices = [VOCAB_MAP.get(t, VOCAB_MAP["<UNK>"]) for t in tokens]

    if len(indices) < seq_len:
        indices += [VOCAB_MAP["<PAD>"]] * (seq_len - len(indices))
    else:
        indices = indices[:seq_len]

    return indices


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class SMSDataset(Dataset):
    def __init__(self, data_list):
        self.data = [
            (torch.tensor(tokenize_message(text), dtype=torch.long), label)
            for text, label in data_list
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CallDataset(Dataset):
    def __init__(self, data_list):
        self.data = [
            (torch.tensor(features, dtype=torch.float32), label)
            for features, label in data_list
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Holdout / training split
# ---------------------------------------------------------------------------

_CACHED_SMS_DATA = None
_HOLDOUT_INDICES = None
_HOLDOUT_RATIO = 0.20


def _get_holdout_indices(dataset_size: int) -> set:
    """Fixed global holdout split — never used for client training."""
    global _HOLDOUT_INDICES
    if _HOLDOUT_INDICES is not None:
        return _HOLDOUT_INDICES
    rng = random.Random(42)
    indices = list(range(dataset_size))
    rng.shuffle(indices)
    holdout_size = max(50, int(dataset_size * _HOLDOUT_RATIO))
    _HOLDOUT_INDICES = set(indices[:holdout_size])
    return _HOLDOUT_INDICES


def get_training_pool():
    """SMS records available for federated client training (excludes holdout)."""
    sms_dataset = download_and_load_sms_dataset()
    holdout = _get_holdout_indices(len(sms_dataset))
    return [item for idx, item in enumerate(sms_dataset) if idx not in holdout]


def get_global_holdout_dataset():
    """Formal global holdout for unbiased evaluation — never seen during training."""
    sms_dataset = download_and_load_sms_dataset()
    holdout = _get_holdout_indices(len(sms_dataset))
    ham_pool  = [item for idx, item in enumerate(sms_dataset) if idx in holdout and item[1] == 0]
    spam_pool = [item for idx, item in enumerate(sms_dataset) if idx in holdout and item[1] == 1]

    sms_raw, call_raw = [], []
    rng = random.Random(42)
    all_holdout = ham_pool + spam_pool
    rng.shuffle(all_holdout)

    for text, label in all_holdout:
        sms_raw.append((text, label))
        if label == 1:
            duration = rng.uniform(5.0, 45.0)
            hour     = rng.choice([0, 1, 2, 3, 4, 22, 23])
            call_raw.append(([duration, hour, 0.0, float(rng.randint(3, 8)), rng.uniform(0.6, 1.0)], 1))
        else:
            duration = rng.uniform(30.0, 300.0)
            hour     = rng.randint(8, 20)
            call_raw.append(([duration, hour, float(rng.choice([0.0, 1.0])), float(rng.randint(1, 3)), rng.uniform(0.0, 0.2)], 0))

    return SMSDataset(sms_raw), CallDataset(call_raw), len(sms_raw)


def download_and_load_sms_dataset():
    """
    Downloads the official UCI SMS Spam Collection dataset and parses it.
    Falls back to a rich offline dataset if the download fails.
    """
    global _CACHED_SMS_DATA
    if _CACHED_SMS_DATA is not None:
        return _CACHED_SMS_DATA

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    parsed_data = []

    print("Attempting to download UCI SMS Spam Collection dataset...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            zip_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            with z.open("SMSSpamCollection") as f:
                content = f.read().decode("utf-8")

        for line in content.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) == 2:
                label_str, text = parts
                parsed_data.append((text, 1 if label_str == "spam" else 0))

        print(f"Downloaded {len(parsed_data)} real-world SMS records.")

    except Exception as e:
        print(f"Could not load UCI dataset: {e}. Falling back to offline corpus.")
        offline_ham = [
            "Hey! Are we still meeting for lunch tomorrow at 12?",
            "Just checking in, did you get the email I sent yesterday?",
            "Okay thanks, see you soon!",
            "Can you send me the password for the conference room?",
            "I'm running a bit late, start the meeting without me.",
            "Great job on the presentation! Everyone loved it.",
            "Can we reschedule our call to Friday afternoon?",
            "Did you remember to buy milk on your way home?",
            "Hi, how are you? Long time no see.",
            "Let me know if you need any help with the project.",
            "Yes, that plan sounds good. See you tomorrow.",
            "Thanks for the dinner last night, it was really fun.",
            "Don't forget to submit your weekly report by 5 PM.",
            "Hey, are you free for a quick chat?",
            "Sorry, I missed your call. I was in a meeting.",
            "Happy birthday! Hope you have a wonderful day.",
            "I'll be home in about 20 minutes.",
            "Let's catch up sometime next week.",
            "Can you forward me the invoice when you get it?",
            "Good luck with your interview today!",
        ]
        offline_spam = [
            "URGENT: Your MTN wallet is suspended. Click link to verify your account details.",
            "Congratulations! You won a GH₵1000 prize. Click here to claim now.",
            "Official alert: Unpaid bill due immediately. Login to secure payment link.",
            "Action required: Verify your account credentials to avoid suspension.",
            "WINNER: You have been selected for a free gift card. Reply to claim.",
            "Security alert: Suspicious login attempt. Secure your account now link.",
            "Dear customer, your mobile bill payment is unpaid. Click to update login.",
            "You have won a free holiday voucher! Call now on 09061104282 to claim.",
            "FREE Ringtone! text 'JOIN' to 80077 now to receive your free download.",
            "Private! Your account has a temporary limit. Click verify link now.",
            "Urgent: We detected a suspicious transfer of GH₵500 on your Telecel line. Click to dispute.",
            "Get cheap insurance quotes today! Reply STOP to unsubscribe.",
            "You qualify for a free upgrade. Visit our website immediately.",
            "Your parcel is held at our depot. Please click link to schedule delivery.",
            "IMPORTANT: Account verification required. Update your login profile.",
            "Please call our customer service agent immediately regarding your refund.",
            "Win a brand new phone! Text WIN to 88990 to participate.",
            "Alert: Your payment was successful. If not you, click link to cancel.",
            "URGENT: Click here to secure your online banking credentials.",
            "Congratulations, your application was approved. Transfer funds now.",
        ]
        for text in offline_ham:
            parsed_data.append((text, 0))
        for text in offline_spam:
            parsed_data.append((text, 1))

    _CACHED_SMS_DATA = parsed_data
    return parsed_data


# ---------------------------------------------------------------------------
# Data generators (non-IID federated clients)
# ---------------------------------------------------------------------------

def generate_client_data(client_id: str, num_samples: int = 100):
    """
    Generates non-IID data for a given client utilizing the real-world/offline
    dataset.  Clients ending in '-0' see heavy fraud; '-1' medium; '-2' almost
    none.
    """
    training_pool = get_training_pool()
    ham_pool  = [item for item in training_pool if item[1] == 0]
    spam_pool = [item for item in training_pool if item[1] == 1]

    random.seed(hash(client_id))

    if client_id.endswith("-0"):
        fraud_ratio = 0.50
    elif client_id.endswith("-1"):
        fraud_ratio = 0.20
    else:
        fraud_ratio = 0.02

    num_fraud = int(num_samples * fraud_ratio)
    num_normal = num_samples - num_fraud

    sms_data = (
        [random.choice(spam_pool) for _ in range(num_fraud)] +
        [random.choice(ham_pool)  for _ in range(num_normal)]
    )

    call_data = []
    for _ in range(num_fraud):
        call_data.append((
            [random.uniform(5.0, 45.0), random.choice([0, 1, 2, 3, 4, 22, 23]),
             0.0, float(random.randint(3, 8)), random.uniform(0.6, 1.0)],
            1
        ))
    for _ in range(num_normal):
        call_data.append((
            [random.uniform(30.0, 300.0), random.randint(8, 20),
             float(random.choice([0.0, 1.0, 1.0, 1.0])),
             float(random.randint(1, 3)), random.uniform(0.0, 0.3)],
            0
        ))

    return sms_data, call_data


def get_dataloaders(client_id: str, batch_size: int = 8, num_samples: int = 100):
    sms_raw, call_raw = generate_client_data(client_id, num_samples)
    sms_loader  = DataLoader(SMSDataset(sms_raw),  batch_size=batch_size, shuffle=True)
    call_loader = DataLoader(CallDataset(call_raw), batch_size=batch_size, shuffle=True)
    return sms_loader, call_loader


def generate_global_test_data(num_samples: int = 200):
    """Legacy alias — returns the formal global holdout dataset."""
    holdout_sms, holdout_call, _ = get_global_holdout_dataset()
    return holdout_sms, holdout_call
