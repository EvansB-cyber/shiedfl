import torch
from torch.utils.data import Dataset, DataLoader
import random
import urllib.request
import zipfile
import io
import os

# Vocabulary definition for SMS messages
VOCAB = [
    "<PAD>", "<UNK>", "hello", "hi", "how", "are", "you", "meeting", "tomorrow", 
    "lunch", "dinner", "ok", "thanks", "sender", "receiver", "amount", "transfer", 
    "bank", "verify", "suspend", "alert", "urgent", "link", "click", "claim", 
    "prize", "winner", "cash", "account", "secure", "credentials", "login", 
    "password", "service", "payment", "due", "unpaid", "bill", "official", 
    "update", "temporary", "limit", "access", "gift", "card", "congratulations", 
    "immediate", "action", "required", "call", "now", "free", "txt", "text", 
    "stop", "mobile", "claim", "customer", "contact", "reply", "urgent", "msg",
    "please", "won", "service", "latest", "important"
]
VOCAB_MAP = {word: idx for idx, word in enumerate(VOCAB)}
VOCAB_SIZE = 1000  # We set a slightly larger vocab size to allow for out-of-vocab indexing

def tokenize_message(text, seq_len=20):
    """
    Cleans, tokenizes, and pads/truncates a message text into integer sequence.
    """
    # Quick cleaning
    clean_text = text.lower()
    for char in [",", ".", "!", "?", "\"", "'", ":", ";", "(", ")", "-", "_", "/"]:
        clean_text = clean_text.replace(char, " ")
    tokens = clean_text.split()
    
    indices = []
    for t in tokens:
        idx = VOCAB_MAP.get(t, 1) # 1 is <UNK>
        if idx < VOCAB_SIZE:
            indices.append(idx)
        else:
            indices.append(1)
    
    # Pad or truncate
    if len(indices) < seq_len:
        indices += [0] * (seq_len - len(indices)) # 0 is <PAD>
    else:
        indices = indices[:seq_len]
    return indices

class SMSDataset(Dataset):
    def __init__(self, data_list):
        # Each item in data_list: (message_text, label)
        self.data = []
        for text, label in data_list:
            seq = tokenize_message(text)
            self.data.append((torch.tensor(seq, dtype=torch.long), label))
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

class CallDataset(Dataset):
    def __init__(self, data_list):
        # Each item in data_list: (features_list, label)
        self.data = []
        for features, label in data_list:
            self.data.append((torch.tensor(features, dtype=torch.float32), label))
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

# Cache for real-world SMS data to avoid redownloading
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
    ham_pool = [item for idx, item in enumerate(sms_dataset) if idx in holdout and item[1] == 0]
    spam_pool = [item for idx, item in enumerate(sms_dataset) if idx in holdout and item[1] == 1]

    sms_raw = []
    call_raw = []
    rng = random.Random(42)
    all_holdout = ham_pool + spam_pool
    rng.shuffle(all_holdout)

    for text, label in all_holdout:
        sms_raw.append((text, label))
        if label == 1:
            duration = rng.uniform(5.0, 45.0)
            hour = rng.choice([0, 1, 2, 3, 4, 22, 23])
            call_raw.append(([duration, hour, 0.0, float(rng.randint(3, 8)), rng.uniform(0.6, 1.0)], 1))
        else:
            duration = rng.uniform(30.0, 300.0)
            hour = rng.randint(8, 20)
            call_raw.append(([duration, hour, float(rng.choice([0.0, 1.0])), float(rng.randint(1, 3)), rng.uniform(0.0, 0.2)], 0))

    return SMSDataset(sms_raw), CallDataset(call_raw), len(sms_raw)

def download_and_load_sms_dataset():
    """
    Downloads the official UCI SMS Spam Collection dataset and parses it.
    If offline or download fails, falls back to a rich offline dataset.
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
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            zip_file_bytes = response.read()
            
        with zipfile.ZipFile(io.BytesIO(zip_file_bytes)) as z:
            # The file inside the zip is named 'SMSSpamCollection'
            with z.open('SMSSpamCollection') as f:
                content = f.read().decode('utf-8')
                
        for line in content.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) == 2:
                label_str, text = parts
                label = 1 if label_str == 'spam' else 0
                parsed_data.append((text, label))
                
        print(f"Successfully downloaded and loaded {len(parsed_data)} real-world SMS records.")
        
    except Exception as e:
        print(f"Could not load UCI SMS Dataset: {e}. Falling back to rich offline dataset.")
        # Rich offline fallback corpus
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
            "Good luck with your interview today!"
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
            "Congratulations, your application was approved. Transfer funds now."
        ]
        
        for text in offline_ham:
            parsed_data.append((text, 0))
        for text in offline_spam:
            parsed_data.append((text, 1))
            
    _CACHED_SMS_DATA = parsed_data
    return parsed_data

# Synthetic Data Generators for each Edge client
def generate_client_data(client_id, num_samples=100):
    """
    Generates non-IID data for a given client utilizing the real-world/offline dataset.
    Clients with '0' (e.g. S1-0) are exposed to a high percentage of fraud.
    Clients with '1' (e.g. S1-1) are exposed to a medium percentage.
    Clients with '2' (e.g. S1-2) are exposed to almost no fraud.
    """
    # Training pool excludes global holdout
    training_pool = get_training_pool()
    ham_pool = [item for item in training_pool if item[1] == 0]
    spam_pool = [item for item in training_pool if item[1] == 1]
    
    # Seeding for reproducibility per client
    random.seed(hash(client_id))
    
    # Determine fraud ratio
    if client_id.endswith("-0"):
        fraud_ratio = 0.50
    elif client_id.endswith("-1"):
        fraud_ratio = 0.20
    else:
        fraud_ratio = 0.02

    # Generate SMS data
    sms_data = []
    num_fraud_sms = int(num_samples * fraud_ratio)
    num_normal_sms = num_samples - num_fraud_sms
    
    for _ in range(num_fraud_sms):
        text, label = random.choice(spam_pool)
        sms_data.append((text, label))
    for _ in range(num_normal_sms):
        text, label = random.choice(ham_pool)
        sms_data.append((text, label))
        
    # Generate Call data (keeps identical numerical profiles)
    call_data = []
    for _ in range(num_fraud_sms):
        duration = random.uniform(5.0, 45.0)
        hour = random.choice([0, 1, 2, 3, 4, 22, 23])
        is_contact_saved = 0.0
        times_called = float(random.randint(3, 8))
        source_risk = random.uniform(0.6, 1.0)
        call_data.append(([duration, hour, is_contact_saved, times_called, source_risk], 1))
        
    for _ in range(num_normal_sms):
        duration = random.uniform(30.0, 300.0)
        hour = random.randint(8, 20)
        is_contact_saved = float(random.choice([0.0, 1.0, 1.0, 1.0]))
        times_called = float(random.randint(1, 3))
        source_risk = random.uniform(0.0, 0.3)
        call_data.append(([duration, hour, is_contact_saved, times_called, source_risk], 0))
        
    return sms_data, call_data

def get_dataloaders(client_id, batch_size=8, num_samples=100):
    sms_raw, call_raw = generate_client_data(client_id, num_samples)
    
    sms_dataset = SMSDataset(sms_raw)
    call_dataset = CallDataset(call_raw)
    
    sms_loader = DataLoader(sms_dataset, batch_size=batch_size, shuffle=True)
    call_loader = DataLoader(call_dataset, batch_size=batch_size, shuffle=True)
    
    return sms_loader, call_loader

def generate_global_test_data(num_samples=200):
    """Legacy alias — returns the formal global holdout dataset."""
    holdout_sms, holdout_call, size = get_global_holdout_dataset()
    return holdout_sms, holdout_call
