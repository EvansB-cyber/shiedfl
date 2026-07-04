import io
import torch
from cryptography.fernet import Fernet

# A fixed symmetric key for the simulation. 
# In production, this would be securely distributed via Diffie-Hellman or a KMS.
SECRET_KEY = b'xL53r71jJ_-8tA0uHlq2eZp6SjFk0C2zT8vH4hXG1Lg='
fernet = Fernet(SECRET_KEY)

def encrypt_weights(weights_dict):
    """
    Serializes a PyTorch state_dict to bytes and encrypts it.
    Returns:
        bytes: Encrypted byte string representing the state dict.
    """
    buffer = io.BytesIO()
    # Use torch.save to serialize
    torch.save(weights_dict, buffer)
    buffer.seek(0)
    raw_bytes = buffer.read()
    
    encrypted_bytes = fernet.encrypt(raw_bytes)
    return encrypted_bytes

def decrypt_weights(encrypted_bytes):
    """
    Decrypts the byte string and deserializes back into a PyTorch state_dict.
    Returns:
        dict: The decrypted PyTorch state_dict.
    """
    raw_bytes = fernet.decrypt(encrypted_bytes)
    buffer = io.BytesIO(raw_bytes)
    # Deserialize back to tensors
    weights_dict = torch.load(buffer, map_location=torch.device('cpu'), weights_only=False)
    return weights_dict
