from fastapi import FastAPI

def apply_security(app: FastAPI):
    """
    Security middlewares and configurations can be applied here.
    Authentication has been temporarily removed for this deployment.
    """
    pass

def require_admin(user_record: dict):
    """
    Validates if the provided user record has admin privileges.
    Since authentication is removed, this acts as a bypass.
    """
    pass
