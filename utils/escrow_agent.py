"""
Automated Escrow Agent: heuristic + optional LLM reasoning for auto-approvals.

Runs after risk assessment to auto-release low-risk transfers or auto-block
clear fraud, reducing manual escrow queue load.
"""
import os
import json
import urllib.request


PHISHING_KEYWORDS = [
    "urgent", "verify", "suspend", "click", "link", "claim", "prize",
    "winner", "login", "credentials", "official alert", "congratulations",
    "action required", "secure your account", "unpaid bill"
]


def heuristic_decision(risk_report: dict, amount: float, message: str = "") -> dict:
    """
    Rule-based auto-decision engine.
    Returns: {"action": "AUTO_APPROVE"|"AUTO_BLOCK"|"MANUAL_REVIEW", "confidence": float, "reason": str}
    """
    total = risk_report.get("total_risk_score", 0)
    sms = risk_report.get("sms_risk_score", 0)
    contact = risk_report.get("contact_risk_score", 0)
    msg_lower = (message or "").lower()
    keyword_hits = sum(1 for kw in PHISHING_KEYWORDS if kw in msg_lower)

    if total < 0.25 and sms < 0.3 and contact < 0.2 and amount < 200:
        return {
            "action": "AUTO_APPROVE",
            "confidence": 0.92,
            "reason": "Low aggregate risk; trusted contact pattern and small amount.",
            "agent": "heuristic"
        }

    if total >= 0.80 or sms >= 0.85 or (keyword_hits >= 3 and amount > 100):
        return {
            "action": "AUTO_BLOCK",
            "confidence": 0.88,
            "reason": f"High fraud indicators: risk={total:.2f}, phishing_keywords={keyword_hits}.",
            "agent": "heuristic"
        }

    if contact >= 0.90 and amount > 500:
        return {
            "action": "AUTO_BLOCK",
            "confidence": 0.85,
            "reason": "Flagged untrusted contact with large transfer amount.",
            "agent": "heuristic"
        }

    return {
        "action": "MANUAL_REVIEW",
        "confidence": 0.55,
        "reason": "Borderline risk score; escalated to provider escrow queue.",
        "agent": "heuristic"
    }


def llm_decision(risk_report: dict, amount: float, message: str = "") -> dict:
    """
    Optional LLM-backed decision when OPENAI_API_KEY is set.
    Falls back to heuristic if unavailable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        result = heuristic_decision(risk_report, amount, message)
        result["agent"] = "heuristic (LLM unavailable)"
        return result

    prompt = (
        "You are a mobile-money fraud escrow agent for Ghana. "
        "Given the risk report, respond ONLY with JSON: "
        '{"action":"AUTO_APPROVE"|"AUTO_BLOCK"|"MANUAL_REVIEW","confidence":0-1,"reason":"..."}\n'
        f"Risk: {json.dumps(risk_report)}\nAmount GHS: {amount}\nMessage: {message[:300]}"
    )
    try:
        payload = json.dumps({
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = json.loads(resp.read().decode())
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        parsed["agent"] = "llm"
        return parsed
    except Exception as exc:
        result = heuristic_decision(risk_report, amount, message)
        result["reason"] = f"Heuristic fallback ({exc})"
        result["agent"] = "heuristic"
        return result


def auto_resolve_escrow(risk_report: dict, amount: float, message: str = "",
                        use_llm: bool = False) -> dict:
    """Main entry: pick LLM or heuristic agent."""
    if use_llm or os.environ.get("ESCROW_USE_LLM", "").lower() in ("1", "true", "yes"):
        return llm_decision(risk_report, amount, message)
    return heuristic_decision(risk_report, amount, message)
