"""MITRE_DESCRIPTIONS = {
    "Execution": "Execution techniques.",
    "Persistence": "Persistence techniques.",
    "Credential Access": "Credential Access techniques.",
    "Collection": "Collection techniques.",
    "Defense Evasion": "Defense Evasion techniques.",
    "Exploitation": "Exploitation techniques.",
    "Privilege Escalation": "Privilege Escalation techniques.",
    "Exfiltration": "Exfiltration techniques.",
    "Command and Control": "Command and Control techniques.",
    "Discovery": "Discovery techniques.",
    "Initial Access": "Initial Access techniques",
    "Lateral Movement": "Lateral Movement techniques.",
    "Impact": "Impact techniques.",
    "Reconnaissance": "Reconaissance techniques",
    "Delivery": "Delivery techniques",
    "Resource Development": "Resource Development techniques."
}

"""

MITRE_DESCRIPTIONS = {
    "TA0002": {"name": "Execution", "description": "Execution consists of techniques that result in adversary-controlled code running on a local or remote system."},
    "TA0003": {"name": "Persistence", "description": "Persistence consists of techniques that adversaries use to maintain access to systems across restarts, changed credentials, and other interruptions that could cut off their access."},
    "TA0006": {"name": "Credential Access", "description": "Credential Access consists of techniques for stealing credentials like account names and passwords."},
    "TA0009": {"name": "Collection", "description": "Collection consists of techniques used to gather information relevant to the adversary's goal."},
    "TA0005": {"name": "Defense Evasion", "description": "Defense Evasion consists of techniques an adversary may use to evade detection or avoid other defenses."},
    "TA9998": {"name": "Exploitation", "description": "Exploitation consists of techniques that involve exploiting a vulnerability in software, a computer system, or an associated device."},  # Custom ID for Exploitation
    "TA0004": {"name": "Privilege Escalation", "description": "Privilege Escalation consists of techniques that adversaries use to gain higher-level permissions on a system or network."},
    "TA0010": {"name": "Exfiltration", "description": "Exfiltration consists of techniques that adversaries may use to steal data from your network."},
    "TA0011": {"name": "Command and Control", "description": "Command and Control consists of techniques that adversaries may use to communicate with systems under their control within a victim network."},
    "TA0007": {"name": "Discovery", "description": "Discovery consists of techniques that allow the adversary to gain knowledge about the system and internal network."},
    "TA0001": {"name": "Initial Access", "description": "Initial Access consists of techniques that adversaries use to gain an initial foothold within a network."},
    "TA0008": {"name": "Lateral Movement", "description": "Lateral Movement consists of techniques that adversaries use to move through your environment."},
    "TA0040": {"name": "Impact", "description": "Impact consists of techniques that adversaries use to disrupt availability or compromise integrity by manipulating business and operational processes."},
    "TA0043": {"name": "Reconnaissance", "description": "The adversary is trying to gather information they can use to plan future operations."},
    "TA9999": {"name": "Delivery", "description": "Delivery consists of techniques that involve the adversary transmitting a malicious file to the victim."},  # Custom ID for Delivery
    "TA0042": {"name": "Resource Development", "description": "Resource Development consists of techniques that involve adversaries creating, purchasing, or compromising resources that can be used to support targeting."}
}
