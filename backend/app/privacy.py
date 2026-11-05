"""Remove personal identifiers from text before it leaves this computer for the AI model."""

import re

PATTERNS = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("[AADHAAR]", re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b")),
    ("[PAN]", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("[PHONE]", re.compile(r"(?<![\w.])(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)")),
    # Labelled reference numbers. The label stays so the sentence still makes sense.
    ("[NUMBER]", re.compile(r"(?<=\bpolicy no\. )\S+|(?<=\bpolicy no: )\S+|(?<=\bpolicy number: )\S+|(?<=\bpolicy number )\S+|"
                            r"(?<=\bclaim no\. )\S+|(?<=\bclaim no: )\S+|(?<=\bclaim number: )\S+|(?<=\bclaim number )\S+|"
                            r"(?<=\bmember id: )\S+|(?<=\buhid: )\S+", re.I)),
]
NAME_LINE = re.compile(r"^(\s*(?:name of (?:the )?(?:insured|patient|proposer|policyholder)|insured name|patient name|"
                       r"policyholder|proposer)\s*[:-]\s*).+$", re.I | re.M)


def redact(text: str) -> str:
    text = NAME_LINE.sub(r"\1[NAME]", text)
    for label, pattern in PATTERNS:
        text = pattern.sub(label, text)
    return text
