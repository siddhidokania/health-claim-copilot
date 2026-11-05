from app.agent import format_hits
from app.claims import ClaimReview, review_prompt
from app.privacy import redact


def test_redact_contact_details():
    text = "Call +91 98765 43210 or mail ravi.k@example.com. PAN ABCDE1234F, Aadhaar 1234 5678 9012."
    out = redact(text)
    for private in ("98765", "ravi.k", "ABCDE1234F", "1234 5678"):
        assert private not in out
    assert "[PHONE]" in out and "[EMAIL]" in out and "[PAN]" in out and "[AADHAAR]" in out


def test_redact_labelled_numbers_and_names():
    out = redact("Patient Name: Ravi Kumar\nPolicy No: 123/456/789 and Claim Number: CL-22")
    assert "Ravi" not in out and "123/456" not in out and "CL-22" not in out
    assert out.startswith("Patient Name: [NAME]")


def test_redact_keeps_policy_terms():
    text = "A waiting period of 24 months applies. Room rent is limited to 1% of sum insured, up to Rs 5,000 per day."
    assert redact(text) == text


def test_document_text_is_redacted_before_it_reaches_the_model():
    hits = [{"chunk_id": 7, "document_id": 1, "title": "Policy", "page": 1, "heading": None,
             "content": "Insured mobile 9876543210", "source_url": None}]
    assert "9876543210" not in format_hits(hits)


def test_review_prompt_handles_missing_details():
    prompt = review_prompt(ClaimReview(document_ids=[1], treatment="Appendix surgery"))
    assert "Appendix surgery" in prompt
    assert "has not been given a reason" in prompt
    assert "## Draft grievance letter" in prompt
