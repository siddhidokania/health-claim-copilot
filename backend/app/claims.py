"""Turn the details of a refused or disputed claim into a review request for the assistant."""

from pydantic import BaseModel, Field


TOPICS = ["waiting period", "pre-existing disease", "exclusions", "room rent limit and sub-limits",
          "time limit to inform the insurer", "documents required and time limit to submit a claim",
          "grievance procedure"]


class ClaimReview(BaseModel):
    document_ids: list[int] = Field(min_length=1, max_length=5)
    claim_type: str = Field(default="reimbursement", pattern="^(reimbursement|cashless)$")
    treatment: str = Field(min_length=3, max_length=1000)
    amount: str = Field(default="", max_length=100)
    policy_start: str = Field(default="", max_length=100)
    treatment_date: str = Field(default="", max_length=100)
    reason_given: str = Field(default="", max_length=6000)
    documents_sent: str = Field(default="", max_length=2000)


def topics(c: ClaimReview) -> list[str]:
    """What to look up in the policy for this claim, all in one go."""
    wanted = [c.treatment, *TOPICS]
    if c.reason_given:
        wanted.insert(1, c.reason_given[:300])
    return wanted


def review_prompt(c: ClaimReview) -> str:
    reason = c.reason_given or "The person has not been given a reason, or has not entered one."
    return f"""Review this health insurance claim against the policy excerpts.

Claim type: {c.claim_type}
Treatment: {c.treatment}
Amount claimed: {c.amount or "not given"}
Policy start date: {c.policy_start or "not given"}
Treatment date: {c.treatment_date or "not given"}
Reason given by the insurer: {reason}
Documents already sent: {c.documents_sent or "not given"}

Write these sections with these exact headings:

## In simple words
Two or three sentences on what seems to have happened.

## What the policy says
The clauses that bear on this claim, each with its citation.

## Possible problems
Anything the person could check, such as a clause that may not apply, a missing document or a date that matters. \
Call these questions to check. Never say the insurer is wrong or that the claim will succeed.

## Documents to gather
A checklist, naming each document exactly as the policy does.

## Details still needed
What you could not judge from what you were given.

## Draft grievance letter
A short, polite letter to the insurer's grievance officer, in plain text, asking for the specific clause relied on and a \
review. Use [YOUR NAME], [POLICY NUMBER] and [CLAIM NUMBER] as placeholders. State only facts given above, and never say \
the insurer cited a clause number unless the reason above names it."""
