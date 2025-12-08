from enum import Enum

from pydantic import BaseModel

from reviewing_agents.shared.mini_search_agent import ResearchAgent


class Reference(BaseModel):
    title: str
    authors: str
    journal: str


class References(BaseModel):
    references: list[Reference]


class VerificationStatus(Enum):
    VERIFIED = "verified"
    SUSPECT = "suspect"
    ERROR = "error"


def parse_verification_status(text: str) -> VerificationStatus:
    if "Error checking reference" in text:
        return VerificationStatus.ERROR
    if "VERIFIED" in text:
        return VerificationStatus.VERIFIED
    if "SUSPECT" in text:
        return VerificationStatus.SUSPECT
    return VerificationStatus.ERROR


def run_verification_worker(data: dict) -> dict:
    research_agent = ResearchAgent()
    result = research_agent.run_query(data["query"])
    return {"reference": data["reference"], "verification_result": result}
