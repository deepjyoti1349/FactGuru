"""
Simple Semantic Verifier - Online Model Only
"""

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ClaimRelation(Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    IRRELEVANT = "irrelevant"
    UNCLEAR = "unclear"
    ERROR = "error"


@dataclass
class VerificationResult:
    relation: ClaimRelation
    confidence: float
    processing_time: float
    evidence: List[str] = None
    reasoning: List[str] = None
    support_score: float = 0.0
    contradict_score: float = 0.0
    irrelevant_score: float = 0.0
    semantic_scores: Dict[str, float] = None
    nli_result: Dict = None
    nli_label: str = ""

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.reasoning is None:
            self.reasoning = []
        if self.semantic_scores is None:
            self.semantic_scores = {}


class SimpleNLIVerifier:
    """
    Simple semantic verifier using only online NLI model
    """

    def __init__(self):
        self.model = None
        self._load_nli_model()

    def _load_nli_model(self):
        """Load the NLI model - online only"""
        try:
            from transformers import pipeline
            
            logger.info("🌐 Loading online NLI model: facebook/bart-large-mnli...")
            self.model = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                max_length=512,
                truncation=True,
            )
            logger.info("✅ Online NLI model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load NLI model: {e}")
            logger.warning("⚠️ NLI system disabled")
            self.model = None

    def verify_claim(self, claim: str, content: str) -> VerificationResult:
        """
        Verify claim against content using NLI model
        """
        start_time = time.time()

        try:
            if not claim or not content:
                return VerificationResult(ClaimRelation.ERROR, 0.0, 0.0)

            if not self.model:
                return VerificationResult(
                    ClaimRelation.ERROR,
                    0.0,
                    0.0,
                    reasoning=["NLI model not available"],
                )

            logger.info(f"Analyzing claim: '{claim}'")

            # Use NLI model to analyze relationship
            result = self.model(
                {
                    "text": content[:1000],  # premise (truncated)
                    "text_pair": claim,      # hypothesis
                }
            )

            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            nli_label = result["label"]
            nli_confidence = float(result["score"])

            # Map NLI labels to our relation types
            if "ENTAIL" in nli_label.upper():
                relation = ClaimRelation.SUPPORT
                reasoning = [f"NLI determined entailment with {nli_confidence:.3f} confidence"]
            elif "CONTRAD" in nli_label.upper():
                relation = ClaimRelation.CONTRADICT
                reasoning = [f"NLI determined contradiction with {nli_confidence:.3f} confidence"]
            else:
                relation = ClaimRelation.IRRELEVANT
                reasoning = [f"NLI determined neutral relation with {nli_confidence:.3f} confidence"]

            processing_time = time.time() - start_time

            logger.info(f"Result: {relation.value} (confidence: {nli_confidence:.3f})")

            return VerificationResult(
                relation=relation,
                confidence=nli_confidence,
                processing_time=processing_time,
                evidence=[],
                reasoning=reasoning,
                nli_result=result,
                nli_label=nli_label,
            )

        except Exception as e:
            logger.error(f"Verification error: {e}")
            processing_time = time.time() - start_time
            return VerificationResult(
                ClaimRelation.ERROR,
                0.0,
                processing_time,
                reasoning=[str(e)],
            )


if __name__ == "__main__":
    verifier = SimpleNLIVerifier()
    print("✅ Semantics verifier ready")
