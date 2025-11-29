"""
Simple Semantic Verifier using NLI Model Only
"""

import os
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
    Simple semantic verifier using only NLI model
    """

    def __init__(self, model_path: str = None):
        self.model = None

        # Default: look for a local model inside this project's ml/model directory
        if model_path is not None:
            self.model_path = model_path
        else:
            # semantics.py is in ml/, so ml/model is a sibling folder
            project_root = os.path.dirname(os.path.abspath(__file__))  # .../ml
            self.model_path = os.path.join(
                project_root, "model")      # .../ml/model

        self._load_nli_model()

    def _load_nli_model(self):
        """Safely load the NLI model with fallback and graceful failure."""
        try:
            # Import transformers lazily so failures (torch DLL, etc.) are caught here
            from transformers import pipeline
        except Exception as e:
            logger.error(f"❌ transformers library not available: {e}")
            logger.warning(
                "⚠️ NLI system disabled — proceeding without semantic verifier.")
            self.model = None
            return

        # 1) Try local model
        try:
            logger.info(f"🔍 Looking for local NLI model at: {self.model_path}")

            if os.path.exists(self.model_path):
                logger.info(
                    "Local model directory found. Attempting to load...")
                self.model = pipeline(
                    "text-classification",
                    model=self.model_path,
                    tokenizer=self.model_path,
                    max_length=512,
                    truncation=True,
                )
                logger.info(
                    "✅ NLI model loaded successfully from local directory")
                return
            else:
                logger.info(
                    "No local NLI model directory found; will use fallback model.")

        except Exception as e:
            logger.error(f"⚠️ Local NLI model found but failed to load: {e}")

        # 2) Fallback – download a standard NLI model
        try:
            logger.info(
                "🌐 Downloading fallback NLI model: facebook/bart-large-mnli ...")
            self.model = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                max_length=512,
                truncation=True,
            )
            logger.info("✅ Fallback NLI model loaded successfully")
        except Exception as e:
            logger.error(
                f"❌ Failed to initialize any NLI model (fallback also failed): {e}")
            logger.warning(
                "⚠️ NLI system disabled — pattern analysis will be used only.")
            self.model = None

    def verify_claim(self, claim: str, content: str) -> VerificationResult:
        """
        Verify claim against content using NLI model only
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

            # For HF pipelines, single input returns a dict, list returns list[dict]
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            nli_label = result["label"]
            nli_confidence = float(result["score"])

            # Map NLI labels to our relation types
            if "ENTAIL" in nli_label.upper():
                relation = ClaimRelation.SUPPORT
                reasoning = [
                    f"NLI determined entailment with {nli_confidence:.3f} confidence"]
                support_score = nli_confidence
                contradict_score = 0.0
                irrelevant_score = 0.0
            elif "CONTRAD" in nli_label.upper():
                relation = ClaimRelation.CONTRADICT
                reasoning = [
                    f"NLI determined contradiction with {nli_confidence:.3f} confidence"]
                support_score = 0.0
                contradict_score = nli_confidence
                irrelevant_score = 0.0
            else:  # NEUTRAL
                relation = ClaimRelation.IRRELEVANT
                reasoning = [
                    f"NLI determined neutral relation with {nli_confidence:.3f} confidence"]
                support_score = 0.0
                contradict_score = 0.0
                irrelevant_score = nli_confidence

            processing_time = time.time() - start_time

            logger.info(
                f"Result: {relation.value} (confidence: {nli_confidence:.3f})")

            return VerificationResult(
                relation=relation,
                confidence=nli_confidence,
                processing_time=processing_time,
                evidence=[],  # Empty for simple version
                reasoning=reasoning,
                support_score=support_score,
                contradict_score=contradict_score,
                irrelevant_score=irrelevant_score,
                semantic_scores={},  # Empty for simple version
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


# Backward compatibility
IntelligentSemanticVerifier = SimpleNLIVerifier


if __name__ == "__main__":
    verifier = SimpleNLIVerifier()

    # Test cases
    test_cases = [
        (
            "COVID-19 vaccines cause infertility",
            "Scientific studies show no link between COVID-19 vaccines and infertility",
        ),
        ("Vaccines are completely safe",
         "Some vaccines have rare side effects that need monitoring"),
        ("Exercise improves heart health",
         "Studies show that regular exercise improves cardiovascular health"),
    ]

    print("🧪 Testing Simple NLI Verifier")
    print("=" * 50)

    for i, (claim, content) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Claim: {claim}")

        result = verifier.verify_claim(claim, content)

        print(f"Result: {result.relation.value.upper()}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"NLI Label: {result.nli_label}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Processing Time: {result.processing_time:.3f}s")

    print(f"\n{'=' * 50}")
    print("✅ Simple NLI Verifier Test Complete")
