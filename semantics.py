"""
Simple Semantic Verifier with Hugging Face Download
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
    Semantic verifier that downloads NLI models from Hugging Face
    """

    def __init__(self):
        self.model = None
        self.model_path = "ml/model"  # Correct relative path
        self._load_nli_model()

    def _download_nli_model(self):
        """Download NLI model from Hugging Face if not exists locally"""
        try:
            from huggingface_hub import snapshot_download
            
            # Your Hugging Face repository
            repo_id = "rockOn08/factguru-models"
            
            # Create directory if it doesn't exist
            os.makedirs(self.model_path, exist_ok=True)
            
            # Check if models already exist locally
            if not os.path.exists(self.model_path) or len(os.listdir(self.model_path)) == 0:
                logger.info("📥 Downloading NLI model from Hugging Face...")
                
                # Download model files from Hugging Face
                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=self.model_path,
                    local_dir=self.model_path
                )
                logger.info(f"✅ NLI model downloaded to: {downloaded_path}")
                return downloaded_path
            else:
                logger.info("✅ Using existing NLI model from ml/model/")
                return self.model_path
                
        except Exception as e:
            logger.error(f"⚠️ NLI model download failed: {e}")
            return None

    def _load_nli_model(self):
        """Load the NLI model - try Hugging Face first, then online fallback"""
        try:
            from transformers import pipeline
            
            # First, download models from Hugging Face
            model_path = self._download_nli_model()
            
            if model_path and os.path.exists(model_path):
                logger.info(f"🔍 Loading NLI model from: {model_path}")
                self.model = pipeline(
                    "text-classification",
                    model=model_path,
                    tokenizer=model_path,
                    max_length=512,
                    truncation=True,
                )
                logger.info("✅ NLI model loaded successfully from Hugging Face")
                return
            
        except Exception as e:
            logger.error(f"⚠️ Hugging Face model loading failed: {e}")

        # Fallback to online model
        try:
            logger.info("🌐 Using online NLI model as fallback...")
            self.model = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                max_length=512,
                truncation=True,
            )
            logger.info("✅ Online NLI model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load any NLI model: {e}")
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

            # For HF pipelines, single input returns a dict, list returns list[dict]
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            nli_label = result["label"]
            nli_confidence = float(result["score"])

            # Map NLI labels to our relation types
            if "ENTAIL" in nli_label.upper():
                relation = ClaimRelation.SUPPORT
                reasoning = [f"NLI determined entailment with {nli_confidence:.3f} confidence"]
                support_score = nli_confidence
                contradict_score = 0.0
                irrelevant_score = 0.0
            elif "CONTRAD" in nli_label.upper():
                relation = ClaimRelation.CONTRADICT
                reasoning = [f"NLI determined contradiction with {nli_confidence:.3f} confidence"]
                support_score = 0.0
                contradict_score = nli_confidence
                irrelevant_score = 0.0
            else:  # NEUTRAL
                relation = ClaimRelation.IRRELEVANT
                reasoning = [f"NLI determined neutral relation with {nli_confidence:.3f} confidence"]
                support_score = 0.0
                contradict_score = 0.0
                irrelevant_score = nli_confidence

            processing_time = time.time() - start_time

            logger.info(f"Result: {relation.value} (confidence: {nli_confidence:.3f})")

            return VerificationResult(
                relation=relation,
                confidence=nli_confidence,
                processing_time=processing_time,
                evidence=[],
                reasoning=reasoning,
                support_score=support_score,
                contradict_score=contradict_score,
                irrelevant_score=irrelevant_score,
                semantic_scores={},
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
    ]

    print("🧪 Testing NLI Verifier with Hugging Face")
    print("=" * 50)

    for i, (claim, content) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Claim: {claim}")

        result = verifier.verify_claim(claim, content)

        print(f"Result: {result.relation.value.upper()}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"NLI Label: {result.nli_label}")
        print(f"Processing Time: {result.processing_time:.3f}s")

    print(f"\n{'=' * 50}")
    print("✅ NLI Verifier Test Complete")
