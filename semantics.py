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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        # Set the path to your local model
        #self.model_path = model_path or r"F:\project factGuru\codes\web_scrapper\backend\ml\model"
        self.model_path = "C:\\Users\\hiran\\OneDrive\\Desktop\\proj FactGuru\\web_scrapper\\backend\\ml\\model"
        self._load_nli_model()
    
    def _load_nli_model(self):
        """Load the NLI model from local directory"""
        try:
            from transformers import pipeline
            
            logger.info(f"Loading local NLI model from: {self.model_path}")
            
            # Check if the model directory exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model path does not exist: {self.model_path}")
                self.model = None
                return
                
            self.model = pipeline(
                "text-classification",
                model=self.model_path,
                max_length=512,
                truncation=True
            )
            logger.info("✅ NLI model loaded successfully from local directory!")
            
        except Exception as e:
            logger.error(f"Failed to load NLI model from {self.model_path}: {e}")
            logger.info("Trying to download model as fallback...")
            try:
                self.model = pipeline(
                    "text-classification",
                    model="roberta-large-mnli",
                    max_length=512,
                    truncation=True
                )
                logger.info("✅ NLI model downloaded as fallback")
            except Exception as e2:
                logger.error(f"Failed to download fallback model: {e2}")
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
                return VerificationResult(ClaimRelation.ERROR, 0.0, 0.0, reasoning=["NLI model not available"])
            
            logger.info(f"Analyzing claim: '{claim}'")
            
            # Use NLI model to analyze relationship
            result = self.model({
                "text": content[:1000],  # premise (truncated)
                "text_pair": claim  # hypothesis
            })
            
            # Process NLI result
            nli_label = result['label']
            nli_confidence = result['score']
            
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
                evidence=[],  # Empty for simple version
                reasoning=reasoning,
                support_score=support_score,
                contradict_score=contradict_score,
                irrelevant_score=irrelevant_score,
                semantic_scores={},  # Empty for simple version
                nli_result=result,
                nli_label=nli_label
            )
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            processing_time = time.time() - start_time
            return VerificationResult(ClaimRelation.ERROR, 0.0, processing_time, reasoning=[str(e)])

# Backward compatibility
IntelligentSemanticVerifier = SimpleNLIVerifier

# Example usage
if __name__ == "__main__":
    verifier = SimpleNLIVerifier()
    
    # Test cases
    test_cases = [
        ("COVID-19 vaccines cause infertility", "Scientific studies show no link between COVID-19 vaccines and infertility"),
        ("Vaccines are completely safe", "Some vaccines have rare side effects that need monitoring"),
        ("Exercise improves heart health", "Studies show that regular exercise improves cardiovascular health"),
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

    print(f"\n{'='*50}")
    print("✅ Simple NLI Verifier Test Complete")
