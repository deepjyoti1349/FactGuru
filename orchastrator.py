import sys
import os
import time
import math
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.dirname(__file__))

def log(msg: str):
    """Log messages with timestamp for debugging and monitoring"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{ts} - {msg}")

# Import core system components
try:
    from semantics import SimpleNLIVerifier
    from input_handler import InputHandler
    from search_connector import SearchConnector
    from credibility_extractor import CredibilityExtractor
    from scrapper_engine import get_scraper_engine
    log("✅ Intelligent semantic system ready")
except ImportError as e:
    log(f"❌ Import failed: {e}")
    raise

# Import Pattern Analysis module
try:
    pattern_analysis_path = os.path.join(os.path.dirname(__file__), 'ml', 'pattern_analysis')
    if pattern_analysis_path not in sys.path:
        sys.path.append(pattern_analysis_path)
    
    from pattern import ArticleAnalyzer
    log("✅ Pattern analysis system ready")
    PatternAnalyzerAvailable = True
except ImportError as e:
    log(f"⚠️ Pattern analysis not available: {e}")
    PatternAnalyzerAvailable = False
    ArticleAnalyzer = None

class IntelligentFactGuru:
    """
    Main fact verification system that combines multiple analysis techniques:
    - Semantic analysis (NLI model)
    - Source credibility assessment
    - Pattern-based fake news detection
    - Web content scraping and analysis
    """
    
    def __init__(self):
        """Initialize all system components"""
        log("🧠 Initializing Intelligent FactGuru...")
        self.input_handler = InputHandler()
        self.search_connector = SearchConnector(max_results=5)
        self.credibility_extractor = CredibilityExtractor()
        self.scraper_engine = get_scraper_engine()
        
        # Download models from Hugging Face first
        self._download_models_from_hf()
        
        # Initialize the NLI-enhanced semantic verifier
        self.semantics_verifier = SimpleNLIVerifier()
        
        # Initialize Pattern Analysis if available
        self.pattern_analyzer = None
        if PatternAnalyzerAvailable:
            try:
                # Make sure pattern models are available
                if not os.path.exists("ml/pattern_analysis"):
                    self._download_pattern_models()
                    
                self.pattern_analyzer = ArticleAnalyzer()
                log("✅ Pattern Analyzer initialized")
            except Exception as e:
                log(f"⚠️ Pattern Analyzer initialization failed: {e}")
                self.pattern_analyzer = None
        else:
            log("⚠️ Pattern Analyzer not available")
                
        log("✅ Intelligent system ready")

    def _download_models_from_hf(self):
        """Download all models from Hugging Face if not exists locally"""
        try:
            from huggingface_hub import snapshot_download
            import os
            
            # Check if models already exist locally
            if not os.path.exists("ml/model") or (os.path.exists("ml/model") and len(os.listdir("ml/model")) == 0):
                log("📥 Downloading models from Hugging Face...")
                
                # Download from your Hugging Face repository
                model_path = snapshot_download(
                    repo_id="rockOn08/factguru-models",
                    cache_dir="ml/",
                    local_dir="ml/",
                    force_download=False  # Don't re-download if exists
                )
                log("✅ All models downloaded successfully from Hugging Face")
            else:
                log("✅ Using locally cached models")
                
        except Exception as e:
            log(f"⚠️ Model download failed: {e}")
            log("ℹ️ System will use online models as fallback")

    def _download_pattern_models(self):
        """Download pattern analysis models from Hugging Face"""
        try:
            from huggingface_hub import snapshot_download
            
            log("📥 Downloading pattern models from Hugging Face...")
            snapshot_download(
                repo_id="rockOn08/factguru-models",
                allow_patterns=["*pattern_analysis*", "*.pkl"],
                cache_dir="ml/pattern_analysis",
                local_dir="ml/pattern_analysis",
                force_download=False
            )
            log("✅ Pattern models downloaded successfully")
        except Exception as e:
            log(f"❌ Failed to download pattern models: {e}")

    def process_claim(self, claim: str) -> Dict[str, Any]:
        """
        Main method to process a claim through the entire verification pipeline
        
        Args:
            claim: The factual claim to verify
            
        Returns:
            Dictionary containing verification results with confidence scores
        """
        start_time = time.time()
        results = {
            "claim": claim, 
            "timestamp": start_time, 
            "components": {},
            "status": "processing"
        }

        try:
            # Step 1: Input validation and cleaning
            is_valid, error = self.input_handler.validate_claim(claim)
            if not is_valid:
                results["error"] = f"Validation failed: {error}"
                results["status"] = "failed"
                return results
            
            cleaned_claim, _ = self.input_handler.process_input(claim)

            # Step 2: Pattern analysis on the claim itself
            pattern_analysis = self._analyze_with_patterns(cleaned_claim)
            if pattern_analysis:
                results["components"]["pattern_analysis"] = pattern_analysis

            # Step 3: Search for relevant content online
            search_results = self.search_connector.search_driver(cleaned_claim)
            if not search_results:
                results["error"] = "No search results found"
                results["status"] = "failed"
                return results

            # Step 4: Scrape article content from search results
            urls = [result.get('url', '') for result in search_results]
            scraped_articles = self.scraper_engine.scrape_parallel(urls)
            
            # Filter for successful articles with sufficient content
            successful_articles = []
            for article in scraped_articles:
                if (isinstance(article, dict) and 
                    article.get('status') in ['success', 'successful'] and
                    article.get('content') and len(article.get('content', '')) > 200):
                    successful_articles.append(article)

            if not successful_articles:
                results["error"] = "No articles could be scraped with sufficient content"
                results["status"] = "failed"
                return results

            log(f"Analyzing {len(successful_articles)} articles with NLI-enhanced semantics...")

            # Step 5: Assess source credibility
            domains = [article.get('domain', 'unknown') for article in successful_articles]
            credibility_results = self.credibility_extractor.analyze_multiple_websites(domains)
            credibility_scores = {result['website']: result['score'] for result in credibility_results}
            log(f"🔍 Credibility analysis completed for {len(credibility_scores)} domains")

            # Step 6: Perform semantic analysis on each article
            semantic_results = []
            for article in successful_articles:
                article_result = self._analyze_article(article, cleaned_claim, credibility_scores)
                semantic_results.append(article_result)

            # Step 7: Aggregate all evidence and determine final verdict
            verdict_data = self._aggregate_evidence(semantic_results, pattern_analysis, cleaned_claim)
            
            # Step 8: Compile final results
            results["final_results"] = {
                "verdict": verdict_data['verdict'],
                "confidence": verdict_data['confidence'],
                "support_sources": verdict_data['support_count'],
                "contradict_sources": verdict_data['contradict_count'],
                "irrelevant_sources": verdict_data['irrelevant_count'],
                "total_sources": len(semantic_results),
                "semantic_results": semantic_results,
                "analysis_method": "ENHANCED_SEMANTIC_ANALYSIS",
                "pattern_enhanced": verdict_data.get('pattern_enhanced', False),
                "nli_enhanced": True,
                "credibility_enhanced": True,
                "average_credibility": verdict_data.get('average_credibility', 0.5),
                "combined_support_prob": verdict_data.get('combined_support_prob', 0.5),
                "combined_contradict_prob": verdict_data.get('combined_contradict_prob', 0.5)
            }

            # Add pattern analysis to final results if available
            if pattern_analysis:
                results["final_results"]["pattern_analysis"] = pattern_analysis

            results["processing_time"] = time.time() - start_time
            results["status"] = "completed"
            log("✅ Enhanced analysis with pattern analysis and credibility scoring completed")
            return results

        except Exception as e:
            log(f"❌ Processing error: {e}")
            results["error"] = str(e)
            results["status"] = "failed"
            return results

    def _analyze_article(self, article: Dict, claim: str, credibility_scores: Dict) -> Dict[str, Any]:
        """
        Analyze a single article against the claim using multiple techniques
        
        Args:
            article: Scraped article content
            claim: The claim to verify
            credibility_scores: Dictionary of domain credibility scores
            
        Returns:
            Analysis results for this article
        """
        domain = article.get('domain', 'unknown')
        title = article.get('title', '')
        content = article.get('content', '')
        
        log(f"NLI analysis: {domain}")
        
        # Semantic analysis using NLI model
        semantic_result = self.semantics_verifier.verify_claim(claim, content)
        
        # Extract NLI information if available
        nli_info = {}
        if hasattr(semantic_result, 'nli_result') and semantic_result.nli_result:
            nli_info = {
                "nli_label": semantic_result.nli_result.get('nli_label', 'N/A'),
                "nli_confidence": semantic_result.nli_result.get('confidence', 0.0)
            }
        
        # Get credibility score for this domain
        credibility_score = credibility_scores.get(domain, 0.5)
        
        # Perform pattern analysis on the article content
        article_pattern_analysis = self._analyze_article_patterns(title, content)
        
        return {
            "domain": domain,
            "title": title,
            "relation": semantic_result.relation.value,
            "confidence": semantic_result.confidence,
            "evidence": semantic_result.evidence[:3] if semantic_result.evidence else [],
            "reasoning": semantic_result.reasoning[:2] if semantic_result.reasoning else [],
            "support_score": semantic_result.support_score,
            "contradict_score": semantic_result.contradict_score,
            "nli_analysis": nli_info,
            "credibility_score": credibility_score,
            "credibility_level": self._get_credibility_level(credibility_score),
            "pattern_analysis": article_pattern_analysis
        }

    def _analyze_with_patterns(self, claim: str) -> Dict[str, Any]:
        """Analyze claim using pattern recognition for fake news indicators"""
        if not self.pattern_analyzer:
            return None
            
        try:
            # Using claim as both title and content for pattern analysis
            pattern_results = self.pattern_analyzer.analyze_article(claim, "")  
            log(f"🔍 Pattern Analysis completed: {pattern_results['prediction']}")
            return pattern_results
            
        except Exception as e:
            log(f"⚠️ Pattern analysis failed: {e}")
            return None

    def _analyze_article_patterns(self, title: str, content: str) -> Dict[str, Any]:
        """Analyze individual article for suspicious patterns and fake news indicators"""
        if not self.pattern_analyzer:
            return None
            
        try:
            pattern_results = self.pattern_analyzer.analyze_article(title, content)
            return {
                "prediction": pattern_results['prediction'],
                "suspicious_words": pattern_results['suspicious_words'],
                "suspicious_word_count": pattern_results['suspicious_word_count'],
                "clickbait_score": pattern_results['clickbait_score'],
                "confidence": pattern_results['confidence']
            }
        except Exception as e:
            log(f"⚠️ Article pattern analysis failed: {e}")
            return None

    def _get_credibility_level(self, score: float) -> str:
        """Convert numerical credibility score to human-readable level"""
        if score >= 0.9:
            return "Very High"
        elif score >= 0.8:
            return "High"
        elif score >= 0.7:
            return "Medium"
        elif score >= 0.6:
            return "Low"
        else:
            return "Very Low"

    def _aggregate_evidence(self, semantic_results: List[Dict], pattern_analysis: Dict, claim: str) -> Dict:
        """
        Aggregate evidence from all sources using log odds probability combination
        
        This method uses proper Bayesian probability theory to combine evidence
        from multiple sources, giving more weight to strong evidence and
        properly handling conflicting information.
        """
        support_probs = []
        contradict_probs = []
        credibility_scores = []
        
        for result in semantic_results:
            base_confidence = result['confidence']
            credibility_score = result.get('credibility_score', 0.5)
            credibility_scores.append(credibility_score)
            
            # Apply credibility adjustment (0.7-1.0 multiplier based on source trust)
            credibility_adjustment = 0.7 + (0.3 * credibility_score)
            adjusted_confidence = base_confidence * credibility_adjustment
            
            # Apply pattern analysis adjustment if available
            pattern_data = result.get('pattern_analysis', {})
            if pattern_data and pattern_data.get('prediction') == 'FAKE':
                adjusted_confidence *= 0.8  # 20% penalty for fake patterns
            elif pattern_data and pattern_data.get('prediction') == 'TRUE':
                adjusted_confidence *= 1.1  # 10% boost for clean patterns
            
            # Convert to probabilities for evidence combination
            relation = result['relation']
            if relation == 'support':
                support_probs.append(adjusted_confidence)
                contradict_probs.append(1.0 - adjusted_confidence)
            elif relation == 'contradict':
                contradict_probs.append(adjusted_confidence)
                support_probs.append(1.0 - adjusted_confidence)
            else:  # irrelevant
                # Treat irrelevant sources as neutral evidence (0.5 probability)
                support_probs.append(0.5)
                contradict_probs.append(0.5)
        
        # Combine probabilities using log odds (proper Bayesian approach)
        combined_support_prob = self._combine_probabilities(support_probs)
        combined_contradict_prob = self._combine_probabilities(contradict_probs)
        
        # Determine verdict based on combined probabilities
        verdict, confidence = self._determine_verdict(
            combined_support_prob, 
            combined_contradict_prob
        )
        
        # Count sources for reporting
        support_count = sum(1 for r in semantic_results if r['relation'] == 'support')
        contradict_count = sum(1 for r in semantic_results if r['relation'] == 'contradict')
        irrelevant_count = sum(1 for r in semantic_results if r['relation'] == 'irrelevant')
        avg_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.5
        
        # Apply pattern analysis enhancement to final verdict
        final_verdict, final_confidence = self._enhance_with_pattern_analysis(
            verdict, confidence, pattern_analysis
        )
        
        log(f"🔍 Evidence Aggregation: {support_count} supports, {contradict_count} contradicts")
        log(f"🔍 Combined Probabilities: Support={combined_support_prob:.3f}, Contradict={combined_contradict_prob:.3f}")
        log(f"🔍 Final Verdict: {final_verdict} ({final_confidence:.3f})")
        
        return {
            'verdict': final_verdict,
            'confidence': final_confidence,
            'support_count': support_count,
            'contradict_count': contradict_count,
            'irrelevant_count': irrelevant_count,
            'pattern_enhanced': pattern_analysis is not None,
            'average_credibility': avg_credibility,
            'combined_support_prob': combined_support_prob,
            'combined_contradict_prob': combined_contradict_prob
        }

    def _combine_probabilities(self, probabilities: List[float]) -> float:
        """
        Combine multiple probabilities using log odds method
        
        This is mathematically equivalent to multiplying probabilities
        but more numerically stable and properly handles extreme values.
        """
        if not probabilities:
            return 0.5  # Neutral probability if no evidence
            
        # Convert probabilities to log odds
        log_odds = []
        for prob in probabilities:
            # Avoid log(0) and division by zero
            safe_prob = max(min(prob, 0.999), 0.001)
            odds = safe_prob / (1 - safe_prob)
            log_odds.append(math.log(odds))
        
        # Sum log odds (equivalent to multiplying probabilities)
        total_log_odds = sum(log_odds)
        
        # Convert back to probability
        total_prob = math.exp(total_log_odds) / (1 + math.exp(total_log_odds))
        return total_prob

    def _determine_verdict(self, support_prob: float, contradict_prob: float) -> Tuple[str, float]:
        """
        Determine final verdict based on combined probabilities
        
        Uses confidence thresholds to ensure reliable verdicts and
        returns 'CAN'T SAY' when evidence is weak or conflicting.
        """
        confidence_threshold = 0.6  # Minimum confidence for TRUE/FALSE verdict
        
        if support_prob > contradict_prob and support_prob > confidence_threshold:
            return "TRUE", support_prob
        elif contradict_prob > support_prob and contradict_prob > confidence_threshold:
            return "FALSE", contradict_prob
        else:
            # Evidence is weak, conflicting, or below threshold
            max_confidence = max(support_prob, contradict_prob)
            return "CAN'T SAY", max_confidence

    def _enhance_with_pattern_analysis(self, base_verdict: str, base_confidence: float, 
                                     pattern_analysis: Dict) -> Tuple[str, float]:
        """
        Enhance the verdict with pattern analysis when appropriate
        
        Pattern analysis can override semantic analysis only when:
        1. Pattern confidence is high (>70%)
        2. Semantic evidence is weak (<60% confidence)
        3. Pattern analysis strongly disagrees with semantic verdict
        """
        if not pattern_analysis:
            return base_verdict, base_confidence
            
        try:
            pattern_prediction = pattern_analysis.get('prediction', '').upper()
            pattern_confidence = pattern_analysis.get('confidence', 50) / 100  # Convert to 0-1
            
            # Check if pattern analysis should override semantic verdict
            should_override = (
                pattern_prediction != base_verdict and 
                pattern_confidence > 0.7 and 
                base_confidence < 0.6
            )
            
            if should_override:
                log(f"🔍 Pattern analysis strongly suggests {pattern_prediction}, adjusting verdict")
                # Use pattern verdict but with reduced confidence
                return pattern_prediction, pattern_confidence * 0.8
            else:
                # Weighted combination: 70% semantic, 30% pattern
                enhanced_confidence = (base_confidence * 0.7) + (pattern_confidence * 0.3)
                return base_verdict, enhanced_confidence
                
        except Exception as e:
            log(f"⚠️ Pattern enhancement failed: {e}")
            return base_verdict, base_confidence

    def display_results(self, results: Dict[str, Any]):
        """Display verification results in a user-friendly format"""
        if "error" in results:
            print(f"\n❌ Error: {results['error']}")
            return

        final = results["final_results"]
        verdict = final['verdict']
        confidence = final['confidence']
        
        print("\n" + "=" * 70)
        print("🧠 ENHANCED FACT VERIFICATION RESULTS")
        print("=" * 70)
        print(f"📝 CLAIM: {results['claim']}")
        print(f"⏱️  Processing Time: {results['processing_time']:.2f}s")
        
        # Show enhancement indicators
        enhancements = []
        if final.get('nli_enhanced'):
            enhancements.append("NLI Model")
        if final.get('pattern_enhanced'):
            enhancements.append("Pattern Analysis")
        if final.get('credibility_enhanced'):
            enhancements.append("Credibility Scoring")
        
        if enhancements:
            print(f"🔬 Enhanced with: {', '.join(enhancements)}")
        
        # Display verdict with appropriate emoji and formatting
        if verdict == "TRUE":
            print(f"✅ VERDICT: TRUE (Confidence: {confidence:.1%})")
        elif verdict == "FALSE":
            print(f"❌ VERDICT: FALSE (Confidence: {confidence:.1%})")
        else:
            print(f"⚠️  VERDICT: CAN'T SAY (Confidence: {confidence:.1%})")
            
        print(f"\n📊 EVIDENCE SUMMARY:")
        print(f"   ✅ Supporting sources: {final['support_sources']}")
        print(f"   ❌ Contradicting sources: {final['contradict_sources']}")
        print(f"   ⚪ Irrelevant sources: {final['irrelevant_sources']}")
        print(f"   📚 Total sources analyzed: {final['total_sources']}")
        print(f"   🏆 Average source credibility: {final['average_credibility']:.1%}")

        # Display Pattern analysis if available
        if final.get('pattern_analysis'):
            pattern_data = final['pattern_analysis']
            print(f"\n🔍 PATTERN ANALYSIS:")
            print(f"   Prediction: {pattern_data.get('prediction', 'N/A')}")
            print(f"   Confidence: {pattern_data.get('confidence', 0):.1%}")
            if pattern_data.get('suspicious_words'):
                print(f"   Suspicious Words: {', '.join(pattern_data['suspicious_words'])}")
            else:
                print(f"   Suspicious Words: None")
            print(f"   Clickbait Score: {pattern_data.get('clickbait_score', 0)}")

        print(f"\n🔍 DETAILED ANALYSIS:")
        for result in final['semantic_results']:
            relation = result['relation'].upper()
            if relation == "SUPPORT":
                emoji = "✅"
            elif relation == "CONTRADICT":
                emoji = "❌"
            else:
                emoji = "⚪"
                
            line = f"{emoji} {result['domain']:25} | {relation:12} | Confidence: {result['confidence']:.3f}"
            
            # Add credibility info
            if result.get('credibility_score'):
                cred_level = result.get('credibility_level', 'Unknown')
                line += f" | 🏆 {cred_level}"
                
            # Add pattern info
            if result.get('pattern_analysis'):
                pattern_pred = result['pattern_analysis'].get('prediction', '')
                line += f" | 🔍 {pattern_pred}"
                
            print(line)
            
        print("=" * 70)

def main():
    """Main function to run the fact verification system interactively"""
    system = IntelligentFactGuru()
    print("\n🧠 Enhanced Fact Verification System")
    enhancements = []
    if system.pattern_analyzer:
        enhancements.append("Pattern Analysis")
    enhancements.append("Credibility Engine")
    
    print(f"   Powered by: {', '.join(enhancements)}")
    print()
    
    while True:
        claim = input("📝 Enter claim to verify (or 'quit'): ").strip()
        if claim.lower() in ("quit", "exit"):
            print("👋 Goodbye!")
            break
        if not claim:
            continue
            
        print("\n🔄 Analyzing with enhanced verification...")
        result = system.process_claim(claim)
        system.display_results(result)
        print()

if __name__ == "__main__":
    main()
