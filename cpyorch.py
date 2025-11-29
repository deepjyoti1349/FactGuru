"""
# flow.py
import sys
import os
import time
from typing import List, Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

class FactGuruFlow:
    def __init__(self):
        #Initialize all components for the fact-checking flow
        print("🚀 Initializing FactGuru Flow...")
        
        # Initialize all components
        try:
            from input_handler import InputHandler
            from search_connector import SearchConnector
            from credibility_extractor import CredibilityExtractor
            from scrapper_engine import ScrappingEngine, get_scraper_engine
            from ml.semantics import LightweightClaimVerifier
            from ml.predictor import get_predictor
            
            self.input_handler = InputHandler()
            self.search_connector = SearchConnector(max_results=5)
            self.credibility_extractor = CredibilityExtractor()
            self.scraper_engine = get_scraper_engine()  # Use the global instance
            self.semantics_verifier = LightweightClaimVerifier()
            self.ml_predictor = get_predictor()
            
            print("✅ All components initialized successfully!")
            
        except ImportError as e:
            print(f"❌ Failed to import components: {e}")
            raise
    
    def process_claim(self, claim: str) -> Dict[str, Any]:
        
       # Main flow: claim → search → credibility → scraping → semantics → ML analysis
        
        start_time = time.time()
        results = {
            'claim': claim,
            'timestamp': time.time(),
            'components': {}
        }
        
        print(f"\n📝 Processing claim: '{claim}'")
        print("=" * 50)
        
        # Step 1: Input Validation
        print("1️⃣ Validating input...")
        is_valid, error = self.input_handler.validate_claim(claim)
        if not is_valid:
            results['error'] = f"Input validation failed: {error}"
            return results
        
        cleaned_claim, error = self.input_handler.process_input(claim)
        results['components']['input_validation'] = {
            'status': 'success',
            'cleaned_claim': cleaned_claim
        }
        print("✅ Input validated")
        
        # Step 2: Search for relevant URLs
        print("2️⃣ Searching for relevant content...")
        search_results = self.search_connector.search_driver(cleaned_claim)
        results['components']['search'] = {
            'status': 'success' if search_results else 'no_results',
            'results_found': len(search_results),
            'urls': [result['url'] for result in search_results]
        }
        
        if not search_results:
            results['error'] = "No search results found"
            return results
        print(f"✅ Found {len(search_results)} search results")
        
        # Step 3: Extract domains and check credibility
        print("3️⃣ Analyzing source credibility...")
        domains = []
        for result in search_results:
            domain = self.credibility_extractor.extract_domain(result['url'])
            domains.append(domain)
        
        # Get credibility scores for all domains
        credibility_scores = {}
        for domain in domains:
            score_result = self.credibility_extractor.get_website_score(domain)
            credibility_scores[domain] = score_result['score']
        
        results['components']['credibility'] = {
            'status': 'success',
            'domains_analyzed': len(domains),
            'scores': credibility_scores
        }
        print(f"✅ Credibility analyzed for {len(domains)} domains")
        
        # Step 4: Scrape articles from URLs - FIXED METHOD CALL
        print("4️⃣ Scraping article content...")
        urls = [result['url'] for result in search_results if result['url'] != 'no URL']
        
        # Use the CORRECT scraper method from the robust engine
        scraped_articles = self.scraper_engine.scrape_parallel(urls)
        
        # Filter successful scrapes - with BETTER content checking
        successful_articles = []
        for article in scraped_articles:
            if (isinstance(article, dict) and 
                article.get('status') in ['success', 'successful']):  # Handle both status types
                
                content = article.get('content', '')
                title = article.get('title', '')
                
                # IMPROVED content filtering for the robust scraper
                has_valid_content = (
                    content and 
                    isinstance(content, str) and
                    len(content.strip()) > 100 and  # At least 100 characters
                    content != "no content extracted" and 
                    content != "failed to extract" and
                    content != "Error extracting content" and
                    content != "No content extracted" and
                    not content.startswith("Content too short:") and
                    not content.startswith("No meaningful content")
                )
                
                has_valid_title = (
                    title and 
                    isinstance(title, str) and
                    title.strip() and
                    title != "no title found" and
                    title != "failed to extract" and
                    title != "Error - Failed to scrape" and
                    title != "No title found"
                )
                
                if has_valid_content and has_valid_title:
                    successful_articles.append(article)
                    print(f"   ✅ Valid article: {article.get('domain')} - '{title[:50]}...' ({len(content)} chars)")
                else:
                    print(f"   ⚠️  Skipped: {article.get('domain')} - Title: '{title}', Content: {len(content) if content else 0} chars")
        
        results['components']['scraping'] = {
            'status': 'success',
            'urls_attempted': len(urls),
            'articles_successful': len(successful_articles),
            'articles': successful_articles
        }
        print(f"✅ Successfully scraped {len(successful_articles)} valid articles")
        
        if not successful_articles:
            # Show why articles were filtered out for debugging
            print(f"🔍 Debug: All {len(scraped_articles)} scraped articles:")
            for i, article in enumerate(scraped_articles):
                status = article.get('status', 'unknown')
                title = article.get('title', 'no title')[:30]
                content_len = len(article.get('content', ''))
                domain = article.get('domain', 'unknown')
                print(f"   {i+1}. {domain} - Status: {status}, Title: '{title}...', Content: {content_len} chars")
            
            results['error'] = "No valid articles could be scraped successfully"
            return results
        
        # Step 5: Semantic analysis for each article
        print("5️⃣ Performing semantic analysis...")
        semantic_results = []
        for article in successful_articles:
            content = article.get('content', '')
            domain = article.get('domain', 'unknown')
            title = article.get('title', '')
            
            if content and len(content) > 100:  # Increased minimum content length
                try:
                    print(f"   🔍 Analyzing: {domain}")
                    semantic_result = self.semantics_verifier.verify_claim(cleaned_claim, content)
                    
                    semantic_results.append({
                        'domain': domain,
                        'title': title,
                        'relation': semantic_result.relation.value,
                        'confidence': semantic_result.confidence,
                        'evidence': semantic_result.evidence,
                        'ml_analysis': semantic_result.ml_analysis if hasattr(semantic_result, 'ml_analysis') else None
                    })
                    print(f"   ✅ {domain}: {semantic_result.relation.value} ({semantic_result.confidence:.1%})")
                except Exception as e:
                    print(f"   ❌ Semantic analysis failed for {domain}: {e}")
                    continue
        
        results['components']['semantics'] = {
            'status': 'success',
            'articles_analyzed': len(semantic_results),
            'results': semantic_results
        }
        print(f"✅ Semantic analysis completed for {len(semantic_results)} articles")
        
        # Step 6: ML Pattern analysis (parallel)
        print("6️⃣ Running ML pattern analysis...")
        
        # ML analysis for the claim itself (once)
        claim_ml_analysis = self.ml_predictor.analyze_claim(cleaned_claim)
        
        # ML analysis for each article content
        article_ml_analyses = []
        for article in successful_articles:
            content = article.get('content', '')
            if content and len(content) > 100:
                try:
                    article_analysis = self.ml_predictor.analyze_claim(content[:1000])
                    article_ml_analyses.append({
                        'domain': article.get('domain', 'unknown'),
                        'analysis': article_analysis
                    })
                except Exception as e:
                    print(f"⚠️ ML analysis failed for {article.get('domain', 'unknown')}: {e}")
                    continue
        
        results['components']['ml_analysis'] = {
            'status': 'success',
            'claim_analysis': claim_ml_analysis,
            'article_analyses': article_ml_analyses
        }
        print("✅ ML pattern analysis completed")
        
        # Step 7: Generate final results
        print("7️⃣ Generating final results...")
        results['processing_time'] = time.time() - start_time
        results['status'] = 'completed'
        
        # Compile final output
        final_output = self._compile_final_results(results)
        results['final_results'] = final_output
        
        print(f"✅ Processing completed in {results['processing_time']:.2f}s")
        return results
    
    def _compile_final_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        #Compile final results from all components
        
        # Extract key data
        credibility_scores = results['components']['credibility']['scores']
        semantic_results = results['components']['semantics']['results']
        ml_claim_analysis = results['components']['ml_analysis']['claim_analysis']
        articles = results['components']['scraping']['articles']
        
        # Find semantic matches (support/contradict)
        supporting_articles = [r for r in semantic_results if r['relation'] == 'support']
        contradicting_articles = [r for r in semantic_results if r['relation'] == 'contradict']
        
        # Calculate overall scores
        support_score = len(supporting_articles)
        contradict_score = len(contradicting_articles)
        
        # Determine final verdict
        if support_score > contradict_score:
            verdict = "SUPPORTED"
            confidence = support_score / (support_score + contradict_score) if (support_score + contradict_score) > 0 else 0.5
        elif contradict_score > support_score:
            verdict = "CONTRADICTED" 
            confidence = contradict_score / (support_score + contradict_score) if (support_score + contradict_score) > 0 else 0.5
        else:
            verdict = "UNCLEAR"
            confidence = 0.5
        
        # Incorporate ML analysis
        ml_risk = ml_claim_analysis.get('risk_level', 'unknown')
        ml_prediction = ml_claim_analysis.get('prediction', 'unknown')
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'support_score': support_score,
            'contradict_score': contradict_score,
            'sources_analyzed': len(articles),
            'credibility_scores': credibility_scores,
            'domains': list(credibility_scores.keys()),
            'articles_analyzed': len(articles),
            'ml_analysis': {
                'prediction': ml_prediction,
                'risk_level': ml_risk,
                'confidence': ml_claim_analysis.get('confidence', 0),
                'clickbait_score': ml_claim_analysis.get('clickbait_score', 0)
            },
            'semantic_matches': {
                'supporting': supporting_articles,
                'contradicting': contradicting_articles
            }
        }
    
    def display_results(self, results: Dict[str, Any]):
        #Display results in a user-friendly format
        if 'error' in results:
            print(f"\n❌ Error: {results['error']}")
            return
        
        final = results.get('final_results', {})
        
        print("\n" + "="*60)
        print("📊 FACTGURU ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n📝 CLAIM: {results['claim']}")
        print(f"⏱️  Processing Time: {results['processing_time']:.2f}s")
        
        # Final Verdict
        verdict = final.get('verdict', 'UNKNOWN')
        confidence = final.get('confidence', 0)
        
        if verdict == "SUPPORTED":
            print(f"✅ VERDICT: SUPPORTED (Confidence: {confidence:.1%})")
        elif verdict == "CONTRADICTED":
            print(f"❌ VERDICT: CONTRADICTED (Confidence: {confidence:.1%})")
        else:
            print(f"⚠️  VERDICT: UNCLEAR (Confidence: {confidence:.1%})")
        
        # Statistics
        print(f"\n📈 EVIDENCE SUMMARY:")
        print(f"   Supporting sources: {final.get('support_score', 0)}")
        print(f"   Contradicting sources: {final.get('contradict_score', 0)}")
        print(f"   Total sources analyzed: {final.get('sources_analyzed', 0)}")
        
        # ML Analysis
        ml_analysis = final.get('ml_analysis', {})
        if ml_analysis:
            print(f"\n🤖 ML ANALYSIS:")
            print(f"   Prediction: {ml_analysis.get('prediction', 'unknown').upper()}")
            print(f"   Risk Level: {ml_analysis.get('risk_level', 'unknown').upper()}")
            print(f"   Confidence: {ml_analysis.get('confidence', 0):.1%}")
            print(f"   Clickbait Score: {ml_analysis.get('clickbait_score', 0)}/10")
        
        # Credibility Scores
        credibility_scores = final.get('credibility_scores', {})
        if credibility_scores:
            print(f"\n🏛️  SOURCE CREDIBILITY:")
            for domain, score in list(credibility_scores.items())[:5]:
                rating = "HIGH" if score >= 0.8 else "MEDIUM" if score >= 0.6 else "LOW"
                print(f"   {domain}: {score:.1%} ({rating})")
        
        # Semantic Results
        semantic_matches = final.get('semantic_matches', {})
        if semantic_matches.get('supporting'):
            print(f"\n✅ SUPPORTING EVIDENCE:")
            for match in semantic_matches['supporting'][:3]:
                print(f"   • {match['domain']} (confidence: {match['confidence']:.1%})")
        
        if semantic_matches.get('contradicting'):
            print(f"\n❌ CONTRADICTING EVIDENCE:")
            for match in semantic_matches['contradicting'][:3]:
                print(f"   • {match['domain']} (confidence: {match['confidence']:.1%})")

def main():
    #Main interactive function
    try:
        flow = FactGuruFlow()
        
        print("\n" + "="*50)
        print("🔍 FACTGURU CLAIM VERIFICATION SYSTEM")
        print("="*50)
        
        while True:
            print("\n📝 Enter a claim to verify (or 'quit' to exit):")
            claim = input("> ").strip()
            
            if claim.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using FactGuru!")
                break
            
            if not claim:
                print("❌ Please enter a claim.")
                continue
            
            # Process the claim
            results = flow.process_claim(claim)
            
            # Display results
            flow.display_results(results)
            
            print("\n" + "-"*50)
    
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled. Goodbye!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""