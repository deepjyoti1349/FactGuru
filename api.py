from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sys
import os
import json
import asyncio
import time
import uuid
from typing import Dict

sys.path.append(os.path.dirname(__file__))

from orchastrator import IntelligentFactGuru  # Fixed spelling

app = FastAPI(title="FactGuru API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimRequest(BaseModel):
    claim: str

# Initialize the system
system = IntelligentFactGuru()

# Progress tracking stages
PROGRESS_STAGES = [
    {"id": "stageInput", "name": "Input Processing", "progress": 5},
    {"id": "stageML", "name": "ML Pattern Analysis", "progress": 15},
    {"id": "stageSearch", "name": "Searching Web", "progress": 35},
    {"id": "stageScrape", "name": "Scraping Articles", "progress": 55},
    {"id": "stageSemantics", "name": "Semantic Analysis", "progress": 80},
    {"id": "stageCredibility", "name": "Credibility Check", "progress": 90},
    {"id": "stageAggregate", "name": "Aggregating Results", "progress": 100}
]

# Store progress in memory (use Redis in production)
progress_store: Dict[str, Dict] = {}

@app.post("/verify")
async def verify_claim(request: ClaimRequest):
    """Original verification endpoint - maintains existing functionality"""
    try:
        result = system.process_claim(request.claim)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-with-progress")
async def verify_claim_with_progress(request: ClaimRequest, background_tasks: BackgroundTasks):
    """New endpoint with progress tracking"""
    request_id = str(uuid.uuid4())
    
    # Initialize progress
    progress_store[request_id] = {
        "current_stage": 0,
        "stage_name": "Starting...",
        "message": "Starting verification...",
        "progress": 0,
        "complete": False,
        "error": None,
        "result": None,
        "logs": [],
        "timestamp": time.time()  # Add timestamp for cleanup
    }
    
    # Start verification in background
    background_tasks.add_task(run_verification_with_progress, request.claim, request_id)
    
    return {"request_id": request_id, "status": "started"}

@app.get("/progress/{request_id}")
async def get_progress(request_id: str):
    """Get current progress for a request"""
    if request_id not in progress_store:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return progress_store[request_id]

def update_progress(request_id: str, stage: int, message: str):
    """Update progress for a request"""
    if request_id in progress_store:
        stage_data = PROGRESS_STAGES[stage]
        progress_store[request_id].update({
            "current_stage": stage,
            "stage_name": stage_data["name"],
            "message": message,
            "progress": stage_data["progress"],
            "complete": False,
            "error": None
        })

def add_log(request_id: str, message: str, level: str = "info"):
    """Add log message to progress"""
    if request_id in progress_store:
        progress_store[request_id]["logs"].append({
            "timestamp": time.time(),
            "message": message,
            "level": level
        })

async def run_verification_with_progress(claim: str, request_id: str):
    """Run verification with progress tracking - mirrors original process_claim logic"""
    start_time = time.time()
    
    try:
        # Stage 0: Input Processing
        update_progress(request_id, 0, "Processing input claim...")
        add_log(request_id, "🧠 Starting NLI-enhanced fact verification")
        
        # Input validation (from original process_claim)
        is_valid, error = system.input_handler.validate_claim(claim)
        if not is_valid:
            progress_store[request_id].update({
                "error": f"Validation failed: {error}",
                "complete": True
            })
            return
        
        cleaned_claim, _ = system.input_handler.process_input(claim)
        add_log(request_id, "✅ Input processed successfully")
        
        # Stage 1: ML Pattern Analysis
        update_progress(request_id, 1, "Running ML pattern analysis...")
        add_log(request_id, "🔍 Running ML pattern analysis...")
        
        # Use the correct method name from orchestrator
        pattern_analysis = system._analyze_with_patterns(cleaned_claim)
        if pattern_analysis:
            add_log(request_id, "✅ Pattern analysis completed")
        else:
            add_log(request_id, "⚠️ Pattern analysis not available")
        
        # Stage 2: Search
        update_progress(request_id, 2, "Searching for relevant content...")
        add_log(request_id, "🌐 Searching web for relevant content...")
        
        search_results = system.search_connector.search_driver(cleaned_claim)
        add_log(request_id, f"✅ Found {len(search_results)} search results")
        
        if not search_results:
            progress_store[request_id].update({
                "error": "No search results found",
                "complete": True
            })
            return
        
        # Stage 3: Scraping
        update_progress(request_id, 3, "Scraping article content...")
        add_log(request_id, "📄 Scraping article content...")
        
        urls = [r.get('url', '') for r in search_results]
        scraped_articles = system.scraper_engine.scrape_parallel(urls)
        successful_articles = []
        for article in scraped_articles:
            if (isinstance(article, dict) and 
                article.get('status') in ['success', 'successful'] and
                article.get('content') and len(article.get('content', '')) > 200):
                successful_articles.append(article)
        
        add_log(request_id, f"✅ Successfully scraped {len(successful_articles)} articles")
        
        if not successful_articles:
            progress_store[request_id].update({
                "error": "No articles could be scraped",
                "complete": True
            })
            return
        
        # Stage 4: Semantic Analysis with Credibility
        update_progress(request_id, 4, "Running semantic analysis with credibility scoring...")
        add_log(request_id, "🤖 Running NLI-enhanced semantic analysis with credibility...")
        
        # Extract domains and get credibility scores (from orchestrator)
        domains = [article.get('domain', 'unknown') for article in successful_articles]
        credibility_results = system.credibility_extractor.analyze_multiple_websites(domains)
        credibility_scores = {result['website']: result['score'] for result in credibility_results}
        add_log(request_id, f"🔍 Credibility analysis completed for {len(credibility_scores)} domains")
        
        semantic_results = []
        for i, article in enumerate(successful_articles):
            domain = article.get('domain', 'unknown')
            title = article.get('title', '')
            content = article.get('content', '')
            
            add_log(request_id, f"🔍 Analyzing article {i+1}/{len(successful_articles)} from {domain}")
            
            result = system.semantics_verifier.verify_claim(cleaned_claim, content)
            
            # Extract NLI information if available
            nli_info = {}
            if hasattr(result, 'nli_result') and result.nli_result:
                nli_info = {
                    "nli_label": result.nli_result.get('nli_label', 'N/A'),
                    "nli_confidence": result.nli_result.get('confidence', 0.0)
                }
            
            # Get credibility score for this domain
            credibility_score = credibility_scores.get(domain, 0.5)
            
            # Perform pattern analysis on the article content
            article_pattern_analysis = system._analyze_article_patterns(title, content)
            
            semantic_results.append({
                "domain": domain,
                "title": title,
                "relation": result.relation.value,
                "confidence": result.confidence,
                "evidence": result.evidence[:3] if result.evidence else [],
                "reasoning": result.reasoning[:2] if result.reasoning else [],
                "support_score": result.support_score,
                "contradict_score": result.contradict_score,
                "nli_analysis": nli_info,
                "credibility_score": credibility_score,
                "credibility_level": system._get_credibility_level(credibility_score),
                "pattern_analysis": article_pattern_analysis
            })
        
        add_log(request_id, "✅ Semantic analysis with credibility completed")
        
        # Stage 5: Aggregation
        update_progress(request_id, 5, "Aggregating final results...")
        add_log(request_id, "📈 Aggregating final results with pattern enhancement...")
        
        # Use the same aggregation logic as original process_claim
        #verdict_data = system._aggregate_intelligently(semantic_results, pattern_analysis, cleaned_claim)
        verdict_data = system._aggregate_evidence(semantic_results, pattern_analysis, cleaned_claim)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build final result matching orchestrator format
        final_result = {
            "claim": claim,
            "timestamp": start_time,
            "components": {"pattern_analysis": pattern_analysis} if pattern_analysis else {},
            "final_results": {
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
                "average_credibility": verdict_data.get('average_credibility', 0.5)
            },
            "processing_time": processing_time,
            "status": "completed"
        }

        # Add pattern analysis to final results if available (matching orchestrator)
        if pattern_analysis:
            final_result["final_results"]["pattern_analysis"] = pattern_analysis
        
        progress_store[request_id].update({
            "complete": True,
            "result": final_result,
            "message": "Analysis complete!"
        })
        add_log(request_id, "🎉 Verification completed successfully")
        
    except Exception as e:
        error_msg = f"Error during verification: {str(e)}"
        progress_store[request_id].update({
            "error": error_msg,
            "complete": True
        })
        add_log(request_id, f"❌ {error_msg}", "error")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "FactGuru API"}

# Cleanup old progress data
import threading
def cleanup_old_progress():
    """Clean up progress data older than 1 hour"""
    while True:
        current_time = time.time()
        keys_to_delete = []
        for key, value in progress_store.items():
            if current_time - value.get('timestamp', 0) > 3600:  # 1 hour
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del progress_store[key]
        
        time.sleep(3600)  # Run every hour

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_progress, daemon=True)
cleanup_thread.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)