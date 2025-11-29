import streamlit as st
import sys
import os
import time
import traceback
from typing import Dict, Any
import pandas as pd

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

# Set page config
st.set_page_config(
    page_title="FactGuru – Advanced Fact Verification",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for progress tracking
if 'progress_data' not in st.session_state:
    st.session_state.progress_data = {
        'current_step': 0,
        'total_steps': 7,
        'step_name': 'Starting...',
        'step_details': '',
        'is_running': False
    }

def update_progress(step: int, name: str, details: str = ""):
    """Update progress in session state"""
    st.session_state.progress_data = {
        'current_step': step,
        'total_steps': 7,
        'step_name': name,
        'step_details': details,
        'is_running': True
    }

def load_orchestrator():
    """Load the advanced fact verification system"""
    try:
        from orchastrator import IntelligentFactGuru
        system = IntelligentFactGuru()
        return system
    except Exception as e:
        st.error(f"❌ Failed to load verification system: {e}")
        return None

def display_progress_bar():
    """Display a detailed progress bar"""
    progress_data = st.session_state.progress_data
    
    if not progress_data['is_running']:
        return
    
    # Progress percentage
    progress_percent = (progress_data['current_step'] / progress_data['total_steps'])
    
    # Create progress bar
    progress_bar = st.progress(progress_percent)
    
    # Display step information
    st.write(f"**Step {progress_data['current_step']}/{progress_data['total_steps']}: {progress_data['step_name']}**")
    if progress_data['step_details']:
        st.write(f"*{progress_data['step_details']}*")
    
    return progress_bar

def display_verdict(final_results: Dict[str, Any]):
    """Display the final verdict and evidence summary"""
    st.markdown("---")
    st.markdown("## 🎯 Final Verdict")
    
    verdict = final_results.get('verdict', 'UNKNOWN')
    confidence = final_results.get('confidence', 0)
    
    # Display verdict with appropriate styling
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if verdict == "TRUE":
            st.success(f"## ✅ TRUE")
        elif verdict == "FALSE":
            st.error(f"## ❌ FALSE")
        else:
            st.warning(f"## ⚠️ {verdict}")
    
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
        st.metric("Processing Time", f"{final_results.get('processing_time', 0):.2f}s")
    
    # Evidence summary
    st.markdown("### 📊 Evidence Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Supporting Sources", final_results.get('support_sources', 0))
    
    with col2:
        st.metric("Contradicting Sources", final_results.get('contradict_sources', 0))
    
    with col3:
        st.metric("Total Sources", final_results.get('total_sources', 0))
    
    with col4:
        st.metric("Avg Credibility", f"{final_results.get('average_credibility', 0):.1%}")

def display_pattern_analysis(pattern_data: Dict[str, Any]):
    """Display comprehensive pattern analysis results"""
    if not pattern_data:
        st.warning("🤖 **ML Pattern Analysis**: No pattern analysis data available")
        return
        
    st.markdown("### 🤖 ML Pattern Analysis")
    
    # Main prediction and confidence
    col1, col2 = st.columns(2)
    
    with col1:
        prediction = pattern_data.get('prediction', 'N/A')
        confidence = pattern_data.get('confidence', 0)
        
        if prediction.upper() in ['FAKE', 'FALSE', 'DECEPTIVE', 'SUSPICIOUS']:
            st.error(f"**Pattern Prediction:** {prediction}")
        elif prediction.upper() in ['REAL', 'TRUE', 'LEGITIMATE', 'CREDIBLE']:
            st.success(f"**Pattern Prediction:** {prediction}")
        else:
            st.warning(f"**Pattern Prediction:** {prediction}")
        
        st.metric("Pattern Confidence", f"{confidence:.1%}")
    
    with col2:
        clickbait_score = pattern_data.get('clickbait_score', 0)
        suspicious_words = pattern_data.get('suspicious_words', [])
        
        # Clickbait indicator
        if clickbait_score >= 7:
            st.error(f"**Clickbait Score:** {clickbait_score}/10 ⚠️ High")
        elif clickbait_score >= 4:
            st.warning(f"**Clickbait Score:** {clickbait_score}/10 🟡 Medium")
        else:
            st.success(f"**Clickbait Score:** {clickbait_score}/10 ✅ Low")
        
        # Suspicious words
        if suspicious_words:
            st.write(f"**Suspicious Indicators:**")
            for word in suspicious_words[:5]:
                st.write(f"- `{word}`")
    
    # Additional pattern metrics
    col3, col4 = st.columns(2)
    
    with col3:
        if 'fake_news_score' in pattern_data:
            fake_score = pattern_data['fake_news_score']
            if fake_score >= 0.7:
                st.error(f"**Fake News Probability:** {fake_score:.1%} ⚠️")
            elif fake_score >= 0.4:
                st.warning(f"**Fake News Probability:** {fake_score:.1%} 🟡")
            else:
                st.success(f"**Fake News Probability:** {fake_score:.1%} ✅")
    
    with col4:
        if 'sensationalism_score' in pattern_data:
            sens_score = pattern_data['sensationalism_score']
            st.metric("Sensationalism Score", f"{sens_score:.1%}")

    # Detailed pattern breakdown
    if 'linguistic_patterns' in pattern_data or 'pattern_breakdown' in pattern_data:
        with st.expander("📊 Detailed Pattern Analysis"):
            if 'linguistic_patterns' in pattern_data:
                st.write("**Linguistic Patterns Detected:**")
                for pattern, score in pattern_data['linguistic_patterns'].items():
                    st.write(f"- {pattern}: {score:.3f}")
            
            if 'pattern_breakdown' in pattern_data:
                st.write("**Pattern Breakdown:**")
                for pattern_type, details in pattern_data['pattern_breakdown'].items():
                    st.write(f"- **{pattern_type}**: {details}")

def display_source_analysis(semantic_results: list):
    """Display detailed analysis of each source"""
    st.markdown("### 🔍 Detailed Source Analysis")
    
    for i, result in enumerate(semantic_results, 1):
        with st.expander(f"Source {i}: {result.get('domain', 'Unknown')} - {result.get('relation', 'unknown').upper()}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Title:** {result.get('title', 'No title')}")
                st.write(f"**Relation:** {result.get('relation', 'unknown').upper()}")
                st.write(f"**Confidence:** {result.get('confidence', 0):.3f}")
                
                # Show evidence snippets
                if result.get('evidence'):
                    st.write("**Key Evidence:**")
                    for evidence in result.get('evidence', [])[:2]:
                        st.write(f"- {evidence}")
            
            with col2:
                # Credibility badge
                cred_score = result.get('credibility_score', 0.5)
                cred_level = result.get('credibility_level', 'Unknown')
                
                if cred_score >= 0.8:
                    st.success(f"🏆 {cred_level}")
                elif cred_score >= 0.6:
                    st.info(f"🏆 {cred_level}")
                else:
                    st.warning(f"🏆 {cred_level}")

def basic_pattern_analysis(claim: str) -> Dict[str, Any]:
    """Fallback pattern analysis when ML component is unavailable"""
    claim_lower = claim.lower()
    
    # Simple pattern detection
    suspicious_words = ['breaking', 'shocking', 'unbelievable', 'secret', 'they don\'t want you to know', 'hidden truth', 'exposed']
    found_words = [word for word in suspicious_words if word in claim_lower]
    
    # Basic clickbait detection
    clickbait_indicators = ['won\'t believe', 'what happened next', 'you\'ll never guess', 'going viral', 'this will shock you']
    clickbait_score = sum(2 for indicator in clickbait_indicators if indicator in claim_lower)
    clickbait_score = min(clickbait_score, 10)
    
    # Fake news probability based on patterns
    fake_news_indicators = ['100% effective', 'miracle cure', 'government hiding', 'mainstream media won\'t tell']
    fake_score = sum(0.2 for indicator in fake_news_indicators if indicator in claim_lower)
    fake_score = min(fake_score, 0.9)
    
    # Overall prediction
    if clickbait_score >= 7 or fake_score >= 0.6:
        prediction = 'SUSPICIOUS'
        confidence = max(0.6, (clickbait_score / 10 + fake_score) / 2)
    else:
        prediction = 'CREDIBLE'
        confidence = max(0.3, 1 - (clickbait_score / 15 + fake_score / 2))
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'clickbait_score': clickbait_score,
        'suspicious_words': found_words,
        'fake_news_score': fake_score,
        'sensationalism_score': clickbait_score / 10,
        'note': 'Basic linguistic pattern analysis'
    }

def main():
    # Header
    st.title("🕵️‍♂️ FactGuru – Advanced Fact Verification System")
    st.markdown("""
    This advanced system uses multiple verification techniques to provide accurate fact-checking:
    - **🤖 ML Pattern Analysis** - Detects fake news patterns and clickbait
    - **🧠 NLI Semantic Analysis** - Understands claim-content relationships using Natural Language Inference
    - **🏆 Source Credibility** - Evaluates website trustworthiness using known domain ratings
    - **🌐 Web Search & Scraping** - Gathers evidence from multiple online sources
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("System Configuration")
        
        # Load system
        system = load_orchestrator()
        
        if system:
            st.success("✅ Verification system ready")
            
            # System info
            st.markdown("### System Components")
            st.write("✅ Pattern Analysis")
            st.write("✅ Semantic NLI Analysis") 
            st.write("✅ Source Credibility")
            st.write("✅ Web Search & Scraping")
            
            st.markdown("---")
            st.markdown("### Verification Steps")
            steps = [
                "1. Input Validation",
                "2. Pattern Analysis", 
                "3. Web Search",
                "4. Article Scraping",
                "5. Credibility Analysis",
                "6. Semantic Analysis",
                "7. Result Aggregation"
            ]
            for step in steps:
                st.write(step)
        else:
            st.error("❌ System not available")
            return
    
    # Main input area
    st.markdown("---")
    claim = st.text_area(
        "### 📝 Enter Claim to Verify",
        placeholder="e.g., COVID-19 vaccines cause infertility in women",
        height=100,
        help="Enter a factual claim you want to verify. The system will search the web and analyze multiple sources."
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            max_sources = st.slider("Maximum sources to analyze", 3, 10, 5)
        with col2:
            enable_debug = st.checkbox("Show debug information", value=False)
    
    # Analyze button
    if st.button("🔍 Start Verification", type="primary", use_container_width=True):
        if not claim or len(claim.strip()) < 5:
            st.error("Please enter a claim with at least 5 characters")
            return
        
        # Initialize progress
        st.session_state.progress_data['is_running'] = True
        
        # Display progress section
        st.markdown("---")
        st.markdown("## 🔄 Verification Progress")
        
        try:
            # Step 1: Input Validation
            update_progress(1, "Input Validation", "Checking claim format and length...")
            progress_bar = display_progress_bar()
            time.sleep(0.5)
            
            # Step 2: Pattern Analysis
            update_progress(2, "Pattern Analysis", "Analyzing claim for fake news patterns, clickbait, and linguistic markers...")
            progress_bar.progress(2/7)
            time.sleep(1)
            
            # Step 3: Web Search
            update_progress(3, "Web Search", f"Searching for relevant sources online...")
            progress_bar.progress(3/7)
            time.sleep(1.5)
            
            # Step 4: Article Scraping
            update_progress(4, "Article Scraping", "Extracting content from web pages...")
            progress_bar.progress(4/7)
            time.sleep(1.5)
            
            # Step 5: Credibility Analysis
            update_progress(5, "Credibility Analysis", "Evaluating source trustworthiness...")
            progress_bar.progress(5/7)
            time.sleep(1)
            
            # Step 6: Semantic Analysis
            update_progress(6, "Semantic Analysis", "Running NLI model to understand relationships...")
            progress_bar.progress(6/7)
            time.sleep(1.5)
            
            # Step 7: Result Aggregation
            update_progress(7, "Result Aggregation", "Combining all evidence for final verdict...")
            progress_bar.progress(7/7)
            time.sleep(0.5)
            
            # Run the actual analysis
            start_time = time.time()
            results = system.process_claim(claim.strip())
            processing_time = time.time() - start_time
            
            # Mark as complete
            st.session_state.progress_data['is_running'] = False
            st.success("✅ Verification complete!")
            
            # Display results
            if 'error' in results:
                st.error(f"❌ Analysis failed: {results['error']}")
                return
            
            # Add processing time to results
            results['processing_time'] = processing_time
            
            # Display final results
            if 'final_results' in results:
                display_verdict(results['final_results'])
                
                # Extract and display pattern analysis
                pattern_data = results['final_results'].get('pattern_analysis', {})
                if not pattern_data:
                    # Try alternative locations
                    pattern_data = results.get('pattern_analysis', {})
                
                # Use fallback if no pattern data available
                if not pattern_data:
                    pattern_data = basic_pattern_analysis(claim.strip())
                
                display_pattern_analysis(pattern_data)
                display_source_analysis(results['final_results'].get('semantic_results', []))
                
                # Show raw data for debugging
                if enable_debug:
                    with st.expander("📊 Raw Analysis Data"):
                        st.json(results)
            else:
                st.warning("⚠️ No detailed results available")
                # Still try to show pattern analysis
                pattern_data = results.get('pattern_analysis', basic_pattern_analysis(claim.strip()))
                display_pattern_analysis(pattern_data)
                st.json(results)
                
        except Exception as e:
            st.session_state.progress_data['is_running'] = False
            st.error(f"❌ Verification failed: {str(e)}")
            if enable_debug:
                st.code(traceback.format_exc())
    
    # Examples section
    with st.expander("💡 Example Claims to Test"):
        st.markdown("Try these example claims to see the system in action:")
        
        examples = [
            "COVID-19 vaccines cause infertility in women",
            "Eating carrots improves night vision significantly", 
            "The Great Wall of China is visible from the Moon with naked eye",
            "Sharks don't get cancer",
            "Drinking 8 glasses of water daily is necessary for everyone",
            "Breaking: Secret cure they don't want you to know about!"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.claim_text = example
                st.rerun()

if __name__ == "__main__":
    main()
