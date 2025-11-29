import sys
import os
import time
import json
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from orchastrator import IntelligentFactGuru

def format_temporal_table(semantic_results):
    """Format the temporal analysis table as text"""
    table_lines = []
    table_lines.append("Domain                    | Pub Date     | Final  | Verdict  | Semantic | Pattern  | Cred   | Temp")
    table_lines.append("-" * 95)
    
    for source in semantic_results:
        domain = source['domain'][:24].ljust(25)
        
        # Format publication date
        pub_date = source.get('publication_date_formatted') or source.get('publication_date_raw') or 'Unknown'
        if pub_date != 'Unknown' and len(pub_date) > 10:
            pub_date = pub_date[:10]
        pub_date = pub_date.ljust(12)
        
        # Format scores
        final_score = source.get('final_confidence', source['confidence'])
        semantic_score = source['confidence']
        cred_score = source.get('credibility_score', 0.5)
        temp_score = source.get('temporal_score', 1.0)
        
        # Format verdict with emoji
        relation = source['relation'].upper()
        verdict_emoji = "✅" if relation == "SUPPORT" else "❌" if relation == "CONTRADICT" else "⚪"
        
        # Format pattern prediction
        pattern_pred = source.get('pattern_analysis', {}).get('prediction', 'N/A')
        pattern_display = "TRUE" if pattern_pred == 'TRUE' else "FAKE" if pattern_pred == 'FAKE' else 'N/A'
        
        line = (f"{domain} | {pub_date} | {final_score:5.3f} | "
               f"{verdict_emoji} {relation:6} | {semantic_score:8.3f} | "
               f"{pattern_display:8} | {cred_score:6.3f} | {temp_score:5.3f}")
        
        table_lines.append(line)
    
    return "\n".join(table_lines)

def save_results_to_file(results, filename="claim_test_results.txt"):
    """Save all test results to a file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("FACTGURU CLAIM TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"CLAIM #{i}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Claim: {result['claim']}\n")
            
            if 'error' in result:
                f.write(f"❌ ERROR: {result['error']}\n\n")
                continue
            
            final = result.get('final_results', {})
            
            # Final verdict and confidence
            verdict = final.get('verdict', 'UNKNOWN')
            confidence = final.get('confidence', 0.0)
            
            verdict_emoji = "✅" if verdict == "TRUE" else "❌" if verdict == "FALSE" else "⚠️"
            f.write(f"{verdict_emoji} FINAL VERDICT: {verdict} (Confidence: {confidence:.1%})\n\n")
            
            # Evidence summary
            f.write("Evidence Summary:\n")
            f.write(f"  ✅ Supporting sources: {final.get('support_sources', 0)}\n")
            f.write(f"  ❌ Contradicting sources: {final.get('contradict_sources', 0)}\n")
            f.write(f"  ⚪ Irrelevant sources: {final.get('irrelevant_sources', 0)}\n")
            f.write(f"  📚 Total sources analyzed: {final.get('total_sources', 0)}\n")
            f.write(f"  🏆 Average credibility: {final.get('average_credibility', 0.5):.1%}\n")
            if final.get('average_temporal_score'):
                f.write(f"  ⏰ Average temporal boost: {final['average_temporal_score']:.2f}x\n")
            f.write("\n")
            
            # Temporal analysis table
            if final.get('semantic_results'):
                f.write("TEMPORAL ANALYSIS TABLE:\n")
                f.write("=" * 95 + "\n")
                table_text = format_temporal_table(final['semantic_results'])
                f.write(table_text + "\n")
                f.write("=" * 95 + "\n\n")
            
            # Pattern analysis if available
            if final.get('pattern_analysis'):
                pattern_data = final['pattern_analysis']
                f.write("Pattern Analysis:\n")
                f.write(f"  Prediction: {pattern_data.get('prediction', 'N/A')}\n")
                f.write(f"  Confidence: {pattern_data.get('confidence', 0)}%\n")
                if pattern_data.get('suspicious_words'):
                    f.write(f"  Suspicious Words: {', '.join(pattern_data['suspicious_words'])}\n")
                f.write(f"  Clickbait Score: {pattern_data.get('clickbait_score', 0)}\n\n")
            
            f.write(f"Processing Time: {result.get('processing_time', 0):.2f}s\n")
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 50 + "\n")
        successful_tests = [r for r in results if 'error' not in r]
        f.write(f"Total claims tested: {len(results)}\n")
        f.write(f"Successful analyses: {len(successful_tests)}\n")
        f.write(f"Failed analyses: {len(results) - len(successful_tests)}\n")
        
        if successful_tests:
            avg_confidence = sum(r['final_results'].get('confidence', 0) for r in successful_tests) / len(successful_tests)
            avg_time = sum(r.get('processing_time', 0) for r in successful_tests) / len(successful_tests)
            f.write(f"Average confidence: {avg_confidence:.1%}\n")
            f.write(f"Average processing time: {avg_time:.2f}s\n")
            
            # Count verdicts
            verdicts = {}
            for result in successful_tests:
                verdict = result['final_results'].get('verdict', 'UNKNOWN')
                verdicts[verdict] = verdicts.get(verdict, 0) + 1
            
            f.write("\nVerdict Distribution:\n")
            for verdict, count in verdicts.items():
                f.write(f"  {verdict}: {count} claims\n")

def test_claims():
    """Test 20 different claims and save results"""
    
    # List of 20 diverse claims to test
    test_claims = [
        "The Great Wall of China is visible from space with the naked eye",
        "Vitamin C can prevent or cure the common cold",
        "Sharks don't get cancer",
        "Humans only use 10% of their brains",
        "Lightning never strikes the same place twice",
        "Goldfish have a memory span of only 3 seconds",
        "Bulls are enraged by the color red",
        "Napoleon Bonaparte was very short",
        "Einstein failed mathematics in school",
        "The Great Wall of China is the only man-made structure visible from space",
        "You need to wait 24 hours before filing a missing person report",
        "Hair and fingernails continue to grow after death",
        "Cracking your knuckles causes arthritis",
        "Sugar causes hyperactivity in children",
        "Reading in dim light ruins your eyesight",
        "Chocolate causes acne",
        "The Great Pyramid of Giza was built by slaves",
        "The Great Barrier Reef is the largest living structure on Earth",
        "Mount Everest is the tallest mountain in the world",
        "The Amazon rainforest produces 20% of the world's oxygen"
    ]
    
    print("🧠 Initializing FactGuru system...")
    system = IntelligentFactGuru()
    
    results = []
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n🔍 Testing claim {i}/20: {claim}")
        
        try:
            start_time = time.time()
            result = system.process_claim(claim)
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                verdict = result['final_results']['verdict']
                confidence = result['final_results']['confidence']
                print(f"✅ Verdict: {verdict} (Confidence: {confidence:.1%}, Time: {processing_time:.2f}s)")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            results.append({
                'claim': claim,
                'error': str(e),
                'processing_time': 0
            })
        
        # Small delay to avoid overwhelming the system
        time.sleep(2)
    
    # Save all results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"factguru_test_results_{timestamp}.txt"
    save_results_to_file(results, filename)
    
    print(f"\n🎉 Testing completed! Results saved to: {filename}")
    
    # Print quick summary
    successful = [r for r in results if 'error' not in r]
    print(f"\n📊 Summary: {len(successful)} successful, {len(results) - len(successful)} failed")
    
    if successful:
        verdicts = {}
        for result in successful:
            verdict = result['final_results']['verdict']
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        
        print("Verdict distribution:")
        for verdict, count in verdicts.items():
            print(f"  {verdict}: {count}")

if __name__ == "__main__":
    test_claims()