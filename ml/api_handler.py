from .predictor import get_predictor
import json

class MLApiHandler:
    def __init__(self):
        self.predictor = get_predictor()
    
    def analyze_news_claim(self, claim_text: str) -> dict:
        """
        Main API method to analyze news claims
        
        Args:
            claim_text: The text claim to analyze
            
        Returns:
            dict: Analysis results in API-friendly format
        """
        try:
            analysis = self.predictor.analyze_claim(claim_text)
            
            # Format response for API
            response = {
                'status': 'success',
                'data': {
                    'claim': claim_text,
                    'verdict': analysis['prediction'].upper(),
                    'confidence': round(analysis['confidence'] * 100, 2),
                    'risk_assessment': {
                        'level': analysis['risk_level'],
                        'score': round(analysis['risk_score'] * 100, 2)
                    },
                    'detailed_scores': {
                        'fake_confidence': round(analysis['fake_confidence'] * 100, 2),
                        'real_confidence': round(analysis['real_confidence'] * 100, 2)
                    },
                    'pattern_analysis': {
                        'suspicious_patterns_detected': analysis['suspicious_patterns_found'],
                        'clickbait_score': analysis['clickbait_score'],
                        'detected_patterns': {
                            k: v for k, v in analysis['suspicious_patterns'].items()
                            if k not in ['overall_suspicious_score', 'clickbait_score']
                        },
                        'red_flags': self._generate_red_flags(analysis)
                    },
                    'explanation': self._generate_explanation(analysis)
                }
            }
            
            return response
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Analysis failed: {str(e)}',
                'data': None
            }
    
    def _generate_red_flags(self, analysis: dict) -> list:
        """Generate list of red flags from analysis"""
        red_flags = []
        
        if analysis['fake_confidence'] > 0.7:
            red_flags.append(f"High fake confidence ({analysis['fake_confidence']:.1%})")
        
        if analysis['clickbait_score'] > 3:
            red_flags.append(f"High clickbait score ({analysis['clickbait_score']})")
        
        patterns = analysis['suspicious_patterns']
        for pattern_type, keywords in patterns.items():
            if pattern_type not in ['overall_suspicious_score', 'clickbait_score'] and keywords:
                red_flags.append(f"{pattern_type.replace('_', ' ').title()}: {', '.join(keywords[:3])}")
        
        return red_flags[:5]  # Return top 5 red flags
    
    def _generate_explanation(self, analysis: dict) -> str:
        """Generate human-readable explanation"""
        verdict = analysis['prediction'].upper()
        confidence = analysis['confidence']
        
        if verdict == 'FAKE':
            if analysis['risk_level'] == 'high':
                return f"This claim appears highly suspicious with {confidence:.1%} confidence. Multiple red flags detected including clickbait language and sensational patterns."
            else:
                return f"This claim shows characteristics of fake news with {confidence:.1%} confidence. Some suspicious elements were identified."
        else:
            if analysis['risk_level'] == 'low':
                return f"This claim appears legitimate with {confidence:.1%} confidence. No significant suspicious patterns detected."
            else:
                return f"This claim appears legitimate but contains some sensational elements. Confidence: {confidence:.1%}"

# API instance
ml_api = MLApiHandler()