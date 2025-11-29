import joblib
import re
import numpy as np
from scipy.sparse import hstack
import os

class ArticleAnalyzer:
    def __init__(self):
        """
        Initialize the analyzer with pre-trained model and vectorizer
        """
        # Get current directory (pattern_analysis folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Model file paths - all files are in the same directory
        
        model_path = os.path.join(current_dir, 'fake_news_model_improved.pkl')
        vectorizer_path = os.path.join(current_dir, 'tfidf_vectorizer_naivebayes_clickbait.pkl')
        
        print(f"Loading model from: {model_path}")
        print(f"Loading vectorizer from: {vectorizer_path}")
        
        # Load model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        self.sensational_words = [
            "shocking", "unbelievable", "secret", "exposed", "breaking", "truth", "viral",
            "you won t believe", "government hides", "hidden", "scandal", "conspiracy",
            "alert", "banned", "leaked", "danger", "fake", "hoax", "will replace", 
            "plan revealed", "will blow your mind", "you wont believe", "exclusive"
        ]
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def analyze_patterns(self, text):
        """
        Analyze text for suspicious patterns and return detailed analysis
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Find suspicious words
        found_words = [word for word in self.sensational_words if word in cleaned_text]
        word_count = len(found_words)
        
        # Transform text for model prediction
        text_tfidf = self.vectorizer.transform([cleaned_text])
        clickbait_col = np.array([word_count]).reshape(-1, 1)
        final_features = hstack([text_tfidf, clickbait_col])
        
        # Make prediction
        prediction = self.model.predict(final_features)[0]
        prediction_proba = self.model.predict_proba(final_features)[0]
        
        # Determine prediction label
        prediction_label = "FAKE" if prediction == 1 else "TRUE"
        confidence = max(prediction_proba)
        
        return {
            'prediction': prediction_label,
            'confidence': round(confidence * 100, 2),
            'suspicious_words': found_words,
            'suspicious_word_count': word_count,
            'clickbait_score': word_count,
            'analysis': f"Article classified as {prediction_label} with {confidence:.1%} confidence. Found {word_count} suspicious pattern(s)."
        }
    
    def analyze_article(self, title, text):
        """
        Analyze a complete article with title and text
        """
        combined_text = f"{title} {text}"
        return self.analyze_patterns(combined_text)

def test_analyzer():
    """Test the analyzer with sample articles"""
    try:
        analyzer = ArticleAnalyzer()
        
        # Test cases
        test_articles = [
            {
                "title": "Breaking: Government Hides Secret Truth About Vaccine",
                "text": "Shocking revelation that will blow your mind. The government has been hiding this secret for years."
            },
            {
                "title": "Regular Weather Report",
                "text": "Today's weather will be sunny with a high of 25 degrees Celsius."
            }
        ]
        
        print("🔍 ARTICLE PATTERN ANALYZER")
        print("=" * 50)
        
        for i, article in enumerate(test_articles, 1):
            print(f"\n📄 Article {i} Analysis:")
            print(f"Title: {article['title']}")
            
            result = analyzer.analyze_article(article['title'], article['text'])
            
            print(f"🔮 Prediction: {result['prediction']} ({result['confidence']}% confidence)")
            print(f"📊 Clickbait Score: {result['clickbait_score']}")
            print(f"🔎 Suspicious Words Found: {result['suspicious_word_count']}")
            
            if result['suspicious_words']:
                print(f"🚨 Suspicious Patterns: {', '.join(result['suspicious_words'])}")
            else:
                print("✅ No suspicious patterns detected")
            
            print(f"💡 Analysis: {result['analysis']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure these files are in the same directory:")
        print("- fake_news_model_naivebayes_clickbait.pkl")
        print("- tfidf_vectorizer_naivebayes_clickbait.pkl")

if __name__ == "__main__":
    test_analyzer()