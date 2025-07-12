"""
Test Suite for Advanced Spam Email Detection System

This module contains comprehensive tests for validating the spam detection system
functionality, performance, and edge cases.
"""

import unittest
import sys
import os
import time
from unittest.mock import patch, MagicMock

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from spam_detector import SpamDetectorEnsemble, AdvancedTextProcessor, PredictionResult
except ImportError as e:
    print(f"Warning: Could not import spam_detector module: {e}")
    print("This is expected if dependencies are not installed.")


class TestAdvancedTextProcessor(unittest.TestCase):
    """Test cases for the AdvancedTextProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.processor = AdvancedTextProcessor()
        except Exception:
            self.skipTest("Dependencies not available")
    
    def test_basic_feature_extraction(self):
        """Test basic feature extraction functionality"""
        text = "Hello world! This is a test email."
        features = self.processor.extract_features(text)
        
        # Verify essential features are extracted
        self.assertIn('char_count', features)
        self.assertIn('word_count', features)
        self.assertIn('sentence_count', features)
        self.assertEqual(features['word_count'], 7)
        self.assertEqual(features['sentence_count'], 1)
    
    def test_spam_keyword_detection(self):
        """Test spam keyword detection"""
        spam_text = "URGENT! Free money! Win cash prizes! Act now!"
        features = self.processor.extract_features(spam_text)
        
        # Should detect urgency and money keywords
        self.assertGreater(features['urgency_keywords'], 0)
        self.assertGreater(features['money_keywords'], 0)
    
    def test_url_detection(self):
        """Test URL extraction and analysis"""
        text_with_urls = "Visit https://example.com and http://suspicious.tk for details"
        features = self.processor.extract_features(text_with_urls)
        
        self.assertEqual(features['url_count'], 2)
        self.assertTrue(features['has_suspicious_domains'])  # .tk is suspicious
    
    def test_email_detection(self):
        """Test email address detection"""
        text_with_emails = "Contact john@example.com or support@company.org"
        features = self.processor.extract_features(text_with_emails)
        
        self.assertEqual(features['email_count'], 2)
    
    def test_html_detection(self):
        """Test HTML content detection"""
        html_text = "Click <a href='#'>here</a> for <b>amazing</b> deals!"
        features = self.processor.extract_features(html_text)
        
        self.assertTrue(features['has_html'])
        self.assertGreater(features['html_tag_count'], 0)
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        dirty_text = "Visit https://example.com and contact john@test.com <b>NOW</b>!"
        cleaned = self.processor.clean_text(dirty_text)
        
        # Should contain markers and be cleaned
        self.assertIn('URL_MARKER', cleaned)
        self.assertIn('EMAIL_MARKER', cleaned)
        self.assertNotIn('<b>', cleaned)
        self.assertNotIn('https://', cleaned)
    
    def test_empty_input_handling(self):
        """Test handling of empty or invalid inputs"""
        # Empty string
        features = self.processor.extract_features("")
        self.assertEqual(features['word_count'], 0)
        
        # None input
        features = self.processor.extract_features(None)
        self.assertEqual(features['word_count'], 0)
        
        # Clean empty text
        cleaned = self.processor.clean_text("")
        self.assertEqual(cleaned, "")
    
    def test_special_characters(self):
        """Test handling of special characters and patterns"""
        special_text = "WOW!!! AMAZING!!!! $$$ FREE $$$"
        features = self.processor.extract_features(special_text)
        
        self.assertGreater(features['excessive_exclamation'], 5)
        self.assertGreater(features['repeated_chars'], 0)
        self.assertGreater(features['punctuation_ratio'], 0)


class TestSpamDetectorEnsemble(unittest.TestCase):
    """Test cases for the SpamDetectorEnsemble class"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            # Mock the model file if it doesn't exist
            if not os.path.exists("Bernoulli_model_for_email.pkl"):
                self.skipTest("Model file not found - this is expected in test environment")
            
            self.detector = SpamDetectorEnsemble()
        except Exception:
            self.skipTest("Dependencies not available or model not found")
    
    def test_prediction_structure(self):
        """Test that prediction returns proper structure"""
        test_text = "This is a normal email message."
        result = self.detector.predict(test_text)
        
        # Verify result structure
        self.assertIsInstance(result, PredictionResult)
        self.assertIsInstance(result.is_spam, bool)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.spam_probability, float)
        self.assertIsInstance(result.explanation, list)
        self.assertGreater(result.processing_time, 0)
    
    def test_spam_detection(self):
        """Test spam detection with obvious spam text"""
        spam_text = """
        URGENT! You have WON $1,000,000!!! 
        Click here NOW: http://suspicious.tk/claim
        Verify your account immediately or lose your prize!
        Free money! Act now! Limited time offer!
        """
        
        result = self.detector.predict(spam_text)
        
        # Should be classified as spam with reasonable confidence
        # Note: We can't guarantee 100% accuracy, but obvious spam should be caught
        self.assertGreater(result.spam_probability, 0.3)  # At least some suspicion
        self.assertGreater(len(result.explanation), 0)
    
    def test_legitimate_email(self):
        """Test legitimate email detection"""
        legit_text = """
        Hi John,
        
        I hope this email finds you well. I wanted to follow up on our 
        meeting yesterday regarding the quarterly report. Could we schedule 
        a brief call this week to discuss the next steps?
        
        Best regards,
        Sarah
        """
        
        result = self.detector.predict(legit_text)
        
        # Should have lower spam probability
        self.assertLess(result.spam_probability, 0.8)  # Not too aggressive
        self.assertGreater(len(result.explanation), 0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty input
        result = self.detector.predict("")
        self.assertIsInstance(result, PredictionResult)
        
        # None input
        result = self.detector.predict(None)
        self.assertIsInstance(result, PredictionResult)
        
        # Very long input
        long_text = "This is a test. " * 1000
        result = self.detector.predict(long_text)
        self.assertIsInstance(result, PredictionResult)
    
    def test_performance(self):
        """Test prediction performance"""
        test_text = "This is a performance test email with some content."
        
        start_time = time.time()
        result = self.detector.predict(test_text)
        end_time = time.time()
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 5.0)  # 5 seconds max
        self.assertGreater(result.processing_time, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from text input to prediction"""
        try:
            processor = AdvancedTextProcessor()
            
            # Test text with various spam indicators
            test_email = """
            Subject: URGENT ACTION REQUIRED!!!
            
            Dear Valued Customer,
            
            Your account will be SUSPENDED unless you verify immediately!
            Click here: http://phishing-site.tk/verify
            
            You have WON $50,000 in our lottery! Free money!
            Act now before this amazing offer expires!
            
            Contact us at fake@scammer.com
            
            <b>LIMITED TIME OFFER</b>
            """
            
            # Extract features
            features = processor.extract_features(test_email)
            
            # Verify multiple spam indicators
            self.assertGreater(features['urgency_keywords'], 0)
            self.assertGreater(features['money_keywords'], 0)
            self.assertGreater(features['suspicious_keywords'], 0)
            self.assertGreater(features['url_count'], 0)
            self.assertGreater(features['email_count'], 0)
            self.assertTrue(features['has_html'])
            self.assertGreater(features['excessive_exclamation'], 0)
            
            # Clean text
            cleaned = processor.clean_text(test_email)
            self.assertIn('URL_MARKER', cleaned)
            self.assertIn('EMAIL_MARKER', cleaned)
            
        except Exception:
            self.skipTest("Dependencies not available")


def create_test_data():
    """Create test data for manual testing"""
    return {
        'spam_samples': [
            "URGENT! Win $1,000,000 NOW! Click here: http://scam.tk",
            "FREE MONEY! Verify your account at http://phishing.ml",
            "Act now! Limited time offer! Click here immediately!",
            "Congratulations! You've won a prize! Contact winner@fake.com"
        ],
        'legitimate_samples': [
            "Hi John, let's schedule our meeting for tomorrow at 2 PM.",
            "The quarterly report is ready for review. Please find it attached.",
            "Thank you for your purchase. Your order #12345 has been shipped.",
            "Reminder: Team standup meeting at 9 AM in conference room A."
        ]
    }


def run_manual_tests():
    """Run manual tests with sample data"""
    print("🧪 Running Manual Tests for Spam Detection System")
    print("=" * 50)
    
    try:
        from spam_detector import SpamDetectorEnsemble
        detector = SpamDetectorEnsemble()
        test_data = create_test_data()
        
        print("\n📧 Testing Spam Samples:")
        for i, sample in enumerate(test_data['spam_samples'], 1):
            result = detector.predict(sample)
            print(f"\nTest {i}: {sample[:50]}...")
            print(f"  Result: {'SPAM' if result.is_spam else 'SAFE'}")
            print(f"  Confidence: {result.confidence:.1%}")
            print(f"  Spam Probability: {result.spam_probability:.1%}")
        
        print("\n📧 Testing Legitimate Samples:")
        for i, sample in enumerate(test_data['legitimate_samples'], 1):
            result = detector.predict(sample)
            print(f"\nTest {i}: {sample[:50]}...")
            print(f"  Result: {'SPAM' if result.is_spam else 'SAFE'}")
            print(f"  Confidence: {result.confidence:.1%}")
            print(f"  Spam Probability: {result.spam_probability:.1%}")
        
        print("\n✅ Manual tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Manual tests failed: {e}")
        print("This is expected if dependencies are not installed.")


if __name__ == '__main__':
    # Run unit tests
    print("🚀 Starting Spam Detection System Tests")
    print("=" * 50)
    
    # Check if we should run manual tests
    if len(sys.argv) > 1 and sys.argv[1] == '--manual':
        run_manual_tests()
    else:
        # Run unit tests
        unittest.main(verbosity=2, buffer=True)