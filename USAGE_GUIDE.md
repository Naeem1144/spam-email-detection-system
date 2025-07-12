# 📖 Usage Guide: Advanced Spam Email Detection System

This guide provides step-by-step instructions for setting up and using the Advanced Spam Email Detection System.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd spam-email-detector

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, nltk, sklearn; print('✅ All dependencies installed')"
```

### 2. First Run

```bash
# Start the web application
streamlit run spam_detector.py
```

The application will automatically:
- Download required NLTK data on first run
- Load the trained model
- Open your browser to the application interface

## 🎯 Using the Web Interface

### Main Features

1. **📧 Email Analysis Panel**:
   - Paste your email content in the text area
   - Click "🔍 Analyze Email" to get instant results
   - View confidence scores and detailed explanations

2. **📊 Feature Visualization Panel**:
   - Real-time feature extraction as you type
   - Text statistics (word count, character analysis)
   - Spam indicator metrics
   - URL and domain analysis

3. **📋 Results Display**:
   - Clear SPAM/SAFE classification
   - Confidence percentage
   - Detailed explanation with specific indicators
   - Individual model predictions (when available)

### Example Usage

1. **Testing with Spam Email**:
   ```
   Input: "URGENT! Win $1,000,000 NOW! Click http://scam.tk"
   Expected: 🚨 SPAM DETECTED with high confidence
   ```

2. **Testing with Legitimate Email**:
   ```
   Input: "Hi John, let's schedule our meeting for tomorrow."
   Expected: ✅ LEGITIMATE EMAIL with high confidence
   ```

## 💻 Programmatic Usage

### Basic Usage

```python
from spam_detector import SpamDetectorEnsemble

# Initialize the detector
detector = SpamDetectorEnsemble()

# Analyze an email
email_text = """
Subject: Meeting Tomorrow

Hi Sarah,

Just confirming our meeting tomorrow at 2 PM in the conference room.
Let me know if you need to reschedule.

Best,
John
"""

result = detector.predict(email_text)

# Print results
print(f"Spam: {result.is_spam}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Probability: {result.spam_probability:.1%}")
print("\nExplanations:")
for explanation in result.explanation:
    print(f"  - {explanation}")
```

### Advanced Usage

```python
from spam_detector import AdvancedTextProcessor, SpamDetectorEnsemble

# Custom text processor
processor = AdvancedTextProcessor()

# Add custom spam keywords
processor.spam_keywords['custom'] = ['phishing', 'malware', 'trojan']

# Extract features manually
features = processor.extract_features(email_text)
print(f"Word count: {features['word_count']}")
print(f"URLs found: {features['url_count']}")
print(f"Spam keywords: {features['urgency_keywords'] + features['money_keywords']}")

# Clean text for analysis
cleaned_text = processor.clean_text(email_text)
print(f"Cleaned text: {cleaned_text}")

# Initialize detector with custom model path
detector = SpamDetectorEnsemble(model_path="path/to/your/model.pkl")
```

### Batch Processing

```python
def analyze_emails_batch(email_list):
    """Analyze multiple emails efficiently"""
    detector = SpamDetectorEnsemble()
    results = []
    
    for i, email in enumerate(email_list):
        try:
            result = detector.predict(email)
            results.append({
                'index': i,
                'is_spam': result.is_spam,
                'confidence': result.confidence,
                'spam_probability': result.spam_probability,
                'processing_time': result.processing_time
            })
        except Exception as e:
            results.append({
                'index': i,
                'error': str(e),
                'is_spam': False,
                'confidence': 0.0
            })
    
    return results

# Example usage
emails = [
    "Hi John, let's meet tomorrow",
    "URGENT! Free money! Click now!",
    "Your order #12345 has shipped"
]

results = analyze_emails_batch(emails)
for result in results:
    print(f"Email {result['index']}: {'SPAM' if result['is_spam'] else 'SAFE'}")
```

## 🧪 Testing the System

### 1. Unit Tests

```bash
# Run all tests
python test_spam_detector.py

# Run with verbose output
python test_spam_detector.py -v

# Run manual tests with sample data
python test_spam_detector.py --manual
```

### 2. Interactive Testing

Use the provided test samples:

```python
# Test with obvious spam
spam_text = """
URGENT! CONGRATULATIONS! You have won $1,000,000!!!
Click here immediately: http://phishing-site.tk/claim
Verify your account NOW or lose your prize forever!
FREE MONEY! Act now! Limited time offer!
Contact winner@scam.com for details
"""

# Test with legitimate email
legit_text = """
Subject: Quarterly Meeting Agenda

Dear Team,

Please find attached the agenda for our quarterly review meeting
scheduled for next Tuesday at 10 AM in Conference Room B.

Items to discuss:
1. Q3 performance review
2. Q4 planning
3. Budget allocation

Please review the documents beforehand and come prepared with
your department updates.

Best regards,
Sarah Johnson
Project Manager
"""

detector = SpamDetectorEnsemble()
spam_result = detector.predict(spam_text)
legit_result = detector.predict(legit_text)

print("Spam Analysis:")
print(f"  Result: {'SPAM' if spam_result.is_spam else 'SAFE'}")
print(f"  Confidence: {spam_result.confidence:.1%}")

print("\nLegitimate Email Analysis:")
print(f"  Result: {'SPAM' if legit_result.is_spam else 'SAFE'}")
print(f"  Confidence: {legit_result.confidence:.1%}")
```

## 🔧 Configuration

### 1. Model Configuration

```python
# Use custom model path
detector = SpamDetectorEnsemble(model_path="/path/to/custom/model.pkl")

# The model should be a scikit-learn Pipeline with:
# - TfidfVectorizer as first step
# - Classifier (BernoulliNB, etc.) as second step
```

### 2. Logging Configuration

```python
import logging

# Set logging level
logging.getLogger('spam_detector').setLevel(logging.DEBUG)

# Add custom handler
handler = logging.FileHandler('custom_spam_logs.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.getLogger('spam_detector').addHandler(handler)
```

### 3. Feature Engineering

```python
processor = AdvancedTextProcessor()

# Customize spam keywords
processor.spam_keywords.update({
    'phishing': ['verify', 'suspend', 'security', 'update', 'confirm'],
    'scam': ['inheritance', 'lottery', 'beneficiary', 'transfer'],
    'tech_support': ['virus', 'infected', 'support', 'fix', 'repair']
})

# Add suspicious domains
suspicious_domains = ['.tk', '.ml', '.ga', '.cf', '.click']
# This would require modifying the _check_suspicious_domains method
```

## 📊 Understanding Results

### Confidence Levels

- **90-100%**: Very high confidence in classification
- **80-89%**: High confidence
- **60-79%**: Moderate confidence
- **40-59%**: Low confidence (review recommended)
- **Below 40%**: Very low confidence (manual review needed)

### Spam Probability Interpretation

- **80-100%**: Almost certainly spam
- **60-79%**: Likely spam
- **40-59%**: Uncertain (could be either)
- **20-39%**: Likely legitimate
- **0-19%**: Almost certainly legitimate

### Common Spam Indicators

The system looks for these patterns:
- **Urgency words**: urgent, immediate, act now, expires
- **Money words**: free, cash, prize, win, lottery
- **Suspicious phrases**: click here, verify account, suspend
- **Excessive punctuation**: Multiple exclamation marks
- **Suspicious URLs**: Short domains, suspicious TLDs
- **HTML content**: Rich formatting in emails
- **Character patterns**: All caps, repeated characters

## 🚨 Troubleshooting

### Common Issues

1. **Model file not found**:
   ```
   FileNotFoundError: Bernoulli_model_for_email.pkl
   ```
   **Solution**: Ensure the model file is in the same directory as the script

2. **NLTK data missing**:
   ```
   LookupError: Resource punkt not found
   ```
   **Solution**: The system will auto-download, or run manually:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

3. **Import errors**:
   ```
   ImportError: No module named 'streamlit'
   ```
   **Solution**: Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Performance issues**:
   - For large batch processing, consider processing in chunks
   - Monitor memory usage with very long emails
   - Check logs for processing time warnings

### Performance Tips

1. **For better accuracy**:
   - Include email headers when available
   - Provide complete email content
   - Use consistent text encoding

2. **For faster processing**:
   - Pre-clean very long emails
   - Cache the detector instance for multiple predictions
   - Use batch processing for multiple emails

3. **For debugging**:
   - Enable debug logging
   - Check feature extraction results
   - Review individual model predictions

## 🔒 Security Considerations

- All processing is done locally
- No data is sent to external services
- Input sanitization prevents injection attacks
- Error handling prevents system compromise
- Logging provides audit trail

## 📈 Performance Monitoring

Monitor these metrics:
- **Processing time**: Should be <1 second per email
- **Memory usage**: Monitor for memory leaks in batch processing
- **Accuracy**: Compare predictions with known spam/legitimate emails
- **Error rates**: Check logs for prediction failures

## 🆘 Getting Help

1. Check the troubleshooting section above
2. Review logs in `spam_detector.log`
3. Test with simple examples first
4. Verify all dependencies are installed
5. Check model file integrity

---

**Remember**: This system is a tool to assist in spam detection. Always use human judgment for critical decisions, especially in business environments.