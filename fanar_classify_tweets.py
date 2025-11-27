"""
Fanar API Tweet Classification Script

Classifies Arabic tweets for mental health indicators using the Fanar LLM.
Rate limit: 50 requests/minute for Fanar model
Results are saved in a separate column from GPT classifications.
"""

import pandas as pd
import json
import time
import os
from datetime import datetime
import requests
from typing import List, Dict, Optional

# Configuration
FANAR_API_KEY = os.getenv('FANAR_API_KEY')
FANAR_BASE_URL = "https://api.fanar.qa/v1"
MODEL_NAME = "Fanar"  # Options: "Fanar", "Fanar-S-1-7B", "Fanar-C-1-8.7B"
RATE_LIMIT_PER_MINUTE = 50
BATCH_SIZE = 50  # Process at rate limit
DELAY_BETWEEN_BATCHES = 0  # No delay needed - already have 1.2s between requests
SAVE_INTERVAL = 10  # Save progress every N tweets for continuous saving
MAX_RETRIES = 3  # Maximum retries for invalid responses

# Classification categories
CATEGORIES = [
    "depression",
    "anxiety", 
    "suicidal_ideation",
    "neutral"
]

# System prompt for mental health classification
SYSTEM_PROMPT = """أنت خبير في تحليل الصحة النفسية. قم بتصنيف التغريدة العربية التالية إلى واحدة من هذه الفئات:
- depression (اكتئاب): علامات الحزن المستمر، فقدان الأمل، انعدام القيمة الذاتية
- anxiety (قلق): علامات القلق المفرط، التوتر، الخوف، القلق المستمر
- suicidal_ideation (أفكار انتحارية): أي إشارة إلى إيذاء النفس أو الرغبة في الموت
- neutral (محايد): لا توجد علامات واضحة على مشاكل الصحة النفسية

قم بالرد بكلمة واحدة فقط من الفئات المذكورة أعلاه."""


class FanarClassifier:
    """Classifier using Fanar API for mental health classification."""
    
    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.api_key = api_key
        self.model = model
        self.base_url = FANAR_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'rate_limit_hits': 0,
            'invalid_responses': 0,
            'retries': 0,
            'content_filter_errors': 0,
            'start_time': None
        }
        print(api_key)
    
    def _is_valid_classification(self, classification: str) -> bool:
        """Check if classification matches one of the expected categories."""
        return classification in CATEGORIES
    
    def classify_tweet(self, tweet_text: str, retry_count: int = 0) -> Optional[str]:
        """
        Classify a single tweet using Fanar API.
        
        Args:
            tweet_text: The tweet text to classify
            retry_count: Current retry attempt (for internal use)
            
        Returns:
            Classification label or None if failed
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"التغريدة: {tweet_text}"
                }
            ],
           
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_classification = result['choices'][0]['message']['content'].strip().lower()
                
                # Standardize classification
                classification = self._standardize_classification(raw_classification)
                
                # Validate classification
                if self._is_valid_classification(classification):
                    self.stats['successful'] += 1
                    return classification
                else:
                    # Invalid response - retry if possible
                    self.stats['invalid_responses'] += 1
                    if retry_count < MAX_RETRIES:
                        self.stats['retries'] += 1
                        print(f"⚠️ Invalid response '{raw_classification}' (attempt {retry_count + 1}/{MAX_RETRIES}). Retrying...")
                        time.sleep(2)  # Brief pause before retry
                        return self.classify_tweet(tweet_text, retry_count + 1)
                    else:
                        print(f"❌ Max retries reached. Invalid response: '{raw_classification}'")
                        self.stats['failed'] += 1
                        return None
                
            elif response.status_code == 429:
                print(f"⚠️ Rate limit hit. Waiting 65 seconds...")
                self.stats['rate_limit_hits'] += 1
                time.sleep(65)
                return self.classify_tweet(tweet_text, retry_count)  # Retry without incrementing retry_count
                
            elif response.status_code == 400:
                # Check if it's a content filter error
                try:
                    error_data = response.json()
                    if error_data.get('error', {}).get('code') == 'content_filter':
                        # Content filter error - try once more
                        if retry_count == 0:
                            print(f"⚠️ Content filter triggered. Retrying once...")
                            time.sleep(2)
                            return self.classify_tweet(tweet_text, retry_count + 1)
                        else:
                            # Second attempt also filtered - mark for manual review
                            print(f"⚠️ Content filter error (2x) - marking for manual review")
                            self.stats['content_filter_errors'] += 1
                            return 'CONTENT_FILTER_ERROR'
                except:
                    pass
                print(f"❌ Error {response.status_code}: {response.text}")
                self.stats['failed'] += 1
                return None
                
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                self.stats['failed'] += 1
                return None
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            self.stats['failed'] += 1
            return None
    
    def _standardize_classification(self, classification: str) -> str:
        """Standardize classification to one of the four categories."""
        classification = classification.lower().strip()
        
        # Map variations to standard categories
        if 'depression' in classification or 'اكتئاب' in classification or 'حزن' in classification:
            return 'depression'
        elif 'anxiety' in classification or 'قلق' in classification or 'توتر' in classification:
            return 'anxiety'
        elif 'suicidal' in classification or 'انتحار' in classification or 'إيذاء' in classification:
            return 'suicidal_ideation'
        else:
            return 'neutral'
    
    def classify_batch(self, tweets: List[str], batch_num: int) -> List[Optional[str]]:
        """
        Classify a batch of tweets with rate limiting.
        
        Args:
            tweets: List of tweet texts
            batch_num: Batch number for logging
            
        Returns:
            List of classifications
        """
        classifications = []
        batch_start = time.time()
        
        print(f"\n📦 Processing Batch {batch_num} ({len(tweets)} tweets)")
        print("=" * 70)
        
        for i, tweet in enumerate(tweets, 1):
            classification = self.classify_tweet(tweet)
            classifications.append(classification)
            self.stats['total_processed'] += 1
            
            # Progress indicator
            if i % 10 == 0:
                print(f"  Processed {i}/{len(tweets)} tweets...")
        
        batch_duration = time.time() - batch_start
        print(f"\n✓ Batch {batch_num} completed in {batch_duration:.1f}s")
        
        return classifications
    
    def print_stats(self):
        """Print classification statistics."""
        if self.stats['start_time']:
            duration = time.time() - self.stats['start_time']
            duration_str = f"{duration/60:.1f} minutes"
        else:
            duration_str = "N/A"
        
        print("\n" + "=" * 70)
        print("FANAR CLASSIFICATION STATISTICS")
        print("=" * 70)
        print(f"Total Processed:    {self.stats['total_processed']:,}")
        print(f"Successful:         {self.stats['successful']:,}")
        print(f"Failed:             {self.stats['failed']:,}")
        print(f"Invalid Responses:  {self.stats['invalid_responses']:,}")
        print(f"Content Filters:    {self.stats['content_filter_errors']:,}")
        print(f"Retries:            {self.stats['retries']:,}")
        print(f"Rate Limit Hits:    {self.stats['rate_limit_hits']}")
        print(f"Duration:           {duration_str}")
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            print(f"Success Rate:       {success_rate:.2f}%")
        print("=" * 70)


def test_fanar_api(api_key: str, num_samples: int = 3) -> bool:
    """
    Test the Fanar API with a few sample tweets.
    
    Args:
        api_key: Fanar API key
        num_samples: Number of test samples
        
    Returns:
        True if test successful, False otherwise
    """
    print("\n" + "=" * 70)
    print("TESTING FANAR API")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv('datasets/arabic_tweets_classified.csv')
    test_samples = df.sample(num_samples)['text'].tolist()
    
    classifier = FanarClassifier(api_key)
    
    print(f"\nTesting with {num_samples} random tweets:\n")
    
    for i, tweet in enumerate(test_samples, 1):
        print(f"\n{i}. Tweet: {tweet[:80]}..." if len(tweet) > 80 else f"\n{i}. Tweet: {tweet}")
        classification = classifier.classify_tweet(tweet)
        print(f"   Classification: {classification}")
        time.sleep(2)  # Be respectful during testing
    
    classifier.print_stats()
    
    if classifier.stats['successful'] == num_samples:
        print("\n✅ API test PASSED - All samples classified successfully!")
        return True
    else:
        print(f"\n⚠️ API test PARTIAL - {classifier.stats['successful']}/{num_samples} successful")
        return classifier.stats['successful'] > 0


def classify_all_tweets(api_key: str, input_csv: str, output_csv: str, test_first: bool = True, save_interval: int = SAVE_INTERVAL):
    """
    Classify all tweets in the dataset using Fanar API.
    Saves progress continuously and can resume if interrupted.
    
    Args:
        api_key: Fanar API key
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        test_first: Whether to run API test first
        save_interval: Save progress every N tweets
    """
    # Test API first
    if test_first:
        print("\n🧪 Running API test first...")
        if not test_fanar_api(api_key, num_samples=3):
            print("\n❌ API test failed. Please check your API key and try again.")
            return
        
        print("\n" + "=" * 70)
        input("\nPress Enter to continue with full classification, or Ctrl+C to cancel...")
        print("=" * 70)
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df):,} tweets")
    
    # Check if Fanar column already exists
    if 'fanar_classification' in df.columns:
        print(f"\n⚠️ 'fanar_classification' column already exists")
        existing_count = df['fanar_classification'].notna().sum()
        print(f"   {existing_count:,} tweets already classified")
        
        choice = input("\nOptions:\n  1. Continue from where we left off\n  2. Start fresh (overwrite)\n  3. Cancel\nChoice (1/2/3): ")
        
        if choice == '3':
            print("❌ Cancelled by user")
            return
        elif choice == '2':
            df['fanar_classification'] = None
            print("✓ Starting fresh")
        else:
            print("✓ Continuing from existing progress")
    else:
        df['fanar_classification'] = None
    
    # Get tweets that need classification
    unclassified_mask = df['fanar_classification'].isna()
    unclassified_indices = df[unclassified_mask].index.tolist()
    
    if len(unclassified_indices) == 0:
        print("\n✅ All tweets already classified!")
        return
    
    print(f"\n📊 Tweets to classify: {len(unclassified_indices):,}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Estimated batches: {(len(unclassified_indices) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"   Rate limit: {RATE_LIMIT_PER_MINUTE} requests/minute")
    
    # Initialize classifier
    classifier = FanarClassifier(api_key)
    classifier.stats['start_time'] = time.time()
    
    # Process tweets one by one with continuous saving
    print(f"\n🚀 Starting classification (saving every {save_interval} tweets)...")
    
    last_save_time = time.time()
    tweets_since_save = 0
    
    for i, idx in enumerate(unclassified_indices, 1):
        tweet_text = df.at[idx, 'text']
        
        # Classify tweet
        classification = classifier.classify_tweet(tweet_text)
        df.at[idx, 'fanar_classification'] = classification
        classifier.stats['total_processed'] += 1
        
        tweets_since_save += 1
        
        # Progress indicator
        if i % 10 == 0:
            elapsed = time.time() - classifier.stats['start_time']
            rate = i / elapsed if elapsed > 0 else 0
            remaining = len(unclassified_indices) - i
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_hours = eta_seconds / 3600
            print(f"  Progress: {i:,}/{len(unclassified_indices):,} ({i/len(unclassified_indices)*100:.1f}%) | Rate: {rate*60:.1f}/min | ETA: {eta_hours:.1f}h")
        
        # Save progress continuously
        if tweets_since_save >= save_interval:
            df.to_csv(output_csv, index=False)
            save_time = time.time() - last_save_time
            print(f"    💾 Auto-saved {tweets_since_save} tweets ({save_time:.1f}s)")
            last_save_time = time.time()
            tweets_since_save = 0
    
    # Final save
    if tweets_since_save > 0:
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Final save: {tweets_since_save} tweets saved")
    
    # Final statistics
    classifier.print_stats()
    
    # Classification distribution
    print("\n" + "=" * 70)
    print("FANAR CLASSIFICATION DISTRIBUTION")
    print("=" * 70)
    print(df['fanar_classification'].value_counts())
    print(f"\nTotal classified: {df['fanar_classification'].notna().sum():,}")
    
    # Compare with GPT classifications if available
    if 'mental_health_classification' in df.columns:
        print("\n" + "=" * 70)
        print("COMPARISON: FANAR vs GPT CLASSIFICATIONS")
        print("=" * 70)
        
        both_classified = df[df['fanar_classification'].notna() & df['mental_health_classification'].notna()]
        agreement = (both_classified['fanar_classification'] == both_classified['mental_health_classification']).sum()
        total = len(both_classified)
        
        if total > 0:
            agreement_rate = (agreement / total) * 100
            print(f"Agreement: {agreement:,} / {total:,} ({agreement_rate:.2f}%)")
            
            # Cross-tabulation
            print("\nCross-tabulation (rows=Fanar, cols=GPT):")
            ct = pd.crosstab(
                both_classified['fanar_classification'],
                both_classified['mental_health_classification'],
                margins=True
            )
            print(ct)
    
    print(f"\n✅ All done! Results saved to: {output_csv}")


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("FANAR API TWEET CLASSIFICATION")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv('FANAR_API_KEY')
    if not api_key or api_key == 'YOUR_FANAR_API_KEY_HERE':
        print("\n❌ Fanar API key not found!")
        print("\nPlease set your API key:")
        print("  export FANAR_API_KEY='your-api-key-here'")
        print("\nOr edit the script to include your key directly.")
        return
    
    # File paths
    input_csv = 'datasets/arabic_tweets_classified.csv'
    output_csv = 'datasets/arabic_tweets_classified.csv'  # Save to same file
    
    if not os.path.exists(input_csv):
        print(f"\n❌ Input file not found: {input_csv}")
        return
    
    print(f"\nAPI Key: {api_key[:10]}..." + "*" * 20)
    print(f"Model: {MODEL_NAME}")
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_csv}")
    
    # Start classification
    classify_all_tweets(api_key, input_csv, output_csv, test_first=True)


if __name__ == "__main__":
    main()
