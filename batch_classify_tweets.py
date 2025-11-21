"""
Batch Classification Script for Arabic Tweets
Classifies tweets into: depression, anxiety, suicidal ideation, or neutral
Uses OpenAI's Batch API for cost-efficient processing
"""

import json
import os
import time
import pandas as pd
from openai import OpenAI
from pathlib import Path
from datetime import datetime

# Configuration
INPUT_CSV = "datasets/arabic_tweets_cleaned.csv"
OUTPUT_CSV = "datasets/arabic_tweets_classified.csv"
BATCH_INPUT_JSONL = "batches/batch_input_{batch_num}.jsonl"
BATCH_OUTPUT_JSONL = "batches/batch_output_{batch_num}.jsonl"
STATE_FILE = "batches/batch_state.json"
BATCH_SIZE = 5000  # Process in smaller batches
MODEL = "gpt-4o-mini"  # Cost-effective model for classification (0.15$ per 1M input tokens)

# System prompt for classification
SYSTEM_PROMPT = """You are an expert mental health classifier for Arabic text. 
Classify the following tweet into exactly ONE of these categories:
- depression: Indicates symptoms of depression, sadness, hopelessness, or lack of motivation
- anxiety: Shows signs of worry, fear, stress, or anxious thoughts
- suicidal_ideation: Contains references to self-harm, suicide, or wanting to end one's life
- neutral: Does not fit into any of the above categories

Respond with ONLY the category name, nothing else."""


class BatchClassifier:
    def __init__(self):
        self.client = OpenAI()
        self.state = self.load_state()
        
    def load_state(self):
        """Load previous state if exists"""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {
            "batches": [],  # List of batch info: {batch_num, batch_id, status, start_idx, end_idx, output_file_id}
            "current_batch_num": 0,
            "total_tweets": 0,
            "completed_batches": [],
            "status": "not_started"
        }
    
    def save_state(self):
        """Save current state for recovery"""
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"State saved to {STATE_FILE}")
    
    def create_batch_input(self, tweets_df, start_idx, end_idx, batch_num):
        """Create JSONL file for batch processing"""
        print(f"\n[Batch {batch_num}] Creating input for tweets {start_idx} to {end_idx-1}...")
        
        batch_file = BATCH_INPUT_JSONL.format(batch_num=batch_num)
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            for idx in range(start_idx, end_idx):
                tweet = tweets_df.iloc[idx]['text']
                request = {
                    "custom_id": f"request-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": tweet}
                        ],
                        
                    }
                }
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        
        print(f"[Batch {batch_num}] Created {batch_file} with {end_idx - start_idx} requests")
        return batch_file
    
    def upload_batch_file(self, batch_file, batch_num):
        """Upload the batch input file to OpenAI"""
        print(f"[Batch {batch_num}] Uploading batch input file...")
        try:
            with open(batch_file, 'rb') as f:
                batch_input_file = self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            print(f"[Batch {batch_num}] File uploaded: {batch_input_file.id}")
            return batch_input_file.id
        except Exception as e:
            print(f"[Batch {batch_num}] Error uploading file: {e}")
            raise
    
    def create_batch(self, input_file_id, batch_num):
        """Create a batch job"""
        print(f"[Batch {batch_num}] Creating batch job...")
        try:
            batch = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": f"Arabic tweets classification - Batch {batch_num}",
                    "batch_number": str(batch_num),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            print(f"[Batch {batch_num}] Batch created: {batch.id}")
            print(f"[Batch {batch_num}] Status: {batch.status}")
            return batch.id
        except Exception as e:
            print(f"[Batch {batch_num}] Error creating batch: {e}")
            raise
    
    def check_batch_status(self, batch_id, batch_num):
        """Check the status of a batch"""
        try:
            batch = self.client.batches.retrieve(batch_id)
            return batch
        except Exception as e:
            print(f"[Batch {batch_num}] Error checking status: {e}")
            raise
    
    def wait_for_completion(self, batch_id, batch_num, check_interval=60):
        """Wait for batch to complete, checking periodically"""
        print(f"[Batch {batch_num}] Waiting for completion...")
        print(f"[Batch {batch_num}] Checking status every {check_interval} seconds...")
        
        while True:
            batch = self.check_batch_status(batch_id, batch_num)
            status = batch.status
            
            print(f"[Batch {batch_num}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}")
            
            if status == "completed":
                print(f"[Batch {batch_num}] Completed successfully!")
                return batch
            elif status == "failed":
                print(f"[Batch {batch_num}] Failed with errors: {batch.errors}")
                raise Exception(f"Batch {batch_num} failed: {batch.errors}")
            elif status == "expired":
                print(f"[Batch {batch_num}] Expired before completion")
                raise Exception(f"Batch {batch_num} expired")
            elif status == "cancelled":
                print(f"[Batch {batch_num}] Was cancelled")
                raise Exception(f"Batch {batch_num} cancelled")
            elif status in ["validating", "in_progress", "finalizing"]:
                # Show progress if available
                if hasattr(batch, 'request_counts'):
                    counts = batch.request_counts
                    print(f"[Batch {batch_num}]   Progress: {counts.completed}/{counts.total} completed, {counts.failed} failed")
                time.sleep(check_interval)
            else:
                print(f"[Batch {batch_num}] Unknown status: {status}")
                time.sleep(check_interval)
    
    def download_results(self, output_file_id, batch_num):
        """Download and save the batch results"""
        print(f"[Batch {batch_num}] Downloading results from {output_file_id}...")
        try:
            file_response = self.client.files.content(output_file_id)
            content = file_response.text
            
            output_file = BATCH_OUTPUT_JSONL.format(batch_num=batch_num)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"[Batch {batch_num}] Results saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"[Batch {batch_num}] Error downloading results: {e}")
            raise
    
    def parse_results(self, tweets_df, output_files):
        """Parse batch results from multiple output files and create classified CSV"""
        print("\nParsing results from all batches...")
        
        # Create a dictionary to store classifications
        classifications = {}
        
        for output_file in output_files:
            print(f"Parsing {output_file}...")
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line)
                    custom_id = result.get('custom_id')
                    
                    if custom_id:
                        # Extract index from custom_id (format: "request-{idx}")
                        idx = int(custom_id.split('-')[1])
                        
                        # Check for errors
                        if result.get('error'):
                            print(f"Error in {custom_id}: {result['error']}")
                            classifications[idx] = "error"
                        else:
                            # Extract classification from response
                            response = result.get('response', {})
                            body = response.get('body', {})
                            choices = body.get('choices', [])
                            
                            if choices:
                                content = choices[0].get('message', {}).get('content', '').strip().lower()
                                # Validate classification
                                valid_classes = ['depression', 'anxiety', 'suicidal_ideation', 'neutral']
                                if content in valid_classes:
                                    classifications[idx] = content
                                else:
                                    print(f"Invalid classification '{content}' for {custom_id}, marking as neutral")
                                    classifications[idx] = "neutral"
                            else:
                                print(f"No choices in response for {custom_id}")
                                classifications[idx] = "error"
        
        print(f"Parsed {len(classifications)} classifications")
        
        # Create results dataframe
        results = []
        for idx in range(len(tweets_df)):
            tweet = tweets_df.iloc[idx]['text']
            classification = classifications.get(idx, "not_processed")
            results.append({
                'tweet': tweet,
                'classification': classification
            })
        
        results_df = pd.DataFrame(results)
        
        # Save to CSV
        results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"\nResults saved to {OUTPUT_CSV}")
        
        # Print summary statistics
        print("\nClassification Summary:")
        print(results_df['classification'].value_counts())
        
        return results_df
    
    def run(self):
        """Main execution flow"""
        print("=" * 60)
        print("Arabic Tweets Batch Classification")
        print("=" * 60)
        
        # Load tweets
        if not os.path.exists(INPUT_CSV):
            raise FileNotFoundError(f"Input file {INPUT_CSV} not found!")
        
        print(f"\nLoading tweets from {INPUT_CSV}...")
        tweets_df = pd.read_csv(INPUT_CSV)
        total_tweets = len(tweets_df)
        print(f"Loaded {total_tweets} tweets")
        
        # Calculate number of batches needed
        num_batches = (total_tweets + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nProcessing in {num_batches} batches of up to {BATCH_SIZE} tweets each")
        
        # Initialize state if starting fresh
        if self.state["status"] == "not_started":
            self.state["total_tweets"] = total_tweets
            self.state["batches"] = []
            for i in range(num_batches):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, total_tweets)
                self.state["batches"].append({
                    "batch_num": i + 1,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "batch_id": None,
                    "status": "pending",
                    "output_file_id": None
                })
            self.state["status"] = "in_progress"
            self.save_state()
        
        # Process each batch
        output_files = []
        for batch_info in self.state["batches"]:
            batch_num = batch_info["batch_num"]
            start_idx = batch_info["start_idx"]
            end_idx = batch_info["end_idx"]
            
            print("\n" + "=" * 60)
            print(f"Processing Batch {batch_num}/{num_batches}")
            print(f"Tweets {start_idx} to {end_idx-1} ({end_idx - start_idx} tweets)")
            print("=" * 60)
            
            # Skip if already completed
            if batch_info["status"] == "completed" and batch_info["output_file_id"]:
                print(f"[Batch {batch_num}] Already completed, downloading results...")
                output_file = self.download_results(batch_info["output_file_id"], batch_num)
                output_files.append(output_file)
                continue
            
            # Resume if batch exists but not completed
            if batch_info["batch_id"] and batch_info["status"] in ["validating", "in_progress", "finalizing"]:
                print(f"[Batch {batch_num}] Resuming existing batch: {batch_info['batch_id']}")
                batch = self.wait_for_completion(batch_info["batch_id"], batch_num)
                batch_info["status"] = "completed"
                batch_info["output_file_id"] = batch.output_file_id
                self.save_state()
            else:
                # Create new batch
                batch_file = self.create_batch_input(tweets_df, start_idx, end_idx, batch_num)
                input_file_id = self.upload_batch_file(batch_file, batch_num)
                batch_id = self.create_batch(input_file_id, batch_num)
                
                batch_info["batch_id"] = batch_id
                batch_info["status"] = "in_progress"
                self.save_state()
                
                batch = self.wait_for_completion(batch_id, batch_num)
                batch_info["status"] = "completed"
                batch_info["output_file_id"] = batch.output_file_id
                self.save_state()
            
            # Download results
            output_file = self.download_results(batch_info["output_file_id"], batch_num)
            output_files.append(output_file)
            
            print(f"[Batch {batch_num}] Complete!")
        
        # Parse all results
        print("\n" + "=" * 60)
        print("All batches completed! Parsing results...")
        print("=" * 60)
        results_df = self.parse_results(tweets_df, output_files)
        
        # Update final state
        self.state["status"] = "completed"
        self.save_state()
        
        print("\n" + "=" * 60)
        print("Classification completed successfully!")
        print("=" * 60)
        
        return results_df


def main():
    """Entry point"""
    try:
        classifier = BatchClassifier()
        results = classifier.run()
        print("\n" + "=" * 60)
        print("Classification completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        print("State has been saved. Run the script again to resume.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("State has been saved. Fix the issue and run the script again to resume.")
        raise


if __name__ == "__main__":
    main()
