# Labeling Scripts Documentation

This folder contains scripts for automated labeling and classification of Arabic tweets for mental health analysis. The pipeline uses multiple Large Language Models (LLMs) to classify tweets into four categories: **depression**, **anxiety**, **suicidal ideation**, and **neutral**.

---

## 📁 Scripts Overview

### 1. `batch_classify_tweets.py`

**Purpose**: Batch classification of Arabic tweets using OpenAI's GPT-4o-mini via the Batch API.

**Key Features**:

- Cost-efficient batch processing (50% discount vs real-time API)
- Processes 5,000 tweets per batch
- Automatic resume capability with state management
- Rate-aware processing with concurrent batch management
- Robust error handling and retry mechanisms

**Configuration**:

```python
MODEL = "gpt-4o-mini"              # Cost: $0.15 per 1M input tokens
BATCH_SIZE = 5000                  # Tweets per batch
MAX_CONCURRENT_BATCHES = 2         # Parallel processing limit
CHECK_INTERVAL = 30                # Status check frequency (seconds)
```

**Input**: `datasets/arabic_tweets_cleaned.csv`  
**Output**: `datasets/arabic_tweets_classified.csv` (with `mental_health_classification` column)

**Pseudo Code**:

```
FUNCTION classify_tweets_with_gpt():
    LOAD cleaned_tweets FROM csv

    FOR EACH batch OF 5000 tweets:
        CREATE batch_request_file

        FOR EACH tweet IN batch:
            APPEND classification_request TO batch_file
            REQUEST = {
                model: "gpt-4o-mini"
                prompt: SYSTEM_PROMPT + tweet_text
                response_format: JSON {classification: category}
            }

        UPLOAD batch_file TO OpenAI
        SUBMIT batch_job

        WHILE job_status != "completed":
            WAIT 30 seconds
            CHECK job_status

        DOWNLOAD results
        PARSE classifications
        MERGE with original_data

    SAVE final_results TO csv
END FUNCTION
```

---

### 2. `fanar_classify_tweets.py`

**Purpose**: Classification using Fanar LLM (Arabic-specialized model) for comparison and validation.

**Key Features**:

- Arabic-optimized model (Fanar by Qatar Computing Research Institute)
- Rate-limited processing (50 requests/minute)
- Real-time API classification
- Continuous progress saving
- Separate classification column for comparison

**Configuration**:

```python
MODEL_NAME = "Fanar"               # Arabic-specialized LLM
RATE_LIMIT_PER_MINUTE = 50        # API constraint
BATCH_SIZE = 50                    # Requests per batch
SAVE_INTERVAL = 10                 # Progress save frequency
```

**Input**: `datasets/arabic_tweets_classified.csv` (with GPT classifications)  
**Output**: Same file updated with `fanar_classification` column

**Pseudo Code**:

```
FUNCTION classify_tweets_with_fanar():
    LOAD tweets_with_gpt_labels FROM csv
    INIT rate_limiter (50 requests/minute)

    FOR EACH tweet IN tweets:
        IF already_classified:
            SKIP tweet
            CONTINUE

        WAIT for_rate_limit (1.2 seconds minimum)

        REQUEST = {
            model: "Fanar"
            messages: [
                {role: "system", content: CLASSIFICATION_PROMPT}
                {role: "user", content: tweet_text}
            ]
            temperature: 0.3
        }

        response = CALL fanar_api(REQUEST)

        IF response is valid_json:
            classification = PARSE response
            ADD classification TO tweet
        ELSE:
            RETRY up_to 3_times
            IF still_fails:
                MARK as "error"

        IF processed_count % 10 == 0:
            SAVE progress TO csv

    SAVE final_results TO csv
END FUNCTION
```

---

### 3. `compare_classifications.py`

**Purpose**: Compare and merge classifications from multiple sources (GPT and Fanar) into a single consensus label.

**Key Features**:

- Configurable comparison strategies
- Option to prefer specific source
- Option to require agreement between sources
- Detailed statistics and disagreement analysis
- Removes redundant classification columns

**Configuration**:

```python
CONFIG = {
    'prefer_source': 'fanar',      # Options: 'gpt', 'fanar'
    'require_match': False         # True = only keep when both agree
}
```

**Behavior Modes**:

#### Mode 1: Prefer Source (require_match = False)

- If preferred source has value → Use it
- If preferred source empty → Use alternative source
- Ignores disagreements

#### Mode 2: Require Agreement (require_match = True)

- Both sources agree → Keep classification
- Both sources disagree → Leave empty
- Only one source has value → Use that value

**Input**: `datasets/arabic_tweets_classified.csv` (with both `mental_health_classification` and `fanar_classification`)  
**Output**: `datasets/arabic_tweets_matched_classifications.csv` (with single `classification` column)

**Pseudo Code**:

```
FUNCTION compare_and_merge_classifications():
    LOAD tweets WITH gpt_label AND fanar_label

    CONFIG = {
        prefer_source: "fanar"
        require_match: False
    }

    FOR EACH tweet IN tweets:
        gpt_empty = IS_EMPTY(tweet.gpt_classification)
        fanar_empty = IS_EMPTY(tweet.fanar_classification)

        IF require_match == True:
            IF NOT gpt_empty AND NOT fanar_empty:
                IF gpt_classification == fanar_classification:
                    final_classification = gpt_classification
                ELSE:
                    final_classification = ""  // Disagreement
            ELSE IF NOT fanar_empty:
                final_classification = fanar_classification
            ELSE IF NOT gpt_empty:
                final_classification = gpt_classification
            ELSE:
                final_classification = ""

        ELSE:  // Prefer source mode
            IF prefer_source == "fanar":
                IF NOT fanar_empty:
                    final_classification = fanar_classification
                ELSE:
                    final_classification = gpt_classification
            ELSE:  // prefer_source == "gpt"
                IF NOT gpt_empty:
                    final_classification = gpt_classification
                ELSE:
                    final_classification = fanar_classification

        tweet.classification = final_classification

    REMOVE gpt_classification_column
    REMOVE fanar_classification_column

    CALCULATE statistics:
        - Total agreements
        - Total disagreements
        - Source usage breakdown
        - Final label distribution

    SAVE merged_results TO csv
    PRINT detailed_statistics
END FUNCTION
```

---

## 🔄 Complete Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    LABELING PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

Step 1: GPT-4o-mini Classification (Batch API)
┌──────────────────────────────────────┐
│  batch_classify_tweets.py            │
│  Input:  arabic_tweets_cleaned.csv   │
│  Output: arabic_tweets_classified.csv│
│          + mental_health_classification
└──────────────────────────────────────┘
                    ↓
Step 2: Fanar LLM Classification (Real-time API)
┌──────────────────────────────────────┐
│  fanar_classify_tweets.py            │
│  Input:  arabic_tweets_classified.csv│
│  Output: arabic_tweets_classified.csv│
│          + fanar_classification      │
└──────────────────────────────────────┘
                    ↓
Step 3: Compare & Merge Classifications
┌──────────────────────────────────────┐
│  compare_classifications.py          │
│  Input:  arabic_tweets_classified.csv│
│  Output: matched_classifications.csv │
│          Single 'classification'     │
└──────────────────────────────────────┘
```

---

## 📊 Classification Categories

All scripts use the following standardized categories:

1. **depression**: Indicators of depressive symptoms

   - Persistent sadness, hopelessness
   - Loss of interest in activities
   - Feelings of worthlessness

2. **anxiety**: Symptoms of anxiety disorders

   - Excessive worry or fear
   - Panic, restlessness
   - Physical symptoms (racing heart, etc.)

3. **suicidal_ideation**: Thoughts about self-harm or suicide

   - Direct or indirect mentions of suicide
   - Expressions of wanting to die
   - Planning or intent to harm oneself

4. **neutral**: No mental health indicators
   - General conversation
   - Daily activities
   - Positive emotions

---

## 🚀 Usage Instructions

### Running the Complete Pipeline

```bash
# Step 1: Classify with GPT (Batch API)
python labeling/batch_classify_tweets.py

# Step 2: Classify with Fanar (Real-time)
export FANAR_API_KEY="your-api-key"
python labeling/fanar_classify_tweets.py

# Step 3: Compare and merge
# Edit CONFIG in compare_classifications.py first
python labeling/compare_classifications.py
```

### Configuration Options

#### For `batch_classify_tweets.py`:

- Modify `INPUT_CSV` and `OUTPUT_CSV` paths
- Adjust `BATCH_SIZE` (default: 5000)
- Change `MODEL` if needed (e.g., gpt-4o-mini, gpt-4o)

#### For `fanar_classify_tweets.py`:

- Set `FANAR_API_KEY` environment variable
- Modify `MODEL_NAME` for different Fanar variants
- Adjust `RATE_LIMIT_PER_MINUTE` based on your quota

#### For `compare_classifications.py`:

```python
# Edit CONFIG dictionary at the top of the script
CONFIG = {
    'prefer_source': 'fanar',  # 'gpt' or 'fanar'
    'require_match': False      # True or False
}
```

---

## 📈 Output Statistics

Each script provides detailed statistics:

### Batch Classification

- Total tweets processed
- Batches created and completed
- Processing time per batch
- Success/error rates
- Cost estimation

### Fanar Classification

- Total requests made
- Success rate
- Error rate by type
- Average response time
- Rate limiting adherence

### Comparison

- Total agreements between sources
- Total disagreements
- Source preference breakdown
- Final label distribution
- Confidence metrics

---

## 🔧 Troubleshooting

### Common Issues

1. **Batch API timeout**

   - Solution: Reduce `BATCH_SIZE` or increase `CHECK_INTERVAL`

2. **Fanar rate limit exceeded**

   - Solution: Ensure `RATE_LIMIT_PER_MINUTE` is correctly set
   - The script automatically adds 1.2s delay between requests

3. **High disagreement rate**

   - Solution: Review classification prompts in both scripts
   - Consider using `require_match: True` for higher precision

4. **Out of memory**
   - Solution: Process data in smaller chunks
   - Reduce `BATCH_SIZE` in batch classification

---

## 💡 Best Practices

1. **Always backup data** before running scripts
2. **Monitor progress** using the printed statistics
3. **Review disagreements** manually for quality assessment
4. **Use batch API** for large datasets (cost-effective)
5. **Validate results** by comparing both sources
6. **Save intermediate results** frequently

---

## 📝 File Formats

### Input Format (Cleaned Tweets)

```csv
text,char_count,word_count,mention_count,...
"Tweet text here",42,9,0,...
```

### Intermediate Format (After GPT/Fanar)

```csv
text,...,mental_health_classification,fanar_classification
"Tweet text",42,depression,depression
```

### Final Format (After Comparison)

```csv
text,...,classification
"Tweet text",42,depression
```

---

## 🎯 Performance Metrics

### GPT-4o-mini Batch API

- **Cost**: ~$0.075 per 1M input tokens (50% batch discount)
- **Latency**: 24 hours maximum (usually 2-4 hours)
- **Throughput**: Limitted batch size based on account history
- **Recommended**: Large datasets (>10K tweets)

### Fanar Real-time API

- **Cost**: Free (requires approval for API Key)
- **Latency**: ~1-2 seconds per request
- **Throughput**: 50 requests/minute
- **Recommended**: Validation and comparison

### Comparison Script

- **Speed**: Local processing (milliseconds per row)
- **Memory**: Low (processes line by line)
- **Recommended**: Always run after both classifications

---

## 📚 Additional Resources

- [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch)
- [Fanar LLM Documentation](https://api.fanar.qa/docs)
