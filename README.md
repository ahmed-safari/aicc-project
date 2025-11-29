# Arabic Mental Health Detection System

A comprehensive machine learning system for detecting mental health signals in Arabic social media text, specifically designed to identify depression, anxiety, and suicidal ideation from Arabic tweets.

## 📋 Project Overview

This research project develops a robust multi-class classification system to detect mental health signals in Arabic text using both traditional machine learning and state-of-the-art transformer models. The system addresses critical challenges including severe class imbalance, Arabic-specific linguistic features, and clinical validity requirements for mental health applications.

### Key Features

- **Advanced Arabic NLP**: Custom preprocessing pipeline for Arabic text normalization and feature extraction
- **Multiple ML Approaches**: Traditional ML (Random Forest, XGBoost, LightGBM, etc.) and transformer models (AraBERT)
- **Class Imbalance Handling**: SMOTE, ADASYN, and other resampling techniques
- **Checkpoint System**: Resume interrupted training for long-running models
- **Clinical Metrics**: Sensitivity, specificity, and risk stratification for mental health assessment
- **Model Interpretability**: Feature importance and error analysis

## 🗂️ Project Structure

```
aicc-project/
├── README.md                          # This file - main project documentation
├── requirements.txt                    # Python dependencies
│
├── notebooks/                          # Research notebooks and documentation
│   ├── arabic_mental_health_research_notebook.ipynb    # Main analysis notebook
│   ├── tweets_analysis.ipynb          # Tweet analysis and EDA
│   ├── ml_notebook_old_notbalanced.ipynb  # Legacy experiments
│   └── RESEARCH_NOTES.md              # Detailed research documentation
│
├── datasets/                           # Data files
│   ├── arabic_tweets_50k.csv          # Raw tweet data
│   ├── arabic_tweets_cleaned.csv      # Preprocessed tweets
│   ├── arabic_tweets_classified.csv   # Labeled data
│   └── arabic_tweets_matched_classifications.csv  # Final dataset
│
├── models/                             # Saved models and checkpoints
│   └── training_progress/             # Training checkpoints
│       ├── ml_models_checkpoint.pkl   # Traditional ML checkpoints
│       └── arabert_checkpoint.pt      # AraBERT training checkpoints
│
├── labeling/                           # Data labeling scripts
│   ├── batch_classify_tweets.py       # Batch classification
│   ├── fanar_classify_tweets.py       # Fanar API classification
│   ├── compare_classifications.py     # Compare labeling approaches
│   └── README.md                      # Labeling documentation
│
├── scrapers/                           # Data collection scripts
│   ├── x/                             # Twitter/X scraping
│   │   └── script.py
│   ├── x-lib/                         # Library-based scraping
│   └── instagram/                     # Instagram scraping
│
└── batches/                            # Batch processing files
    ├── batch_input_*.jsonl            # Input files for batch processing
    ├── batch_output_*.jsonl           # Output from batch processing
    └── batch_state.json               # Batch processing state
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for transformer models)
- 16GB+ RAM recommended

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ahmed-safari/aicc-project.git
   cd aicc-project
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install emoji package** (special version required)
   ```bash
   pip install emoji==1.7.0
   ```

### Running the Project

#### 1. Main Analysis Notebook

Open and run the main research notebook:

```bash
jupyter notebook notebooks/arabic_mental_health_research_notebook.ipynb
```

This notebook includes:

- Data loading and exploration
- Arabic text preprocessing
- Feature engineering
- Model training with checkpoints
- Evaluation and interpretation

#### 2. Resume Training from Checkpoints

If training is interrupted, simply re-run the training cells. The system will prompt:

```
Found checkpoint at models/training_progress/ml_models_checkpoint.pkl
Resume from checkpoint? (y/n):
```

Type `y` to resume from where you left off.

#### 3. Data Collection

To scrape new data:

```bash
cd scrapers/x
python script.py
```

#### 4. Data Labeling

The project includes a sophisticated automated labeling pipeline using multiple LLMs. See the full pipeline below.

## 🏷️ Data Labeling Pipeline

The labeling pipeline uses a multi-stage approach combining multiple Large Language Models (LLMs) to classify Arabic tweets into mental health categories. This approach ensures quality and allows for validation across different models.

### Labeling Scripts

#### 1. **batch_classify_tweets.py** - Primary Batch Classification

The main labeling script using OpenAI's GPT-4o-mini via the Batch API for cost-efficient processing.

**Key Features:**

- **Cost-Efficient**: 50% discount using Batch API ($0.15 per 1M input tokens)
- **Batch Processing**: Processes 5,000 tweets per batch
- **State Management**: Automatic resume capability if interrupted
- **Concurrent Processing**: Handles up to 2 batches simultaneously
- **Robust Error Handling**: Retry mechanisms and status tracking

**Usage:**

```bash
cd labeling
python batch_classify_tweets.py
```

**Process:**

1. Loads cleaned tweets from `datasets/arabic_tweets_cleaned.csv`
2. Splits into batches of 5,000 tweets
3. Creates JSONL batch files with classification requests
4. Submits batches to OpenAI Batch API
5. Monitors batch status (checks every 30 seconds)
6. Downloads and parses results
7. Saves to `datasets/arabic_tweets_classified.csv`

**Classification Prompt:**

```
You are an expert mental health classifier for Arabic text.
Classify into: depression, anxiety, suicidal_ideation, or neutral
```

#### 2. **fanar_classify_tweets.py** - Arabic-Specialized Classification

Uses Fanar LLM (Qatar Computing Research Institute) for validation with Arabic-optimized model.

**Key Features:**

- **Arabic-Specialized**: Built specifically for Arabic language understanding
- **Real-Time API**: Immediate classification results
- **Rate-Limited**: 50 requests/minute with automatic throttling
- **Progressive Saving**: Saves progress every 10 classifications
- **Separate Column**: Adds `fanar_classification` for comparison

**Usage:**

```bash
cd labeling
python fanar_classify_tweets.py
```

**Configuration:**

- Rate limit: 50 requests/minute
- Temperature: 0.3 (consistent classifications)
- Batch size: 50 tweets
- Auto-retry on failures (up to 3 attempts)

#### 3. **compare_classifications.py** - Consensus & Validation

Compares and merges classifications from GPT and Fanar into a single consensus label.

**Key Features:**

- **Flexible Strategies**: Choose to prefer one source or require agreement
- **Disagreement Analysis**: Detailed statistics on classification differences
- **Quality Metrics**: Agreement rates and confidence scores
- **Final Dataset**: Creates `arabic_tweets_matched_classifications.csv`

**Usage:**

```bash
cd labeling
python compare_classifications.py
```

**Configuration Options:**

```python
CONFIG = {
    'prefer_source': 'fanar',      # or 'gpt'
    'require_match': False,        # True = only keep agreements
    'output_file': 'datasets/arabic_tweets_matched_classifications.csv'
}
```

**Comparison Strategies:**

- **Prefer Source**: Use one model as primary, other as validation
- **Require Match**: Keep only tweets where both models agree (higher quality, smaller dataset)
- **Statistics**: Outputs agreement rates per category

### Labeling Pipeline Workflow

```
Raw Tweets → Cleaning → Batch Classification (GPT) → Validation (Fanar) → Consensus → Final Dataset
                ↓                    ↓                       ↓                ↓
         (cleaned.csv)      (classified.csv)        (classified.csv)  (matched.csv)
```

### Quality Assurance

The multi-model approach ensures:

1. **Cross-Validation**: Two independent models validate each classification
2. **Arabic Expertise**: Fanar specializes in Arabic nuances
3. **Cost Efficiency**: Batch API reduces costs by 50%
4. **Reliability**: State management prevents data loss during processing
5. **Reproducibility**: Saved batch files and state for audit trail

### Classification Categories

All labeling scripts use these four categories:

- **depression**: Sadness, hopelessness, lack of motivation
- **anxiety**: Worry, fear, stress, nervous thoughts
- **suicidal_ideation**: Self-harm references, suicide mentions (critical)
- **neutral**: No mental health signals detected

### Cost Estimation

For 50,000 tweets:

- **GPT-4o-mini (Batch)**: ~$2-3 total
- **Fanar API**: Varies by provider
- **Total Processing Time**: 2-4 hours depending on API rate limits

See `labeling/README.md` for detailed documentation and troubleshooting.

## 📊 Dataset

The dataset consists of Arabic tweets classified into four categories:

- **Depression**: Tweets expressing depressive symptoms
- **Anxiety**: Tweets indicating anxiety or stress
- **Suicidal Ideation**: Tweets with suicide-related content (critical risk)
- **Neutral**: General tweets without mental health signals

**Important**: The dataset is highly imbalanced, with neutral tweets dominating. The system uses SMOTE and other techniques to address this.

## 🎯 Model Performance

The system evaluates models using:

- **F1 Score (Macro)**: Best for imbalanced data
- **F1 Score (Weighted)**: Overall performance
- **Clinical Metrics**: Sensitivity, specificity, PPV, NPV
- **Per-Class Performance**: Detection rate for each condition

Best performing models typically include:

- XGBoost with SMOTE
- LightGBM with class weights
- AraBERT fine-tuned models

## 🔧 Key Technologies

- **NLP Libraries**: CAMeL Tools, Farasa, AraBERT, pyarabic
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly

## ⚠️ Ethical Considerations

This system is designed to **supplement, not replace** professional mental health assessment. Key considerations:

1. **Privacy**: All data is anonymized
2. **Clinical Integration**: Requires validation by mental health professionals
3. **Crisis Management**: High-risk cases need immediate professional referral
4. **Bias Monitoring**: Regular audits for fairness across demographics
5. **Limitations**: Model performance depends on data quality and cultural context

## 🤖 AI-Assisted Development

**Important Attribution**: While the core research ideas, methodology, and project design are entirely original work by the project team, AI tools (including GitHub Copilot and ChatGPT) were used to:

- Enhance code efficiency and implement best practices
- Generate boilerplate code and documentation
- Debug technical issues and optimize algorithms
- Improve code readability and structure

**Original Contributions** by the research team include:

- Research design and methodology
- Arabic-specific preprocessing pipeline design
- Feature engineering strategies for mental health detection
- Clinical validity framework
- Model selection and evaluation approach
- Ethical guidelines and limitations analysis
- Data collection and labeling strategy

The intellectual property, research insights, and innovative approaches remain the original work of the project authors.

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{arabic_mental_health_detection,
  title={Arabic Mental Health Detection System},
  author={Ahmed Safari},
  year={2025},
  url={https://github.com/ahmed-safari/aicc-project}
}
```

## 📄 License

This project is for research and educational purposes. It was created as a project for the AICC course for Dr. Somaiyeh MahmoudZadeh

## 🙏 Acknowledgments

- AraBERT team for pre-trained models
- CAMeL Tools and Farasa teams for Arabic NLP tools
- Mental health research community for guidance on ethical AI

---

**⚠️ Disclaimer**: This tool is for research purposes only and should not be used as a substitute for professional mental health diagnosis or treatment. If you or someone you know is experiencing a mental health crisis, please contact local emergency services or mental health hotlines immediately.
