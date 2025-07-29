# SEAL-drip: A Lifelong-Updating Llama

A continually self-updating LLM that learns from fresh data while minimizing catastrophic forgetting.

## Current Status: Environment & Data Collection Setup ✅

This is the initial setup phase - we have a working data collection system that gathers information from:

- 📰 **HackerNews** top stories
- 📚 **arXiv** computer science papers
- 🌐 **Wikipedia** featured content (simplified for now)

## Project Structure

```
├── config.py              # Configuration settings
├── collect.py             # Main data collection script
├── test_collection.py     # Test suite for data collection
├── requirements.txt       # Python dependencies
├── data/                  # Data storage directory
│   ├── raw/              # Raw collected data (JSONL files)
│   └── processed/        # Processed data (for future use)
└── README.md             # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd "Week_5_Projects/A Lifelong-Updating Llama"
pip install -r requirements.txt
```

### 2. Test the Setup

Run the test suite to verify everything works:

```bash
python test_collection.py
```

This will:

- ✅ Test configuration loading
- 📡 Test each data collector with a small sample
- 💾 Verify data saving works correctly
- 🧹 Clean up test files

### 3. Collect Real Data

Once tests pass, run the full collection:

```bash
python collect.py
```

This will:

- Collect up to 30 HackerNews stories
- Collect up to 50 arXiv papers
- Collect Wikipedia featured content
- Save everything to `data/raw/` as JSONL files

## Configuration

All settings are in `config.py`. Key parameters:

- `MODEL_NAME`: Your Ollama model (`llama3.1:8b-instruct-q4_K_M`)
- `HACKERNEWS_MAX_ITEMS`: Max HN stories per day (30)
- `ARXIV_MAX_ITEMS`: Max arXiv papers per day (50)
- `MAX_DAILY_ITEMS`: Overall daily limit (10,000)

## Data Format

Each collected item is saved as JSON with these common fields:

- `collected_at`: ISO timestamp when collected
- Source-specific fields (title, content, URLs, etc.)

### HackerNews Format

```json
{
  "id": 12345,
  "title": "Story title",
  "url": "https://example.com",
  "score": 100,
  "by": "username",
  "collected_at": "2024-01-01T12:00:00"
}
```

### arXiv Format

```json
{
  "title": "Paper title",
  "summary": "Abstract text...",
  "authors": ["Author 1", "Author 2"],
  "link": "https://arxiv.org/abs/2401.00000",
  "collected_at": "2024-01-01T12:00:00"
}
```

## Next Steps (Future Development)

This is just the foundation! The next phases will add:

1. **Self-Edit Generation** - SEAL prompting to create learning notes
2. **Note Filtering** - MemoryBank scoring + Self-Taught Evaluator
3. **LoRA Training** - Gradient updates on filtered notes
4. **Checkpoint Management** - Weekly model snapshots
5. **Evaluation Suite** - Performance tracking
6. **Monitoring Dashboard** - Streamlit UI

## Notes

- The Wikipedia collector is simplified for now (uses featured content API)
- For production, you'd want the actual Wikipedia recent changes API
- The system is designed to be extended - each component is modular
- All data is saved locally in JSONL format for easy processing

## Troubleshooting

**Import errors?** Make sure you're in the right directory and installed requirements.

**Network errors?** Some APIs may be temporarily unavailable - this is normal.

**Empty data files?** Check your internet connection and try again.

**Want to customize?** Edit `config.py` to adjust collection parameters.
