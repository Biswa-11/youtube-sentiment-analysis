# YouTube Comment Sentiment Analysis (Flask)

A production-style Flask web app that analyzes YouTube video comments with spam filtering, language translation, hybrid sentiment scoring (VADER + DistilBERT), timeline visualization, and a dark themed dashboard.

## Features

- YouTube video metadata extraction (title, channel, views, likes, publish date, thumbnail)
- Comment fetch with pagination via YouTube Data API v3
- Spam detection and removal using rule-based filters
- Multilingual detection + translation to English before analysis
- Hybrid sentiment engine:
  - VADER for fast baseline sentiment
  - DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) for neutral-edge refinement
- Sentiment distribution pie chart and monthly sentiment timeline chart
- Word cloud generation from cleaned comments
- Top positive and top negative comments
- Flask-Caching to cache results by video ID for 1 hour
- HTML API endpoint (`/analyze`) + JSON API endpoint (`/api/analyze`)

## Prerequisites

- Python 3.10+
- pip
- Internet connection (for YouTube API calls, translation, model download)
- YouTube Data API v3 key

## Installation

1. Clone and enter project:

   ```bash
   git clone <your-repo-url>
   cd youtube_sentiment
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK assets:

   ```bash
   python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
   ```

4. Get YouTube Data API v3 key from Google Cloud Console:
   - Enable **YouTube Data API v3**
   - Create API key credentials

5. Add your key in `config.py`:

   ```python
   YOUTUBE_API_KEY = "YOUR_ACTUAL_KEY"
   ```

6. Run the app:

   ```bash
   python app.py
   ```

7. Open:
   - [http://localhost:5000](http://localhost:5000)

## Usage

1. Open the homepage.
2. Paste a valid YouTube video URL.
3. Click **Analyze**.
4. Review:
   - Video details
   - Sentiment summary
   - Pie and timeline charts
   - Word cloud
   - Top positive/negative comments

## Configuration

`config.py` supports:

- `YOUTUBE_API_KEY`: YouTube API key
- `MAX_COMMENTS`: Max comments to fetch (default: 500)
- `SPAM_THRESHOLD`: Minimum valid comment length (default: 3)
- `CACHE_TIMEOUT`: Cache TTL in seconds (default: 3600)
- `SECRET_KEY`: Flask session secret

## Deployment (Render/Railway)

1. Push project to GitHub.
2. Create a new Web Service on Render or Railway.
3. Set environment/runtime to Python 3.10+.
4. Install command:
   - `pip install -r requirements.txt`
5. Start command:
   - `python app.py`
6. Add your API key in `config.py` or inject through environment variables (recommended for production).
7. Ensure outbound internet is enabled for API/model calls.

## Known Limitations

- YouTube Data API quota is limited to **10,000 units/day**.
- DistilBERT initial model download is large (~250MB).
- Translation quality depends on upstream service availability.
- Some videos disable comments or hide likes/comments metadata.
