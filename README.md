# NVIDIA-news-coverage-analysis

A computational analysis of NVIDIA's newsroom articles using NLP techniques including topic modeling (LDA/NMF), sentiment analysis, and temporal trend analysis.

## Project Overview

This project scrapes and analyzes articles from NVIDIA's newsroom to identify:
- Key themes and topics in NVIDIA's public communications
- Sentiment patterns over time
- Temporal trends in coverage

**Total articles analyzed:** 175
**Time period: 2020-2026**

## Methodology

- **Web Scraping**: Custom scraper for NVIDIA newsroom
- **Topic Modeling**: LDA and NMF for theme extraction
- **Sentiment Analysis**: Rule-based
- **Temporal Analysis**: Time-series analysis of trends

```
nvidia-newsroom-analysis/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── .gitignore               # Git ignore rules
│
├── src/                     # Source code
│   ├── scraping/
│   │   └── nvidia_newsroom_scraper.py    # Web scraper for NVIDIA newsroom
│   └── analysis/
│       └── news_nlp_analysis.py          # NLP analysis pipeline
│
└──── data/                 # Data files
    ├── scraped_content/     # Raw scraped data (CSV, JSON, XLSX)
    └── nlp_results/         # Processed analysis outputs


## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
git clone https://github.com/Glbste/nvidia-newsroom-analysis
cd nvidia-newsroom-analysis
pip install -r requirements.txt
```

### Usage
```bash
# Run scraper
python src/scraping/nvidia_newsroom_scraper.py

# Run analysis
python src/analysis/news_nlp_analysis.py
```


## Technologies Used

- Python (pandas, scikit-learn, nltk/spacy)
- BeautifulSoup/Scrapy for web scraping
- LDA/NMF for topic modeling

## 📝 License

MIT License

## Author

**Stefano Rolesu**
- PhD applicant
- Research Focus: Corporate Ontological Narratives

```
