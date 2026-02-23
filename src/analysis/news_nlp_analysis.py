"""
Comprehensive NLP Analysis of NVIDIA Newsroom Content

This script performs:
1. Topic Modeling (LDA and NMF)
2. Sentiment Analysis (multiple approaches)
3. Temporal Analysis
4. Word Frequency Analysis
5. N-gram Analysis
6. Statistical Summaries
7. Visualizations

"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# NLP and ML libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


# CONFIGURATION


CONFIG = {
    'input_file': 'data/nvidia_newsroom_20260208_183539.xlsx',
    'sheet_name': 'Articles',
    'n_topics': 6,
    'n_top_words': 15,
    'min_df': 2,
    'max_df': 0.85,
    'ngram_range': (1, 2),
    'output_dir': 'data/nlp_results'
}

# DATA LOADING AND PREPROCESSING

def load_data(file_path, sheet_name='Articles'):
    """Load and prepare the dataset"""
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    df['year'] = df['publication_date'].dt.year
    df['month'] = df['publication_date'].dt.month
    df['year_month'] = df['publication_date'].dt.to_period('M')
    df['quarter'] = df['publication_date'].dt.to_period('Q')
    print(f"✓ Loaded {len(df)} articles")
    print(f"✓ Date range: {df['publication_date'].min()} to {df['publication_date'].max()}")
    return df

def create_stopwords():
    """Create comprehensive stopword list"""
    basic_stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
        'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
        'just', 'don', 'now', 'nvidia', 'nvidias', 've', 'd', 'll', 'm', 're',
        'one', 'two', 'three', 'get', 'says', 'new', 'year', 'us', 'go'
    }
    return list(basic_stopwords)

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces and hyphens
    text = re.sub(r'[^a-zA-Z\s-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove very short words (less than 2 characters)
    text = ' '.join([word for word in text.split() if len(word) > 2])
    
    return text

# TOPIC MODELING

def perform_lda_topic_modeling(texts, n_topics=6, n_top_words=15):
    """
    Perform topic modeling using Latent Dirichlet Allocation (LDA)
    Best for: Finding semantic topics in document collections
    """
    print(f"\n{'='*60}")
    print("TOPIC MODELING: LDA (Latent Dirichlet Allocation)")
    print(f"{'='*60}")
    
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=CONFIG['max_df'],
        min_df=CONFIG['min_df'],
        max_features=1000,
        stop_words=create_stopwords(),
        ngram_range=CONFIG['ngram_range']
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Document-term matrix shape: {doc_term_matrix.shape}")
    print(f"Vocabulary size: {len(feature_names)}")
    
    # Fit LDA model
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=30,
        learning_method='online',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    doc_topics = lda_model.fit_transform(doc_term_matrix)
    
    # Extract topics
    topics = {}
    print(f"\nDiscovered {n_topics} Topics:")
    print("-" * 60)
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]
        
        topics[f'Topic_{topic_idx + 1}'] = {
            'words': top_words,
            'weights': [float(w) for w in top_weights],
            'top_5': ', '.join(top_words[:5])
        }
        
        print(f"\nTopic {topic_idx + 1}:")
        print(f"  Top words: {', '.join(top_words[:10])}")
    
    # Assign dominant topic to each document
    dominant_topics = np.argmax(doc_topics, axis=1)
    topic_distribution = pd.Series(dominant_topics).value_counts().sort_index()
    
    print(f"\n{'='*60}")
    print("Topic Distribution across Documents:")
    for topic_num, count in topic_distribution.items():
        percentage = (count / len(dominant_topics)) * 100
        print(f"  Topic {topic_num + 1}: {count} documents ({percentage:.1f}%)")
    
    return {
        'topics': topics,
        'doc_topics': doc_topics,
        'dominant_topics': dominant_topics,
        'vectorizer': vectorizer,
        'model': lda_model,
        'method': 'LDA'
    }

def perform_nmf_topic_modeling(texts, n_topics=6, n_top_words=15):
    """
    Perform topic modeling using Non-negative Matrix Factorization (NMF)
    Best for: Finding more distinct, interpretable topics
    """
    print(f"\n{'='*60}")
    print("TOPIC MODELING: NMF (Non-negative Matrix Factorization)")
    print(f"{'='*60}")
    
    # Use TF-IDF for NMF (generally works better than raw counts)
    vectorizer = TfidfVectorizer(
        max_df=CONFIG['max_df'],
        min_df=CONFIG['min_df'],
        max_features=1000,
        stop_words=create_stopwords(),
        ngram_range=CONFIG['ngram_range']
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Fit NMF model
    nmf_model = NMF(
        n_components=n_topics,
        random_state=42,
        max_iter=300,
        init='nndsvda'
    )
    
    doc_topics = nmf_model.fit_transform(tfidf_matrix)
    
    # Extract topics
    topics = {}
    print(f"\nDiscovered {n_topics} Topics:")
    print("-" * 60)
    
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]
        
        topics[f'Topic_{topic_idx + 1}'] = {
            'words': top_words,
            'weights': [float(w) for w in top_weights],
            'top_5': ', '.join(top_words[:5])
        }
        
        print(f"\nTopic {topic_idx + 1}:")
        print(f"  Top words: {', '.join(top_words[:10])}")
    
    dominant_topics = np.argmax(doc_topics, axis=1)
    
    return {
        'topics': topics,
        'doc_topics': doc_topics,
        'dominant_topics': dominant_topics,
        'vectorizer': vectorizer,
        'model': nmf_model,
        'method': 'NMF'
    }

# SENTIMENT ANALYSIS

def create_sentiment_lexicons():
    """Create comprehensive sentiment lexicons"""
    
    positive_words = {
        # Achievement & Success
        'achievement', 'accomplish', 'advance', 'advantage', 'amazing', 'award',
        'best', 'better', 'breakthrough', 'brilliant', 'celebrate', 'champion',
        
        # Innovation & Technology
        'cutting-edge', 'deliver', 'efficient', 'enhance', 'excellent', 'exceptional',
        'exciting', 'expand', 'fastest', 'first', 'flagship', 'forefront',
        
        # Growth & Performance
        'gain', 'great', 'grow', 'growth', 'high-performance', 'improve', 'increase',
        'innovation', 'innovative', 'launch', 'launches', 'lead', 'leader', 'leading',
        
        # Market & Business
        'new', 'outstanding', 'partner', 'partnership', 'performance', 'pioneer',
        'powerful', 'premier', 'progress', 'record', 'revolutionary', 'rise',
        
        # Competitive Advantage
        'strong', 'success', 'successful', 'superior', 'top', 'transform',
        'transformative', 'unprecedented', 'win', 'winner', 'accelerate', 'advanced',
        'breakthrough', 'dominant', 'elite', 'fastest', 'premium', 'revolutionary',
        'state-of-the-art', 'ultimate', 'unmatched', 'unveils', 'world-class'
    }
    
    negative_words = {
        'challenge', 'challenging', 'concern', 'concerns', 'crisis', 'decline',
        'decrease', 'difficult', 'difficulty', 'disappointing', 'fail', 'failure',
        'fall', 'fell', 'loss', 'losses', 'problem', 'problems', 'risk', 'risks',
        'threat', 'threatens', 'trouble', 'weak', 'weakness', 'worse', 'worsen',
        'delay', 'delays', 'issue', 'issues', 'struggle', 'struggling'
    }
    
    return positive_words, negative_words

def rule_based_sentiment(text, positive_words, negative_words):
    """Rule-based sentiment analysis"""
    if pd.isna(text) or text == "":
        return {'score': 0.0, 'label': 'neutral', 'pos_count': 0, 'neg_count': 0}
    
    words = text.lower().split()
    
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    total_words = len(words)
    if total_words == 0:
        return {'score': 0.0, 'label': 'neutral', 'pos_count': 0, 'neg_count': 0}
    
    # Calculate sentiment score (-1 to 1)
    score = (pos_count - neg_count) / total_words
    
    # Classify sentiment with adjusted thresholds
    if score > 0.02:
        label = 'positive'
    elif score < -0.02:
        label = 'negative'
    else:
        label = 'neutral'
    
    return {
        'score': score,
        'label': label,
        'pos_count': pos_count,
        'neg_count': neg_count
    }

def analyze_sentiment(df, text_column='processed_title'):
    """Perform comprehensive sentiment analysis"""
    print(f"\n{'='*60}")
    print("SENTIMENT ANALYSIS")
    print(f"{'='*60}")
    
    positive_words, negative_words = create_sentiment_lexicons()
    
    # Apply sentiment analysis
    sentiments = df[text_column].apply(
        lambda x: rule_based_sentiment(x, positive_words, negative_words)
    )
    
    df['sentiment_score'] = sentiments.apply(lambda x: x['score'])
    df['sentiment_label'] = sentiments.apply(lambda x: x['label'])
    df['positive_word_count'] = sentiments.apply(lambda x: x['pos_count'])
    df['negative_word_count'] = sentiments.apply(lambda x: x['neg_count'])
    
    # Summary statistics
    sentiment_dist = df['sentiment_label'].value_counts()
    avg_sentiment = df['sentiment_score'].mean()
    std_sentiment = df['sentiment_score'].std()
    
    print("\nSentiment Distribution:")
    for label, count in sentiment_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {label.capitalize()}: {count} articles ({percentage:.1f}%)")
    
    print(f"\nSentiment Score Statistics:")
    print(f"  Mean: {avg_sentiment:.4f}")
    print(f"  Std Dev: {std_sentiment:.4f}")
    print(f"  Min: {df['sentiment_score'].min():.4f}")
    print(f"  Max: {df['sentiment_score'].max():.4f}")
    print(f"  Median: {df['sentiment_score'].median():.4f}")
    
    # Most positive and negative articles
    print("\nMost Positive Articles:")
    top_positive = df.nlargest(3, 'sentiment_score')[['title', 'sentiment_score', 'publication_date']]
    for idx, row in top_positive.iterrows():
        print(f"  • {row['title'][:60]}... (score: {row['sentiment_score']:.4f})")
    
    print("\nMost Negative Articles:")
    top_negative = df.nsmallest(3, 'sentiment_score')[['title', 'sentiment_score', 'publication_date']]
    for idx, row in top_negative.iterrows():
        print(f"  • {row['title'][:60]}... (score: {row['sentiment_score']:.4f})")
    
    return df

# WORD FREQUENCY ANALYSIS

def analyze_word_frequencies(texts, n_top=30):
    """Extract and analyze word frequencies"""
    print(f"\n{'='*60}")
    print("WORD FREQUENCY ANALYSIS")
    print(f"{'='*60}")
    
    # Unigrams
    vectorizer_1gram = CountVectorizer(
        max_df=CONFIG['max_df'],
        min_df=CONFIG['min_df'],
        stop_words=create_stopwords(),
        ngram_range=(1, 1)
    )
    
    dtm_1gram = vectorizer_1gram.fit_transform(texts)
    word_freq = np.asarray(dtm_1gram.sum(axis=0)).flatten()
    words = vectorizer_1gram.get_feature_names_out()
    
    word_freq_pairs = sorted(zip(words, word_freq), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {min(n_top, len(word_freq_pairs))} Most Frequent Words:")
    for i, (word, freq) in enumerate(word_freq_pairs[:n_top], 1):
        print(f"  {i:2d}. {word:25s} → {int(freq):4d} occurrences")
    
    # Bigrams
    vectorizer_2gram = CountVectorizer(
        max_df=CONFIG['max_df'],
        min_df=CONFIG['min_df'],
        stop_words=create_stopwords(),
        ngram_range=(2, 2)
    )
    
    dtm_2gram = vectorizer_2gram.fit_transform(texts)
    bigram_freq = np.asarray(dtm_2gram.sum(axis=0)).flatten()
    bigrams = vectorizer_2gram.get_feature_names_out()
    
    bigram_freq_pairs = sorted(zip(bigrams, bigram_freq), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 Most Frequent Bigrams:")
    for i, (bigram, freq) in enumerate(bigram_freq_pairs[:20], 1):
        print(f"  {i:2d}. {bigram:35s} → {int(freq):4d} occurrences")
    
    # Trigrams
    vectorizer_3gram = CountVectorizer(
        max_df=0.9,
        min_df=2,
        stop_words=create_stopwords(),
        ngram_range=(3, 3)
    )
    
    dtm_3gram = vectorizer_3gram.fit_transform(texts)
    trigram_freq = np.asarray(dtm_3gram.sum(axis=0)).flatten()
    trigrams = vectorizer_3gram.get_feature_names_out()
    
    trigram_freq_pairs = sorted(zip(trigrams, trigram_freq), key=lambda x: x[1], reverse=True)
    
    if len(trigram_freq_pairs) > 0:
        print(f"\nTop 10 Most Frequent Trigrams:")
        for i, (trigram, freq) in enumerate(trigram_freq_pairs[:10], 1):
            print(f"  {i:2d}. {trigram:45s} → {int(freq):4d} occurrences")
    
    return {
        'word_freq': word_freq_pairs[:n_top],
        'bigram_freq': bigram_freq_pairs[:20],
        'trigram_freq': trigram_freq_pairs[:10] if len(trigram_freq_pairs) > 0 else []
    }

# TEMPORAL ANALYSIS

def temporal_analysis(df):
    """Analyze trends over time"""
    print(f"\n{'='*60}")
    print("TEMPORAL ANALYSIS")
    print(f"{'='*60}")
    
    # Articles by year
    print("\nArticles by Year:")
    yearly_counts = df['year'].value_counts().sort_index()
    for year, count in yearly_counts.items():
        print(f"  {year}: {count:3d} articles")
    
    # Sentiment trends over time
    print("\nAverage Sentiment by Year:")
    sentiment_by_year = df.groupby('year')['sentiment_score'].agg(['mean', 'std', 'count'])
    for year, row in sentiment_by_year.iterrows():
        print(f"  {year}: {row['mean']:7.4f} (±{row['std']:.4f}, n={int(row['count'])})")
    
    # Topic distribution over time
    if 'topic_lda' in df.columns:
        print("\nTopic Distribution by Year:")
        topic_year = pd.crosstab(df['year'], df['topic_lda'], normalize='index') * 100
        print(topic_year.round(1))
    
    # Quarterly analysis
    print("\nArticles by Quarter (Last 2 years):")
    recent_quarters = df[df['year'] >= df['year'].max() - 1].groupby('quarter').size()
    for quarter, count in recent_quarters.items():
        print(f"  {quarter}: {count} articles")
    
    return {
        'yearly_counts': yearly_counts,
        'sentiment_by_year': sentiment_by_year,
        'quarterly_counts': recent_quarters
    }

# ADVANCED ANALYSIS

def keyword_cooccurrence_analysis(texts, top_n=15):
    """Analyze which keywords co-occur frequently"""
    print(f"\n{'='*60}")
    print("KEYWORD CO-OCCURRENCE ANALYSIS")
    print(f"{'='*60}")
    
    vectorizer = CountVectorizer(
        max_df=0.8,
        min_df=3,
        stop_words=create_stopwords(),
        ngram_range=(1, 1),
        max_features=50
    )
    
    dtm = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate co-occurrence matrix
    cooccurrence = (dtm.T @ dtm).toarray()
    np.fill_diagonal(cooccurrence, 0)
    
    # Find top co-occurrences
    cooccurrences_list = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if cooccurrence[i, j] > 0:
                cooccurrences_list.append((
                    feature_names[i],
                    feature_names[j],
                    cooccurrence[i, j]
                ))
    
    cooccurrences_list.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nTop {top_n} Keyword Pairs (Co-occurrence Frequency):")
    for i, (word1, word2, freq) in enumerate(cooccurrences_list[:top_n], 1):
        print(f"  {i:2d}. '{word1}' + '{word2}': {int(freq)} times")
    
    return cooccurrences_list[:top_n]

def content_type_analysis(df):
    """Analyze differences by content type"""
    print(f"\n{'='*60}")
    print("CONTENT TYPE ANALYSIS")
    print(f"{'='*60}")
    
    content_dist = df['content_type'].value_counts()
    print("\nContent Type Distribution:")
    for ctype, count in content_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {ctype}: {count} ({percentage:.1f}%)")
    
    # Sentiment by content type
    if 'sentiment_score' in df.columns:
        print("\nAverage Sentiment by Content Type:")
        sentiment_by_type = df.groupby('content_type')['sentiment_score'].agg(['mean', 'count'])
        for ctype, row in sentiment_by_type.iterrows():
            if row['count'] > 0:
                print(f"  {ctype}: {row['mean']:.4f} (n={int(row['count'])})")

# VISUALIZATION FUNCTIONS

def create_visualizations(df, topic_results, freq_results, output_dir='nlp_results'):
    """Create comprehensive visualizations"""
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 1. Topic Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    topic_counts = pd.Series(topic_results['dominant_topics']).value_counts().sort_index()
    topic_labels = [f"Topic {i+1}" for i in topic_counts.index]
    
    bars = ax.bar(topic_labels, topic_counts.values, color=sns.color_palette("husl", len(topic_labels)))
    ax.set_xlabel('Topic', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Articles', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Articles Across Topics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_topic_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 01_topic_distribution.png")
    plt.close()
    
    # 2. Sentiment Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    sentiment_counts = df['sentiment_label'].value_counts()
    colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    pie_colors = [colors.get(label, '#3498db') for label in sentiment_counts.index]
    
    axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=pie_colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0].set_title('Sentiment Distribution', fontsize=13, fontweight='bold')
    
    # Histogram
    axes[1].hist(df['sentiment_score'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(df['sentiment_score'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {df["sentiment_score"].mean():.4f}')
    axes[1].set_xlabel('Sentiment Score', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Sentiment Score Distribution', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 02_sentiment_distribution.png")
    plt.close()
    
    # 3. Temporal Analysis - Articles Over Time
    fig, ax = plt.subplots(figsize=(12, 6))
    yearly_counts = df['year'].value_counts().sort_index()
    
    ax.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2,
            markersize=8, color='#3498db')
    ax.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3, color='#3498db')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Articles', fontsize=12, fontweight='bold')
    ax.set_title('NVIDIA Newsroom Articles Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for x, y in zip(yearly_counts.index, yearly_counts.values):
        ax.text(x, y + 1, str(int(y)), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_temporal_trends.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 03_temporal_trends.png")
    plt.close()
    
    # 4. Sentiment Over Time
    fig, ax = plt.subplots(figsize=(12, 6))
    sentiment_by_year = df.groupby('year')['sentiment_score'].mean()
    
    ax.plot(sentiment_by_year.index, sentiment_by_year.values, marker='o',
            linewidth=2, markersize=8, color='#e74c3c')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Sentiment Score', fontsize=12, fontweight='bold')
    ax.set_title('Sentiment Trend Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_sentiment_over_time.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 04_sentiment_over_time.png")
    plt.close()
    
    # 5. Top Words Bar Chart
    fig, ax = plt.subplots(figsize=(12, 8))
    words, freqs = zip(*freq_results['word_freq'][:20])
    
    y_pos = np.arange(len(words))
    colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(words)))
    
    ax.barh(y_pos, freqs, color=colors_grad)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Most Frequent Words', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(freqs):
        ax.text(v + 0.5, i, str(int(v)), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_top_words.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 05_top_words.png")
    plt.close()
    
    # 6. Topic-Sentiment Heatmap
    if 'topic_lda' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        topic_sentiment = df.groupby(['topic_lda', 'sentiment_label']).size().unstack(fill_value=0)
        topic_sentiment_pct = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0) * 100
        
        sns.heatmap(topic_sentiment_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                    cbar_kws={'label': 'Percentage (%)'}, ax=ax)
        ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Topic', fontsize=12, fontweight='bold')
        ax.set_title('Sentiment Distribution by Topic (%)', fontsize=14, fontweight='bold')
        ax.set_yticklabels([f'Topic {i+1}' for i in range(len(topic_sentiment_pct))], rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/06_topic_sentiment_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 06_topic_sentiment_heatmap.png")
        plt.close()
    
    print(f"\n  All visualizations saved to '{output_dir}/' directory")

# ============================================================================
# EXPORT RESULTS
# ============================================================================

def export_results(df, topic_results_lda, topic_results_nmf, freq_results, output_dir='nlp_results'):
    """Export all results to files"""
    print(f"\n{'='*60}")
    print("EXPORTING RESULTS")
    print(f"{'='*60}")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Main results DataFrame
    export_cols = ['date_string', 'publication_date', 'title', 'content_type', 'year',
                   'sentiment_score', 'sentiment_label', 'positive_word_count',
                   'negative_word_count', 'topic_lda', 'topic_nmf']
    
    df_export = df[[col for col in export_cols if col in df.columns]].copy()
    df_export.to_csv(f'{output_dir}/analysis_results.csv', index=False)
    print(f"  ✓ Saved: analysis_results.csv ({len(df_export)} rows)")
    
    # 2. LDA Topics
    with open(f'{output_dir}/topics_lda.json', 'w') as f:
        json.dump(topic_results_lda['topics'], f, indent=2)
    print(f"  ✓ Saved: topics_lda.json")
    
    # 3. NMF Topics
    with open(f'{output_dir}/topics_nmf.json', 'w') as f:
        json.dump(topic_results_nmf['topics'], f, indent=2)
    print(f"  ✓ Saved: topics_nmf.json")
    
    # 4. Word frequencies
    word_freq_df = pd.DataFrame(freq_results['word_freq'], columns=['word', 'frequency'])
    word_freq_df.to_csv(f'{output_dir}/word_frequencies.csv', index=False)
    print(f"  ✓ Saved: word_frequencies.csv")
    
    bigram_freq_df = pd.DataFrame(freq_results['bigram_freq'], columns=['bigram', 'frequency'])
    bigram_freq_df.to_csv(f'{output_dir}/bigram_frequencies.csv', index=False)
    print(f"  ✓ Saved: bigram_frequencies.csv")
    
    if len(freq_results['trigram_freq']) > 0:
        trigram_freq_df = pd.DataFrame(freq_results['trigram_freq'], columns=['trigram', 'frequency'])
        trigram_freq_df.to_csv(f'{output_dir}/trigram_frequencies.csv', index=False)
        print(f"  ✓ Saved: trigram_frequencies.csv")
    
    # 5. Temporal analysis
    yearly_sentiment = df.groupby('year').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'title': 'count'
    }).round(4)
    yearly_sentiment.columns = ['avg_sentiment', 'std_sentiment', 'n_sentiment', 'n_articles']
    yearly_sentiment.to_csv(f'{output_dir}/temporal_analysis.csv')
    print(f"  ✓ Saved: temporal_analysis.csv")
    
    # 6. Topic-Sentiment crosstab
    if 'topic_lda' in df.columns:
        topic_sentiment = pd.crosstab(
            df['topic_lda'],
            df['sentiment_label'],
            margins=True,
            margins_name='Total'
        )
        topic_sentiment.to_csv(f'{output_dir}/topic_sentiment_crosstab.csv')
        print(f"  ✓ Saved: topic_sentiment_crosstab.csv")
    
    # 7. Summary statistics
    summary = {
        'total_articles': len(df),
        'date_range': {
            'start': str(df['publication_date'].min()),
            'end': str(df['publication_date'].max()),
            'span_days': (df['publication_date'].max() - df['publication_date'].min()).days
        },
        'sentiment': {
            'mean': float(df['sentiment_score'].mean()),
            'std': float(df['sentiment_score'].std()),
            'median': float(df['sentiment_score'].median()),
            'distribution': df['sentiment_label'].value_counts().to_dict()
        },
        'topics_lda': {
            'n_topics': len(topic_results_lda['topics']),
            'distribution': pd.Series(topic_results_lda['dominant_topics']).value_counts().to_dict()
        },
        'topics_nmf': {
            'n_topics': len(topic_results_nmf['topics']),
            'distribution': pd.Series(topic_results_nmf['dominant_topics']).value_counts().to_dict()
        }
    }
    
    with open(f'{output_dir}/summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: summary_statistics.json")
    
    print(f"\n  All results exported to '{output_dir}/' directory")

# MAIN EXECUTION

def main():
    """Main analysis pipeline"""
    
    print("\n" + "="*70)
    print("  NVIDIA NEWSROOM - COMPREHENSIVE NLP ANALYSIS")
    print("  For Social Science Research & Academic Studies")
    print("="*70)
    
    # Load data
    df = load_data(CONFIG['input_file'], CONFIG['sheet_name'])
    
    # Preprocess
    print(f"\n{'='*60}")
    print("TEXT PREPROCESSING")
    print(f"{'='*60}")
    df['processed_title'] = df['title'].apply(preprocess_text)
    texts = df[df['processed_title'].str.len() > 10]['processed_title'].tolist()
    print(f"✓ Preprocessed {len(texts)} valid texts")
    
    # Topic Modeling - LDA
    topic_results_lda = perform_lda_topic_modeling(texts, n_topics=CONFIG['n_topics'])
    df['topic_lda'] = -1
    df.loc[df['processed_title'].str.len() > 10, 'topic_lda'] = topic_results_lda['dominant_topics']
    
    # Topic Modeling - NMF
    topic_results_nmf = perform_nmf_topic_modeling(texts, n_topics=CONFIG['n_topics'])
    df['topic_nmf'] = -1
    df.loc[df['processed_title'].str.len() > 10, 'topic_nmf'] = topic_results_nmf['dominant_topics']
    
    # Sentiment Analysis
    df = analyze_sentiment(df)
    
    # Word Frequency Analysis
    freq_results = analyze_word_frequencies(texts)
    
    # Temporal Analysis
    temporal_results = temporal_analysis(df)
    
    # Advanced Analysis
    cooccurrence_results = keyword_cooccurrence_analysis(texts)
    content_type_analysis(df)
    
    # Generate Visualizations
    create_visualizations(df, topic_results_lda, freq_results, CONFIG['output_dir'])
    
    # Export Results
    export_results(df, topic_results_lda, topic_results_nmf, freq_results, CONFIG['output_dir'])
    
    # Final Summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"✓ Analyzed {len(df)} articles from NVIDIA Newsroom")
    print(f"✓ Date range: {df['publication_date'].min().date()} to {df['publication_date'].max().date()}")
    print(f"✓ Identified {CONFIG['n_topics']} topics using LDA and NMF")
    print(f"✓ Performed sentiment analysis (avg score: {df['sentiment_score'].mean():.4f})")
    print(f"✓ Generated visualizations and exported results")
    print(f"\nResults saved to: '{CONFIG['output_dir']}/' directory")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
