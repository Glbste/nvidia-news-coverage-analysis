
#triple string """ x """ is not a comment but metadata that are embedded in the documentation of the script and accessible via help(module) 
#
#why including a DOCSTRING? shows the purpose of the bot, who is the author (attribution), also helps the developer to remember what is the script ee nafter some time, professional practice
# 
# is a short explanation of the WHY and WHAT behind the HOW (represented by the code itself) 

"""
NVIDIA Newsroom "In the News" Scraper
Professional web scraper for academic research in computational social science

Website: https://nvidianews.nvidia.com/in-the-news
Purpose: Collect press coverage and media mentions for discourse analysis
Author: Stefano Rolesu
Date: February 2026
License: For academic research purposes only
"""

#libraries that need to be requested are not necessarily buily into python 

import requests #makes HTTP requests to websites 
from bs4 import BeautifulSoup #parses html/xml documents
#uses NumPy to handle arrays in a faster way, helps with the creation of dataframes essential for data science purposes
import pandas as pd #data manipulation and analysis, alias creation is standard prcedure for panda library function calls (are made easier as they become pd.* rather than panda.*)
import time #add delays between requests. system calls to the OS to handle clock and sleep funxtions 
import logging #professional logging instead of print(). essential to assess the data extraction process
from datetime import datetime #handles dates and timestamps
from typing import List, Dict, Optional #type hints for better coding??
import json #save/load JSON data. Java Script Object Notation
from pathlib import Path #modern file path handling 
import re #regular expressions for pattern matching

# Configure logging
logging.basicConfig( # function to configure logger, only called once at the beggining since is a setup for logging behaviour in the rest for the entire program
    level=logging.INFO, #sets the threshold of which information is going the be recorder inf the log file (only debug is excluded in this context so to avoid cluttering the log file)
    format='%(asctime)s - %(levelname)s - %(message)s', #establishes the format for the message (timestamp, log level name (inof,warning,error), actual message)
    handlers=[ #handles the destination of the log information that was decided to be saved 
        logging.FileHandler('logs/nvidia_news_scraper.log'), #appends the log info at the end of the file
        logging.StreamHandler() #this gives runtime feedback and disappears as the program stops running
    ]
)


class NVIDIANewsroomScraper: #standard class creation syntax
   
# metadata explaining what is the prupose of the class
#   
    """
    Professional scraper for NVIDIA Newsroom "In the News" section
    
    Features:
    - Multi-year historical data collection
    - Polite rate limiting and ethical scraping
    - Comprehensive error handling
    - Data validation and cleaning
    - Multiple export formats
    - Progress tracking and resumption capability

    """
    
    def __init__(self, delay: float = 2.0): # constructor of the object that is called automatically when the class is instanciated in the main(). deafault delay time is set as 2 seconds
        """
        Initialize the scraper
        
        Args:
            delay: Delay between requests in seconds (default: 2.0)
                  Recommended: 2-5 seconds for respectful crawling
        """
        #create attributes to the current instance of the object NVIDIAscraper so that each instance has its own values of reference

        self.base_url = "https://nvidianews.nvidia.com" #main website of refrerence. since this is used in multiple ways, it's good practce to avoid repeating code whenever possible
        self.news_url = f"{self.base_url}/in-the-news" #specific section of the website where news are showed. "f"x/text"" allowa for text interpolation, meaning that a variale can be chained to a string
        self.delay = delay #sets the delay to the number explicited in the construction (instantiation) of the object, if input in null. sets it to default
        
        # Professional user agent
        self.headers = { #a header is part of the HTML package that is sent from the client to the server, containing a request. a header contains the metadata that help authenticate your request as a legitimate operation and not as a malicious botting attempt
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                         '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 '
                         '(Academic Research Bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', #accepted content types
            'Accept-Language': 'en-US,en;q=0.9', #accepted languages
            'Accept-Encoding': 'gzip, deflate, br', #accepted compression algorithms since servers might send compressed responses for faster trnasmission. the requests functions operate the decomrpession
            'Connection': 'keep-alive', #tells the server to keep the TCP open for multiple requests. explicitates the intention to create a session that opens once and closes when all the operations are finished. otherwise it would be necessary to establish a connection for every single html request
            'Upgrade-Insecure-Requests': '1' #tells the serve the client supports https connection for more secure connections for those websites that expect it
        }
        
        self.session = requests.Session() #uses session function from the requests library. creates a session object as attribute of the scraper instance that allows for multiple requests keeping coockies, headers consistent and pools connections. ssl verification?
        self.session.headers.update(self.headers) #overwrites default sessions headers with the ones declared above
        
        logging.info(f"Scraper initialized with {delay}s delay between requests") #logging the succsessful initialisation of the scraper
        logging.info(f"Target: {self.news_url}")
    
    def fetch_page(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:#function parameters: self: current instnace of the scraper, url: mandatory parameter, retries counter st to deafult value of 3. count the number of times operations need to be attempted when script catches an exception
        """
        Fetch a page with exponential backoff retry logic
        
        Args:
            url: URL to fetch
            retries: Number of retry attempts (default: 3)
            
        Returns:
            BeautifulSoup object or None if failed
        """
        for attempt in range(retries): #loop through every attemp as many times as allowed retires untile either return the BeautifulSoup object or null
            # prepares to catch an exception potentially generated by the next block of code: 
            # Network errors: Connection timeout, DNS failure, no internet; HTTP errors: 404, 500, 403, 429; SSL errors: Certificate expired, hostname mismatch; Parsing errors: Invalid HTML (rare with BeautifulSoup
            # it avoids program crashes when errors are encountered
            try: 
                logging.info(f"Fetching: {url} (Attempt {attempt + 1}/{retries})") #logs info about what it's being done at this point and which attempt this is 
                
                response = self.session.get(url, timeout=15) #inscribe in variable "response" the result of the HTTP request .get() [the client is going to wait timeout(15) secods for a response before generating exception and moving on to a retry rather that hanging forever]. deeper potential explanation of what happens when a get request is presented from a client to a server. a lot of internet protocols are involved in this very line which is where the connection actually happens
                response.raise_for_status() #check for the HTTP response so to avoid parning pages the were actually received with a 404, 500 error and so forth. If an exception is raised that is catched by 'except' construct 
                
                # Respectful rate limiting only after a successeful request (raised status is OK) so not to eselessly wait seconds onnly for trying again
                time.sleep(self.delay)
                
                return BeautifulSoup(response.content, 'html.parser') #succsessful return of the HTML page that will be analysed in further functions
                
            except requests.HTTPError as e: #create an alias for the generated HTTP exception so it is possible to  parse the error more thoroughly without having to call the requests functions multiple times
                if e.response.status_code == 404: #checks id the error is 404
                    logging.warning(f"Page not found (404): {url}") #logs in page not found error
                    return None
                elif e.response.status_code == 429: #checks if the error is the server requested to wairt longer in between requests
                    wait_time = 2 ** (attempt + 2)  # sets up a longer wait
                    logging.warning(f"Rate limited (429). Waiting {wait_time}s...") #logs in the event 
                    time.sleep(wait_time) # system call for longer wait before retrying
                else:
                    logging.error(f"HTTP Error {e.response.status_code}: {url}")  #logs in other http errors
                    
            except requests.RequestException as e: #catches other non-http errors
                logging.warning(f"Request error: {e}") # logs in the request error 
                
                if attempt < retries - 1: #check if we reached the maximum amount of retries before consifering backing off for longer
                    wait_time = 2 ** attempt  # Exponential backoff
                    logging.info(f"Waiting {wait_time}s before retry...") # logging in waiting time 
                    time.sleep(wait_time) #actual waiting
        
        logging.error(f"Failed to fetch {url} after {retries} attempts") # logs in failed to fetch page if all attemps don't work out
        return None #in this case, returns none
    
    def extract_article_data(self, article_elem) -> Optional[Dict]:
        """
        Extract data from a single article element
        
        Args:
            article_elem: BeautifulSoup element containing article data
            
        Returns:
            Dictionary with article data or None if extraction failed
        """
        try:
            data = {}
            
            # Extract publication date (e.g., "January 9, 2026")
            date_elem = article_elem.find(string=re.compile(r'\w+ \d+, \d{4}'))
            if date_elem:
                data['date_string'] = date_elem.strip()
                # Parse date
                try: #converts extracted dates in ISO format: from "January 9, 2026" to "2026-01-09"
                    data['publication_date'] = datetime.strptime(
                        data['date_string'], 
                        '%B %d, %Y'
                    ).date().isoformat()
                except ValueError:
                    data['publication_date'] = None
            else:
                data['date_string'] = None
                data['publication_date'] = None
            
            # Extract source publication (e.g., "Fortt Knox", "New York Times")
            # It appears after the date with a pipe separator
            source_elem = article_elem.find('a')
            if source_elem and source_elem.parent:
                text = source_elem.parent.get_text()
                parts = text.split('|')
                if len(parts) > 1:
                    data['source_publication'] = parts[1].strip()
                else:
                    data['source_publication'] = None
            else:
                data['source_publication'] = None
            
            # Extract article title
            title_elem = article_elem.find('h3')
            if title_elem:
                # Remove the date/source line
                title_link = title_elem.find('a')
                if title_link:
                    data['title'] = title_link.get_text(strip=True)
                else:
                    data['title'] = title_elem.get_text(strip=True)
            else:
                data['title'] = None
            
            # Extract external URL
            link_elem = article_elem.find('a', href=True)
            if link_elem:
                data['external_url'] = link_elem['href']
                # Classify link type
                if 'youtube.com' in data['external_url'] or 'youtu.be' in data['external_url']:
                    data['content_type'] = 'video'
                else:
                    data['content_type'] = 'article'
            else:
                data['external_url'] = None
                data['content_type'] = None
            
            # Extract thumbnail image
            img_elem = article_elem.find('img')
            if img_elem:
                data['thumbnail_url'] = img_elem.get('src', '')
            else:
                data['thumbnail_url'] = None
            
            # Add metadata
            data['scraped_at'] = datetime.now().isoformat()
            
            # Validation: must have at least title and URL
            if data['title'] and data['external_url']:
                return data
            else:
                logging.warning(f"Incomplete article data: {data}")
                return None
                
        except Exception as e:
            logging.error(f"Error extracting article data: {e}")
            return None
    
    def scrape_year(self, year: int) -> List[Dict]:
        """
        Scrape all articles from a specific year
        
        Args:
            year: Year to scrape (e.g., 2026, 2025, ...)
            
        Returns:
            List of article dictionaries
        """
        url = f"{self.news_url}?year={year}"
        logging.info(f"Scraping year {year}...")
        
        soup = self.fetch_page(url)
        if not soup:
            logging.error(f"Failed to load page for year {year}")
            return []
        
        articles = []
        
        # Find all article containers
        # Based on the HTML structure, articles appear to be in divs with images and links
        article_elements = soup.find_all('div', recursive=True)
        
        # Filter for elements that look like article containers
        # They should contain: date, publication name, title, and link
        potential_articles = []
        for elem in article_elements:
            # Check if element contains date pattern and a link
            if elem.find(string=re.compile(r'\w+ \d+, \d{4}')) and elem.find('a', href=True):
                potential_articles.append(elem)
        
        logging.info(f"Found {len(potential_articles)} potential articles for {year}")
        
        for article_elem in potential_articles:
            article_data = self.extract_article_data(article_elem)
            if article_data:
                article_data['year_scraped'] = year
                articles.append(article_data)
        
        logging.info(f"Successfully extracted {len(articles)} articles from {year}")
        return articles
    
    def scrape_all_years(self, start_year: int = 2020, end_year: int = 2026) -> pd.DataFrame:
        """
        Scrape articles from multiple years
        
        Args:
            start_year: First year to scrape (inclusive)
            end_year: Last year to scrape (inclusive)
            
        Returns:
            DataFrame with all scraped articles
        """
        all_articles = []
        years = range(start_year, end_year + 1)
        
        logging.info(f"Starting scraping from {start_year} to {end_year}")
        logging.info(f"Total years to scrape: {len(years)}")
        
        for i, year in enumerate(years, 1):
            logging.info(f"Progress: {i}/{len(years)} years")
            
            articles = self.scrape_year(year)
            all_articles.extend(articles)
            
            logging.info(f"Running total: {len(all_articles)} articles")
            
            # Extra delay between years to be respectful
            if i < len(years):
                time.sleep(self.delay * 2)
        
        # Create DataFrame
        df = pd.DataFrame(all_articles)
        
        if not df.empty:
            # Remove duplicates based on external URL
            initial_count = len(df)
            df = df.drop_duplicates(subset=['external_url'], keep='first')
            df = df.reset_index(drop=True)
            
            removed = initial_count - len(df)
            if removed > 0:
                logging.info(f"Removed {removed} duplicate entries")
            
            # Sort by date
            df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
            df = df.sort_values('publication_date', ascending=False)
            df = df.reset_index(drop=True)
            
            logging.info(f"Total unique articles scraped: {len(df)}")
        else:
            logging.warning("No articles were scraped!")
        
        return df
    
    def analyze_data(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics and insights from scraped data
        
        Args:
            df: DataFrame with scraped articles
            
        Returns:
            Dictionary with analysis results
        """
        if df.empty:
            return {"error": "No data to analyze"}
        
        analysis = {}
        
        # Temporal analysis
        df['year'] = pd.to_datetime(df['publication_date']).dt.year
        df['month'] = pd.to_datetime(df['publication_date']).dt.to_period('M')
        
        analysis['total_articles'] = len(df)
        analysis['date_range'] = {
            'earliest': df['publication_date'].min(),
            'latest': df['publication_date'].max()
        }
        analysis['articles_by_year'] = df['year'].value_counts().sort_index().to_dict()
        # Convert Period keys to strings for JSON serialization
        month_counts = df['month'].value_counts().sort_index().head(10)
        analysis['articles_by_month'] = {str(k): v for k, v in month_counts.items()}
        
        # Source analysis
        analysis['top_sources'] = df['source_publication'].value_counts().head(10).to_dict()
        analysis['unique_sources'] = df['source_publication'].nunique()
        
        # Content type analysis
        analysis['content_types'] = df['content_type'].value_counts().to_dict()
        
        # Title analysis
        df['title_length'] = df['title'].str.len()
        analysis['avg_title_length'] = float(df['title_length'].mean())
        
        # Keyword analysis (basic)
        all_titles = ' '.join(df['title'].dropna().str.lower())
        keywords = ['ai', 'gpu', 'gaming', 'data center', 'automotive', 'chip']
        analysis['keyword_mentions'] = {}
        for keyword in keywords:
            count = all_titles.count(keyword)
            if count > 0:
                analysis['keyword_mentions'][keyword] = count
        
        return analysis
    
    def save_data(self, df: pd.DataFrame, output_dir: str = 'data', 
                  include_analysis: bool = True):
        """
        Save scraped data to multiple formats with metadata
        
        Args:
            df: DataFrame with scraped data
            output_dir: Directory to save files (created if doesn't exist)
            include_analysis: Whether to generate and save analysis
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f'nvidia_newsroom_{timestamp}'
        
        # Save CSV (best for analysis in R, SPSS, Excel)
        csv_file = output_path / f'{base_name}.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logging.info(f"✓ CSV saved: {csv_file}")
        
        # Save JSON (best for programmatic access)
        json_file = output_path / f'{base_name}.json'
        df.to_json(json_file, orient='records', indent=2, force_ascii=False, date_format='iso')
        logging.info(f"✓ JSON saved: {json_file}")
        
        # Save Excel (best for sharing with non-technical collaborators)
        try:
            excel_file = output_path / f'{base_name}.xlsx'
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Articles', index=False)
                
                # Add analysis sheet if requested
                if include_analysis:
                    analysis = self.analyze_data(df)
                    analysis_df = pd.DataFrame([
                        {'Metric': k, 'Value': str(v)} 
                        for k, v in analysis.items() 
                        if not isinstance(v, dict)
                    ])
                    analysis_df.to_excel(writer, sheet_name='Summary', index=False)
            
            logging.info(f"✓ Excel saved: {excel_file}")
        except ImportError:
            logging.warning("openpyxl not installed. Skipping Excel export.")
            logging.warning("Install with: pip install openpyxl")
        
        # Save metadata and analysis
        metadata = {
            'scrape_timestamp': datetime.now().isoformat(),
            'total_articles': len(df),
            'date_range': {
                'earliest': str(df['publication_date'].min()),
                'latest': str(df['publication_date'].max())
            },
            'columns': list(df.columns),
            'source_url': self.news_url,
            'scraper_version': '1.0.0',
            'delay_seconds': self.delay
        }
        
        if include_analysis:
            metadata['analysis'] = self.analyze_data(df)
        
        metadata_file = output_path / f'{base_name}_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        logging.info(f"✓ Metadata saved: {metadata_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 SCRAPING COMPLETE!")
        print("="*60)
        print(f"Total articles collected: {len(df)}")
        print(f"Date range: {metadata['date_range']['earliest']} to {metadata['date_range']['latest']}")
        print(f"\nFiles saved in '{output_dir}/':")
        print(f"  • {csv_file.name}")
        print(f"  • {json_file.name}")
        if (output_path / f'{base_name}.xlsx').exists():
            print(f"  • {base_name}.xlsx")
        print(f"  • {metadata_file.name}")
        print("="*60 + "\n")


def main():
    """
    Main execution function
    
    Demonstrates how to use the scraper
    """

    print("\n" + "="*60)
    print("NVIDIA Newsroom Scraper")
    print("For Academic Research in Computational Social Science")
    print("="*60 + "\n")
    
    # Initialize scraper with 2-second delay (respectful)
    scraper = NVIDIANewsroomScraper(delay=2.0) #create and instance of the object defined above
    
    # Option 1: Scrape specific years
    print("Starting scraping process...")
    print("This will take several minutes to complete respectfully.\n")
    
    # Scrape from 2020 to 2026 (adjust as needed)
    df = scraper.scrape_all_years(start_year=2020, end_year=2026)
    
    # Save data with analysis
    scraper.save_data(df, output_dir='data/scraped_content', include_analysis=True)

    """
    # Display preview
    if not df.empty:
        print("\n📄 Preview of collected data:")
        print(df[['publication_date', 'source_publication', 'title']].head(10))
        
        print("\n📈 Quick insights:")
        print(f"  • Unique news sources: {df['source_publication'].nunique()}")
        print(f"  • Most covered year: {df['year_scraped'].mode()[0]}")
        print(f"  • Top source: {df['source_publication'].value_counts().index[0]}")
    """

if __name__ == "__main__":
    main()
