# Sentiment Analysis Theory for Reddit News Titles

This Python script performs sentiment analysis on news titles collected from a specified subreddit (r/politics in this script) using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. The script utilizes the PRAW library for Reddit API interaction, the NLTK library for natural language processing tasks, and the Pandas library for data manipulation.

## Prerequisites

Before running the script, ensure you have the following libraries installed:

1. PRAW (Python Reddit API Wrapper)
2. NLTK (Natural Language Toolkit)
3. Pandas
4. NumPy
5. WordCloud
6. Schedule

You can install these libraries using the following command: 
```bash
pip install praw nltk pandas numpy wordcloud schedule
```

## Script Overview

1. Import Required Libraries: Import the necessary Python libraries, including datetime, numpy, praw, pprint, pandas, nltk, wordcloud, schedule, and os.
```bash
import datetime as dt
import numpy as np
import praw
from pprint import pprint
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, RegexpTokenizer  # tokenize words
from nltk.corpus import stopwords
from itertools import chain
from wordcloud import WordCloud, STOPWORDS
import schedule
import time
import os
```
2. Download Lexicons and Tokenizers: Download the VADER lexicon data for sentiment analysis, as well as the necessary tokenizers and stopwords from NLTK.
```bash
# Downloading VADER Date for Sentiment Analysis
nltk.download('vader_lexicon')  # get lexicons data
nltk.download('punkt')  # for tokenizer
nltk.download('stopwords')
```
3. Reddit API Configuration: Set up your Reddit API credentials (client ID, client secret, user agent) to authenticate the script to access Reddit data using the PRAW library.
```bash
# Create an instance of Reddit
reddit = praw.Reddit(
    client_id="your_client_id",
    client_secret="your_client_secret",
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)
```
4. Scrape Data: Define a function scrape_data() that scrapes news data from a specified subreddit using the Reddit API. It collects information like news titles, dates, upvotes, and upvote ratios.
```bash
def scrape_data():
    # Scraping Data from the Subreddit of Choice
    subreddit = reddit.subreddit('politics')
    news = [*subreddit.new(limit=None)]  # top posts all time
    print(len(news))

    # Create lists of the information from each news
    title = [news.title for news in news]
    date = [dt.datetime.fromtimestamp(news.created) for news in news]
    upvotes = [news.score for news in news]
    upvotes_percentage = [news.upvote_ratio for news in news]

    # Creating DataFrame
    news = pd.DataFrame({
        "title": title,
        "date": date,
        "upvotes": upvotes,
        "upvotes_percentage": upvotes_percentage,
    })
    return news
```
5. Perform Sentiment Analysis: Define a function perform_sentiment_analysis(news) that calculates the sentiment polarity of each news title using the VADER sentiment intensity analyzer. The results are appended to the DataFrame.
```bash
def perform_sentiment_analysis(news):
    # Calculating Polarity of the Titles
    sid = SentimentIntensityAnalyzer()
    res = [*news['title'].apply(sid.polarity_scores)]
    pprint(res[:3])

    # Combining the Sentiment Analysis and the Scraped Data in One DataFrame
    sentiment_df = pd.DataFrame.from_records(res)
    news = pd.concat([news, sentiment_df], axis=1, join='inner')
    return news
```
6. Label Posts: Define a function label_posts(news) that labels each news title as positive, negative, or neutral based on a custom threshold value applied to the compound sentiment score.
```bahs
def label_posts(news):
    # Setting Post as Pos, Neg, or Neu based on a Custom set Threshold Value
    THRESHOLD = 0.2
    conditions = [
        (news['compound'] <= -THRESHOLD),
        (news['compound'] > -THRESHOLD) & (news['compound'] < THRESHOLD),
        (news['compound'] >= THRESHOLD),
    ]
    values = ["neg", "neu", "pos"]
    news['label'] = np.select(conditions, values)
    return news
```
7. Save Data to CSV: Define a function save_to_csv(news_data) that saves the news data, including sentiment analysis results and labels, to a CSV file named reddit_news_data.csv.
```bash
def save_to_csv(news_data):
    # File name for the CSV
    csv_filename = "reddit_news_data.csv"
    
    # Check if the CSV file already exists
    if not os.path.exists(csv_filename):
        # If the CSV file doesn't exist, save the DataFrame as a new CSV
        news_data.to_csv(csv_filename, index=False)
        print("Data saved to CSV (New file):", csv_filename)
    else:
        # If the CSV file exists, read the existing data
        existing_data = pd.read_csv(csv_filename)
        
        # Concatenate the existing data with the new data
        combined_data = pd.concat([existing_data, news_data])
        
        # Save the updated DataFrame to the CSV file
        combined_data.to_csv(csv_filename, index=False)
        print("Data saved to CSV (Updated file):", csv_filename)
```
8. Hourly Task: Define a function hourly_task() that orchestrates the complete process. It collects news data, performs sentiment analysis, labels posts, and saves the data to the CSV file. This function is scheduled to run every hour using the schedule library.
```bash
def hourly_task():
    now = dt.datetime.now()
    print("Starting hourly dat  a analysis...", now.strftime("%d/%m/%Y %H:%M:%S"))
    news_data = scrape_data()
    if not news_data.empty:
        news_data = perform_sentiment_analysis(news_data)
        news_data = label_posts(news_data)
        #print(news_data.head())
        # Save the DataFrame to CSV
        save_to_csv(news_data)
```
9. Run Scheduler: Run the scheduler in an infinite loop to execute the hourly_task() function periodically.
```bash
# Schedule the hourly_task to run every hour
schedule.every().hour.do(hourly_task)

# Run the scheduler indefinitely
while True:
    schedule.run_pending()
    time.sleep(1)
```
## Usage
1.	You can configure your Reddit API credentials by replacing the placeholders with your actual credentials: client_id, client_secret, and user_agent.
2.	Specify the subreddit you want to analyze by replacing 'politics' with the desired subreddit name.
3.	Define and adjust the sentiment analysis threshold (THRESHOLD) in the label_posts(news) function to customize the labeling of news titles as positive, negative, or neutral.
4.	Run the script. It will collect news data from the specified subreddit, perform sentiment analysis, label the posts, and save the data to the CSV file.

## Note
•	This script provides a basic implementation of sentiment analysis for Reddit news titles using VADER. Depending on your analysis's specific context and goals, you may want to explore more advanced sentiment analysis techniques and other natural language processing tools.

•	Remember that sentiment analysis may not always be accurate due to the complexity of language and context. Manual validation and analysis of the results are recommended

•	Reddit API usage should follow Reddit's terms of service and API guidelines. Make sure you are familiar with Reddit's API rules before deploying this script on a larger scale.






