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


# Downloading VADER Date for Sentiment Analysis
nltk.download('vader_lexicon')  # get lexicons data
nltk.download('punkt')  # for tokenizer
nltk.download('stopwords')

# Create an instance of Reddit
reddit = praw.Reddit(
    client_id="WWtak5fAYhMJ1ajR6gt6fw",
    client_secret="2l-3vh8F3iX5V844C_Ndgqn8BS546Q",
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)

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

def perform_sentiment_analysis(news):
    # Calculating Polarity of the Titles
    sid = SentimentIntensityAnalyzer()
    res = [*news['title'].apply(sid.polarity_scores)]
    pprint(res[:3])

    # Combining the Sentiment Analysis and the Scraped Data in One DataFrame
    sentiment_df = pd.DataFrame.from_records(res)
    news = pd.concat([news, sentiment_df], axis=1, join='inner')
    return news

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
        

# Schedule the hourly_task to run every hour
schedule.every().hour.do(hourly_task)

# Run the scheduler indefinitely
while True:
    schedule.run_pending()
    time.sleep(1)