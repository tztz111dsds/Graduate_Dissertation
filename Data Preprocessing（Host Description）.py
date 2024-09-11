# Code Summary:
# This script processes Airbnb listing data by cleaning host descriptions, handling missing values, and filtering data 
# based on specific criteria. The main steps include:
#
# 1. Data Loading:
#    - The script reads a CSV file using ISO-8859-1 (latin1) encoding, selecting specific columns for analysis.
#    - The 'id' column is read as a string type.
#
# 2. Text Cleaning:
#    - The 'Host Description' column is cleaned by removing Chinese characters, non-ASCII characters, and extra spaces.
#    - Only English text and alphanumeric characters are retained.
#
# 3. Filtering:
#    - Host descriptions with fewer than 20 words are removed.
#    - Listings with missing prices are discarded.
#    - Numeric columns, such as review scores and number of reviews, are converted to numeric values.
#    - Listings with more than 20 reviews in the last 12 months ('number_of_reviews_ltm') are selected.
#    - Rows with missing values in the six review score dimensions are removed.
#    - Rows containing any Chinese characters are filtered out.
#
# 4. Saving:
#    - The cleaned data is saved to a new CSV file for further use.

import pandas as pd
import re

# Load Airbnb listing data, specifying ISO-8859-1 encoding
listings_df = pd.read_csv(r'D:\py\pythonProject\listings-detailed.csv', encoding='ISO-8859-1',
                          usecols=['id', 'description', 'price', 'review_scores_rating',
                                   'number_of_reviews', 'number_of_reviews_ltm',
                                   'review_scores_accuracy', 'review_scores_cleanliness',
                                   'review_scores_checkin', 'review_scores_communication',
                                   'review_scores_location', 'review_scores_value'],
                          dtype={'id': str})  # Force 'id' column to be read as string type

# Rename columns for easier processing
listings_df.rename(columns={'id': 'List ID', 'description': 'Host Description'}, inplace=True)

# Function to clean the 'Host Description' column - remove extra spaces, special characters, keep only English text
def clean_text(text):
    if isinstance(text, str):  # Only process string data
        text = re.sub(r'[\u4e00-\u9fff]', '', text)  # Remove Chinese characters
        text = re.sub(r'[^\x00-\x7F]', '', text)  # Remove non-ASCII characters (including Chinese punctuation)
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.strip()
    else:
        return ""

# Apply the cleaning function to the 'Host Description' column
listings_df['Host Description'] = listings_df['Host Description'].apply(clean_text)

# Remove rows where 'Host Description' has fewer than 20 words
listings_df = listings_df[listings_df['Host Description'].apply(lambda x: len(x.split()) >= 20)]

# Filter out records with missing price
listings_df = listings_df.dropna(subset=['price'])

# Convert review score columns to numeric, set invalid values to NaN
listings_df['review_scores_rating'] = pd.to_numeric(listings_df['review_scores_rating'], errors='coerce')
listings_df['review_scores_accuracy'] = pd.to_numeric(listings_df['review_scores_accuracy'], errors='coerce')
listings_df['review_scores_cleanliness'] = pd.to_numeric(listings_df['review_scores_cleanliness'], errors='coerce')
listings_df['review_scores_checkin'] = pd.to_numeric(listings_df['review_scores_checkin'], errors='coerce')
listings_df['review_scores_communication'] = pd.to_numeric(listings_df['review_scores_communication'], errors='coerce')
listings_df['review_scores_location'] = pd.to_numeric(listings_df['review_scores_location'], errors='coerce')
listings_df['review_scores_value'] = pd.to_numeric(listings_df['review_scores_value'], errors='coerce')
listings_df['number_of_reviews_ltm'] = pd.to_numeric(listings_df['number_of_reviews_ltm'], errors='coerce')

# Filter rows where 'number_of_reviews_ltm' (reviews in the last 12 months) is greater than 20
listings_df = listings_df[listings_df['number_of_reviews_ltm'] > 20]

# Remove rows with missing values in the six review score dimensions
listings_df = listings_df.dropna(subset=['review_scores_rating', 'review_scores_accuracy',
                                         'review_scores_cleanliness', 'review_scores_checkin',
                                         'review_scores_communication', 'review_scores_location',
                                         'review_scores_value'])

# Filter out rows containing Chinese characters in the 'Host Description' column
listings_df = listings_df[listings_df['Host Description'].apply(lambda x: not bool(re.search('[\u4e00-\u9fff]', x)))]

# Save the cleaned data to a new CSV file
listings_df.to_csv(r'D:\py\pythonProject\processed_listings.csv', index=False)
