# Code Summary:
# This script processes a large dataset of Airbnb reviews by cleaning the data, splitting it into smaller chunks, 
# saving these chunks, and then merging them back together for further processing. The main steps include:
#
# 1. Data Cleaning:
#    - The script reads a CSV file containing reviews.
#    - HTML tags are removed from the 'comments' column to clean up the text.
#
# 2. Data Splitting:
#    - The dataset is divided into 10 chunks of approximately equal size.
#    - Each chunk is processed and saved as a separate CSV file encoded in UTF-8.
#    - If the dataset size is not divisible by 10, the remaining rows are added to a final chunk.
#
# 3. Data Merging:
#    - The 10 individual CSV files are read back into pandas and concatenated to form a single dataset.
#    - The combined dataset is saved as a new CSV file.
#
# 4. Cleanup:
#    - After the data is merged, the individual chunk files are deleted from the system.

import os
import re
import pandas as pd
from langdetect import detect, LangDetectException

# Load the dataset
df = pd.read_csv("D:\\py\\pythonProject\\cleaned_Reviews.csv", encoding='utf-8')

# Function to remove HTML tags from text
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Apply the function to the 'comments' column
df['comments'] = df['comments'].apply(remove_html_tags)

# Split the data into 10 chunks and process each chunk separately
num_chunks = 10
chunk_size = len(df) // num_chunks
chunks = [df[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
if len(df) % num_chunks != 0:
    chunks.append(df[num_chunks*chunk_size:])

# Save each processed chunk to a separate CSV file with UTF-8 encoding
for i, chunk in enumerate(chunks):
    chunk.to_csv(f'cleaned_data_{i+1}.csv', index=False, encoding='utf-8')

# Merge all the processed chunks back into a single dataset
cleaned_data_total = pd.concat([pd.read_csv(f'cleaned_data_{i+1}.csv', encoding='utf-8') for i in range(len(chunks))])

# Save the merged dataset to a new CSV file with UTF-8 encoding
cleaned_data_total.to_csv('cleaned_data_total4.csv', index=False, encoding='utf-8')

# Delete the individual chunk files
for i in range(len(chunks)):
    os.remove(f'cleaned_data_{i+1}.csv')
