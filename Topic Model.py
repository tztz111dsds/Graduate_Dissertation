
# Code Summary:
# This section of code demonstrates a typical NLP workflow for text data preprocessing, topic modeling using LDA,
# and saving the results to a CSV file. The main steps include:
#
# 1. Data Preprocessing:
#    - Text is cleaned by removing non-ASCII characters, Chinese characters, punctuation, and numbers.
#    - Tokenization is performed, and stopwords are removed.
#
# 2. Dictionary and Corpus Creation:
#    - A dictionary is created from the tokenized text.
#    - A bag-of-words representation (corpus) is built for the LDA model.
#
# 3. LDA Model Training:
#    - The LDA (Latent Dirichlet Allocation) model is trained to identify 15 topics from the preprocessed text.
#    - The model is run for 10 passes with 100 iterations for optimal topic extraction.
#
# 4. Visualization:
#    - The results are visualized using pyLDAvis and saved as an HTML file.
#
# 5. Saving Topic Results:
#    - The topics and their associated keywords are extracted from the model.
#    - The topics and keyword distributions are saved as a CSV file for further analysis.


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# 停用词
import nltk
# Download the necessary stopwords from NLTK and extend the stopword list with additional common words.
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
additional_stopwords = ['and', 'br', 'to', 'a', 'with', 'in', 'the', 'of', 'is', 'from', 'for', 'by', 'are']
stop_words.update(additional_stopwords)

# Function for data preprocessing
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[\u4e00-\u9fff]', '', text)  # Remove Chinese characters
        text = re.sub(r'[^\x00-\x7F]', '', text)  # Remove non-ASCII characters (including Chinese punctuation)
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove all numbers
        text = re.sub(r'<br>', ' ', text, flags=re.IGNORECASE)  # Remove "<br>" symbols
        tokens = word_tokenize(text)  # Tokenize the text
        return [word for word in tokens if word.lower() not in stop_words]  # Remove stopwords
    else:
        return []

# Read the preprocessed data
listings_df = pd.read_csv(r'D:\py\pythonProject\processed_listings.csv')
listings_df['Host Description'] = listings_df['Host Description'].apply(clean_text)

# Create a dictionary and bag-of-words representation of the data
dictionary = corpora.Dictionary(listings_df['Host Description'])
dictionary.filter_extremes(no_below=5)  # Filter out terms that occur less than 5 times
corpus = [dictionary.doc2bow(text) for text in listings_df['Host Description']]

# Train the LDA model
num_topics = 15  # You can adjust the number of topics based on your needs
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, iterations=100, random_state=42)

# Visualize the results
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, r'visualization_new_1.html')  # Save the visualization as an HTML file

# Extract topics and keywords, and save them as a CSV file
topics = lda_model.show_topics(formatted=False, num_words=20)
topic_data = []
for topic_num, words in topics:
    for word, weight in words:
        topic_data.append([topic_num, word, weight])

# Convert to DataFrame and save
df = pd.DataFrame(topic_data, columns=['Topic', 'Keyword', 'Weight'])
df.to_csv('keyword_distribution.csv', index=False)
print("Keyword distribution has been saved to 'keyword_distribution.csv'")


