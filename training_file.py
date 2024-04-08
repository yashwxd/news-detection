import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# load dataset
df = pd.read_csv('train.csv')
print(df.head())

# we will only Going to use title and author Columns for Our prediction
df.drop(['id','text'],axis=1,inplace=True)
print(df.head())

####Data Cleaning¶


# Check for Null Values
df.isnull().sum()

# Drop Null values
df = df.dropna()

# Check for Duplicated Values
print(df.duplicated().sum())

# Drop Duplicated Values
df = df.drop_duplicates(keep='first')
# Merge both Columns Author and Title and Create New Column Content
df['content'] = df['title'] + ' ' + df['author'] 
print(df.head())


# import required libaries for preprocessing
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

# Function for entire text transformation

def text_preprocessing(text): 
    # Convert text into lowercase
    text = text.lower()
    
    # Tokenize text into list
    tokenize_text = nltk.word_tokenize(text)
    
    # remove Stopwords
    text_without_stopwords = [i for i in tokenize_text if i not in stopwords.words('english')]
    
    # Remove Punctuation
    text_without_punc = [i for i in text_without_stopwords if i not in string.punctuation]
    
    # fetch only alphanumeric values and apply stemming on that word
    transformed_text = [ps.stem(i) for i in text_without_punc if i.isalnum() == True]
    
    return " ".join(transformed_text)
# Let's Apply This Transformation Function on Our Content Column
df['transformed_content'] = df['content'].apply(text_preprocessing)
# Drop title author and old content column
df = df.drop(['title','author','content'],axis=1)
print("==================================")
print(df.head())
df.to_csv("new_train_dataset.csv")
# Check Count of labels
sns.countplot(x='label',data=df)
# import wordcloud
from wordcloud import WordCloud

# make object of wordcloud
wc = WordCloud(background_color='white',min_font_size=10,width=500,height=500)
# WordCloud for True News
true_news_wc = wc.generate(df[df['label'] == 0]['transformed_content'].str.cat(sep=" "))
plt.figure(figsize=(8,6))
plt.imshow(true_news_wc)
plt.show()

# WordCloud for Fake news
fake_news_wc = wc.generate(df[df['label'] == 1]['transformed_content'].str.cat(sep = " "))
plt.figure(figsize=(8,6))
plt.imshow(fake_news_wc)
plt.show()