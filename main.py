# %%writefile my_streamlit_app.py
import nltk
import tensorflow as tf
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import googleapiclient.discovery
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import text
# from keras.utils import np_utils
from keras.models import Sequential

from tensorflow.keras.models import load_model


class Data_Preparation():
    def __init__(self):
        print('Data Preparation Object Created')
    def __fetch_sentiment_using_textblob(self,text):
        analysis = TextBlob(text)
        return 'pos' if analysis.sentiment.polarity > 0 else 'neg'
    def Concat_Sentiments_to_DataFrame(self,data):
        sentiments_using_textblob = data.Comment.apply(lambda tweet: self.__fetch_sentiment_using_textblob(tweet))
        print("Sentiments Value Counts ::",pd.DataFrame(sentiments_using_textblob.value_counts()))
        data['sentiment'] = sentiments_using_textblob
        return data
    def Text_Processing(self,data):
        data =data.applymap(lambda s:s.lower() if type(s) == str else s)
        data = data[data['Comment']!='']
        data = data.drop_duplicates(subset=['Comment'], keep=False)
        data = data.reset_index(drop=True)
        data['Comment'] = data['Comment'].str.replace("[^a-zA-Z# ]", "")
        data = data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
        data = data[data['Comment']!='']
        data = data.drop_duplicates(subset=['Comment'], keep=False)
        data = data.reset_index(drop=True)
        data['Comment'] = data['Comment'].str.replace("[^a-zA-Z# ]", "")
        return data
    def Tokennizing_tweets(self,data):
        tokenized_tweet = data['Comment'].apply(lambda x: x.split())
        word_lemmatizer = WordNetLemmatizer()
        tokenized_tweet = tokenized_tweet.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])
        for i, tokens in enumerate(tokenized_tweet):
            tokenized_tweet[i] = ' '.join(tokens)
        data['Comment'] = tokenized_tweet
        data = data[['Comment','sentiment']]
        return data
    def __wordcloud(self,x):
        plt.figure(figsize=(14, 10))
        wordcloud = WordCloud(width = 1000, height = 500).generate(" ".join(x))
        plt.imshow(wordcloud)
        plt.axis("off")
        return wordcloud
    def Comments_WordCloud(self,data):
        all_words = ' '.join([text for text in data['Comment'][data.sentiment == 'pos']])
        all_words = all_words.split()
        self.__wordcloud(all_words)

class DataFrame_Preprocessor():
    def __init__(self):
        print("Preprocessor object created")
    def preprocess(self,df):
        df['sentiment'] = np.where(df['sentiment'] == 'pos', 1, 0)
        x = df['Comment']
        y = df['sentiment']
        return train_test_split(x,y,test_size=1, random_state=0)
class Keras_Tokenizer():
    def __init__(self,max_features):
        self.max_features =6000
        print("Tokenizer object created")
    def __label_encoding(self,y_train):
        """
        Encode the given list of class labels
        :y_train_enc: returns list of encoded classes
        :labels: actual class labels
        """
        lbl_enc = LabelEncoder()
        y_train_enc = lbl_enc.fit_transform(y_train)
        labels = lbl_enc.classes_
        return y_train_enc, labels
    def __word_embedding(self,train, test, max_features, max_len=200):
        try:
            #""" Keras Tokenizer class object """
            tokenizer = text.Tokenizer(num_words=max_features)
            tokenizer.fit_on_texts(train)

            train_data = tokenizer.texts_to_sequences(train)
            test_data = tokenizer.texts_to_sequences(test)

            #""" Get the max_len """
            vocab_size = len(tokenizer.word_index) + 1

            #""" Padd the sequence based on the max-length """
            x_train = sequence.pad_sequences(train_data, maxlen=max_len, padding='post')
            x_test = sequence.pad_sequences(test_data, maxlen=max_len, padding='post')
            #""" Return train, test and vocab size """
            return tokenizer, x_train, x_test, vocab_size
        except ValueError as ve:
            raise(ValueError("Error in word embedding {}".format(ve)))
    def preprocess(self,X_train, X_test):
        return self.__word_embedding(X_train, X_test, self.max_features)


# Function to display word cloud
def display_wordcloud(data):
    all_words = ' '.join([text for text in data['Comment'][data.sentiment == 'pos']])
    all_words = all_words.split()
    plt.figure(figsize=(14, 10))
    wordcloud = WordCloud(width=1000, height=500).generate(" ".join(all_words))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot()

# Streamlit UI
def main():
    st.title("YouTube Sentiment Analysis")

    # Background Image
    st.markdown(
        '<style>body {background-image: url("your_background_image_url"); background-size: cover;}</style>',
        unsafe_allow_html=True,
    )

    # Input Fields
    youtube_url = st.text_input("YouTube Video URL")
    youtube_video_id = st.text_input("YouTube Video ID")

    # Button to trigger sentiment analysis
    if st.button("Analyze Sentiment"):
        # Fetch comments and perform sentiment analysis
        api_service_name = "youtube"
        api_version = "v3"
        DEVELOPER_KEY = "AIzaSyDr-FunKsjJtw_YS8Gw_GGw_DSVke5IJHg"

        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey=DEVELOPER_KEY)

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=youtube_video_id,  # Use the input video ID
            maxResults=1000
        )
        response = request.execute()
        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                comment['textDisplay']
            ])
        df = pd.DataFrame(comments, columns=['Name', 'published_at', 'updated_at', 'like_count', 'Comment'])
        columns_to_drop = ['published_at', 'updated_at', 'like_count']
        df = df.drop(columns=columns_to_drop, axis=1)

        # Continue with your existing sentiment analysis code
        DP = Data_Preparation()
        polarity=[]
        df = DP.Concat_Sentiments_to_DataFrame(df)
        df = DP.Text_Processing(df)
        df = DP.Tokennizing_tweets(df)

        # Display word cloud
        # display_wordcloud(df)

        # Display sentiment distribution
        PR = DataFrame_Preprocessor()
        X_train, X_test, y_train, y_test = PR.preprocess(df)
        X_train=df['Comment']
        X_test=df['Comment']
        Y_test=df['sentiment']
        Y_train=df['sentiment']
        KT = Keras_Tokenizer(6000)
        tokenizer, x_pad_train, x_pad_valid, vocab_size = KT.preprocess(X_train, X_test)
        rnn_model = load_model("C:/Users/Dell/Desktop/Major_project/rnn_model86.h5")

        y_preds = rnn_model.predict(x_pad_valid)

        for arr in y_preds:
            for i in range(len(arr)):
                if arr[i] > 0.5:
                    arr[i] = 1
                else:
                    arr[i] = 0

        y_preds = y_preds.astype('int32')

        pred_df = pd.DataFrame(y_preds, columns=['pred'])

        pred_df['pred'] = pred_df['pred'].replace({1: 'Negative', 0: 'Positive'})
        class_counts = pred_df['pred'].value_counts()
        labels = class_counts.index
        sizes = class_counts.values

        plt.figure(figsize=(8, 4))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#AEDFF7', '#FF6F61'])
        plt.title('Pie Chart of Sentiment Distribution')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

if __name__ == "__main__":
    main()
