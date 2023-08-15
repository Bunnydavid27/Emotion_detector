# Emotion_detector

![Screenshot (337)](https://github.com/Bunnydavid27/Emotion_detector/assets/95872007/93b1b6a6-0110-4e02-b394-97cb9696d7ca)


                                                                 Text Emotion Detector 

Emotion Emoji Detector is a system that uses Machine Learning (ML) models and Natural Language Processing (NLP) to automatically recognise and correlate specific emotions and corresponding emojis in text. ML algorithms and NLP approaches are used to analyse textual material and predict the matching emotion emojis that best capture the underlying sentiment or feeling portrayed in the text.
DATA GATHERING AND PREPROCESSING : I gathered a dataset of text samples from website texts, as well as a Twitter sentiment dataset from Kaggle, and then I set the column names to be the same. Concatenated both datasets, then cleaned the dataset by removing stopwords, special characters, and userhandles. Tokenize, clean, and convert the text data into a format appropriate for machine learning models. coupled with their corresponding emotion-related emojis.Preprocessed text data is used to extract relevant features.This uses the Bag of Words approach with the Count Vectorizer approach.
MODEL SELECTION AND DEVELOPMENT: Logistic regression, fasttext, and xgbooster are the ML models used for these classification tasks. Using the labelled and annotated data, train the chosen model.
MODEL EVALUATION: Using validation and test datasets, assess the trained model's performance. To assess how successfully the model predicts emotion emojis, metrics such as accuracy, precision, recall, and F1-score are employed. Among the methods, the logistic regression model ranks reasonably high.
DEPLOYEMENT AND CONTINUOUS INTEGRATION : The trained model was integrated into a streamlit application and placed on the streamlit platform, where users could enter text, and the system would be online and tested by users, as well as automatically generate the relevant emotion emoji. It integrates with the Github repository to update the deployed application



