import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Reading dataset as pandas dataframe
yelp = pd.read_csv('./datasets/yelp.csv')

# total number of reviews
print("Total number of reviews", len(yelp))

yelp['text length'] = yelp['text'].apply(len)

# data cleaning
yelp['text'] = [i.replace("&amp;amp;", '').replace("\'", '')
                for i in yelp['text']]

# Taking one star rating as negative review and 5 star rating as negative preview
data_classes = yelp[(yelp['stars'] == 1) | (
    yelp['stars'] == 3) | (yelp['stars'] == 5)]


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    # train on TF-IDF vectors w/ Naive Bayes classifier
    ('classifier', MultinomialNB()),
])

X = data_classes['text']
y = data_classes['stars']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Example prediction
example_prediction = pipeline.predict(
    ['This is good restaurant, i want to come here again'])
print(example_prediction)

# example_prediction = pipeline.predict(
#     ['disgusting food, worst service, I will not come here again, and do not recommend to others'])
# print(example_prediction)
