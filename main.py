import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


yelp = pd.read_csv('./datasets/yelp.csv')
yelp['text length'] = yelp['text'].apply(len)
yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    # train on TF-IDF vectors w/ Naive Bayes classifier
    ('classifier', MultinomialNB()),
])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
