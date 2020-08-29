

import pandas as pd
msgs = pd.read_csv('D:\ML\data\smsSpamCollection',sep='\t',names=['label','message'])

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

wordNet = WordNetLemmatizer()
corpus =[]
for i in range(len(msgs)):
    review = re.sub('[^a-zA-Z]',' ',msgs['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordNet.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(msgs['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state =0)

from sklearn.naive_bayes import MultinomialNB
spam_model = MultinomialNB().fit(X_train,y_train)

y_pred = spam_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test,y_pred)
confusion_mat

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)
score
