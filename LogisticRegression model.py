import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('train.csv')
X=df.drop(['id','keyword','location','target'],1)
y=df.drop(['id','keyword','location','text'],1)
vectorizer = CountVectorizer(
    analyzer = 'word')
features = vectorizer.fit_transform(
    X['text']
)

features_nd = features.toarray() # for easy usage
X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        y,
        train_size=0.80, 
        random_state=1234)

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)

print(accuracy_score(y_test, y_pred))
