import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier


def load_data(database_filepath):
    """Loads data from filepath
    
    Parameters:
    database_filepath (str): link to the database file path

    Returns:
    X, y, category names: features, target, target variable names 
    
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("disaster_messages", con=engine) 
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    y = y.fillna(0)
    return X, y, category_names


def tokenize(text):
    
    """Tokenizes the data
    Parameters:
    text: text/string

    Returns:
    clean_tokens: cleaned token
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    
    stop_words = set(stopwords.words('english')) 
    tokens = [w for w in words if not w in stop_words]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)


    return clean_tokens


def build_model():
    """Returns Grid Search model with AdaBoostClassifier"""
     
    pipeline_ab = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'tfidf__use_idf':[True, False]
    }

    cv_ab = GridSearchCV(pipeline_ab, param_grid=parameters)
    return cv_ab

def evaluate_model(model, X_test, y_test, category_names):
    
    """Print model results
    Parameters:
    model - estimator-object/algorithm
    X_test - Feature test data
    y_test - Target test data
    category_names = required, list of category strings
    
    Returns: Classification report

    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names = category_names))


def save_model(model, model_filepath):
    
    """Save model as pickle file
    
    Parameters:
    model - estimator-object/algorithm
    model_filepath - filepath to save the pickle file
    
    Returns: Saves the pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
     """Loads the data, run the model and save the model"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()