# Disaster Response Pipeline

### Overview:
This project is aimed at analyzing messages sent by the people who need help during a disaster. Often, disaster response agencies receive so many requests through various sources such as direct, news and social during a disaster that they are unable to respond timely.

One of the major problems is identifying the right authority to send the request to. This web app can help the disaster agencies and citizens to find the right emergency/disaster theme real-time so that the request can be directed to the right authority and can be responded in a timely manner.

### Files:
The repository contains all the necessary files in three folders -

* Data (ETL phase) - It contains following files:
  * process_data.py - reads in the data, cleans and stores it in a SQL database.
  * disaster_categories.csv and disaster_messages.csv (dataset)
  * DisasterResponse - database created after you run the process_data.py

* Models (Pipeline & Modelling phase) - It contains following files:
  * train_classifier.py: loads data, transform it using NLTK, run a machine * learning algorithm using GridSearchCV and fit the training data
  * classifier.pkl - Pickle file created after you run the train_classifier.py

* App (Web app) - It contains following files:
  * run.py: Flask app and the user interface to view the classification results
  * templates: folder containing html files (master.html and go.html)


### Pipeline & App:
This web app (made using Flask) uses real disaster messages data from Figure Eight. I created a machine learning pipeline to categorize these messages using several machine learning models and NLTK. The app also displays visualization of the training data.


### Instructions:
1. To run the project in your local server, clone the repository on your desktop. Then use the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
