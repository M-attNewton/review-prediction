## Restaurant Review Prediction Project

# Summary
This is a personal project to predict the review ratings (1-5) of resturants given the written review and additional information about the restuarant.

This project involved:
* Data cleaning: e.g. Removing stopwords, Lemmatization, Tokenization etc.
* Feature engineering: Creating additional features, One hot encoding, Scaling etc.
* Establishing a baseline model: Multi-class regression
* Experimenting with different Deep Learning methods: LSTMs, Transformers, Gradient Boosting Methods, Neural Networks
* Creating an ensemble machine learning model

# Data Cleaning
The main dataset is a very large collection of resuarant reviews (over two million samples). Consisting of the written review, review title, date, review ID, restaurant ID and rating (target label). There is an additional dataset containing details about the restaurant, consisting of resturant ID, name, price interval, rating and type.

The datasets are merged together on the restaurant ID to incorporate the restaurant data into each review sample. Feature engineering and cleaning is then performed on the dataset, for more details please see the Clean_Data notebook.

The dataset is cleaned using the Clean_Data notebook and is then saved to be reused in the other notebooks, this is to save time when training the machine learning models.

# Feature engineering
Additional features are created, such as the review length which could indicate a relationship with the rating. The Date is converted into integer values and the scaled using a Gaussian distribution along with the average restuarant rating and review length. In some models, the review title and full written review are combined into a single feature to improve the computation effeciency of the approaches. The Price interval is converted into an integer value and then parsed through a One hot encoder along with the resturant type to create the features to be used in the machine learning models.

# Baseline model

# Deep learning models

# Ensemble model
