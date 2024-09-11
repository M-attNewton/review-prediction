# Restaurant Review Prediction Project

## Summary
This is a personal project to predict the review ratings (1-5) of resturants given the written review and additional information about the restuarant.

This project involved:
* Data cleaning: e.g. Removing stopwords, Lemmatization, Tokenization etc.
* Feature engineering: Creating additional features, One hot encoding, Scaling etc.
* Establishing a baseline model: Multi-class regression
* Experimenting with different Deep Learning methods: LSTMs, Transformers, Gradient Boosting Methods, Neural Networks
* Creating an ensemble machine learning model

## Data Cleaning
The main dataset is a very large collection of resuarant reviews (over two million samples). Consisting of the written review, review title, date, review ID, restaurant ID and rating (target label). There is an additional dataset containing details about the restaurant, consisting of resturant ID, name, price interval, rating and type.

The datasets are merged together on the restaurant ID to incorporate the restaurant data into each review sample. Feature engineering and cleaning is then performed on the dataset, for more details please see the Clean_Data notebook.

The dataset is cleaned using the Clean_Data notebook and is then saved to be reused in the other notebooks, this is to save time when training the machine learning models.

## Feature engineering
Additional features are created, such as the review length which could indicate a relationship with the rating. The Date is converted into integer values and the scaled using a Gaussian distribution along with the average restuarant rating and review length. In some models, the review title and full written review are combined into a single feature to improve the computation effeciency of the approaches. The Price interval is converted into an integer value and then parsed through a One hot encoder along with the resturant type to create the features to be used in the machine learning models.

The dataset is split into training, cross-validation and test sets, using a 70%, 15%, 15% split.

## Baseline model
Linear Multi-class Classification is used as the baseline model, using a large feature set will give an idea of a baseline performance. The text features are parsed through a term frequency–inverse document frequency (TF-IDF) Vectorizer to create the text features, these are combined with the restaurant type, price, date, review length and average restaurant rating.

This model achieves an accuracy of 71.5% which establishes our baseline.

## Deep learning models
Many different deep learning methods were tested, this repository highlights some of the models that were interesting to apply.

### Gradient Boosting Methods (LightGBM)
Since the dataset is tabular in nature, a tree based method with gradient boosting is a good model to try. The same features are used as in the linear case. LightGBM is used as the model as it is known to be better at handeling large datasets. The hyperparameters such as the number of leaves, number of estimators and learning rate are tuned to produce an accuracy score of 7X%. Which is similar to the linear model but, asdfadsf.

### Neural Networks
A neural network is a good model to try next due to its ability to handle both strucutred and unstructured data, in this case the tabular and text datatypes respectively. It is unlikely to outperform transformers or LSTMs as it will not be able to capture the context of the sentences as well but it can easily incorporate the other data types into the model. TF-IDF Vectorizer is used as in the linear model, a word embedding such as GloVe or Word2Vec would be more effective but more computationally expensive. Dropout and regularisation techniques are used to avoid overfitting. The neural network hyperparameters are tuned such that it can be run efficiently on a CPU and an accuracy score of 7X% in achieved. asdfadsf

### Transformers


## Ensemble model 

### Linear classification and LightGBM (CPU)

### LSTM and Neural Network (GPU)
To better capture the meaning in the text data, a Long Short-Term Memory (LSTM) network can be used and can be more efficient to train than transformers. They can also be trained alongside another neural network that captures the catergorical and numerical features, creating an ensemble model. An pre-trained embedding layer using GloVe is defined followed by an LSTM layer, the output of this layer is concatenated with the catergorical and numerical features and parsed through a fully-connected layer, a softmax function is used to create the output layer of the network. Dropout and regularisation techniques are used to avoid overfitting.

This model is much larger and hence needs to be run on a GPU, these resources were limited so only a handful of experiments were run. An accuracy score of 73.3% was achieved and likely could be increased with greater resources and a larger model. 


## Future work


