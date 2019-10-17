# ctr-prediction

install the requirements first
```
$ conda create --name new_env --file requirements.txt
```

Then run the two notebooks in succession. Afterwards you can now use the function Predictor(Date, Market, Keyword, CPC) defined in the second notebook.

## Problem Description

An advertising company sells a service of buying keywords in search engines on behalf of their customers. They’re trying to optimise their keyword and funds allocation. The first towards the optimal solution is to predict performance by keyword and fund.

Goal:

Predicting for any keyword (not necessarily the ones in the dataset file), CPC, and market (US/UK) the traffic a website would receive (I.e., the clicks).

Task Evaluation:

The evaluation of the task will use an input dataset of new keywords and CPC for each market (US/UK) at the date of 14/2/2013. The model's results will be compared to real results for that day.

## Questions

1. How did you manipulate the data, and why? Illustrate your answer with plots.

See the EDA notebook. One important point was removing all rows with zeroes for five numeric features (ie CPC), accounting for 42% of the dataset. 

2. How did you perform NLP, if any?

I tried various word embedding algorithms and settled with ELMo as it performed better than word2vec, GloVe and also the CNN-based model of SpaCy. The embedding functions take in keyword strings then output embedding vectors, which are fed to the regressor. More details in the second notebook.

3. How did you model the problem, and why?

As discussed in the EDA notebook, we have a multi-output regression problem. The input features are Date, Market, CPC and the keyword as a vector embedding. The output features are CTR, clicks, impressions, cost and average position. Since the values are related by simple formulas, we only need to construct a regressor for three features: CTR, clicks and average position.

4. How did you evaluate your model? What were the results of the evaluation?

Since there are three regressors, three sets of explained variance scores and mean absolute errors were computed. The closer to 1.0 the score, the better the model is. For model selection, 5-fold cross-validation was performed for each regressor after having chosen hyperparameters with RandomSearch and GridSearch.

5. If you had extra time, what would you do next?

Some points for improvement are:
- The embedding functions generate a vector for each word, but for each keyword (which usually consist of multiple words) I only took the averages over each of the 772 dimensions. A lot of information is lost when averaging. The biggest performance boost would probably come from constructing an LSTM neural network
- More vector embedding methods could have been tried. For example the larger ELMo embedding function could have been used, as it generates vectors of dimension greater than 2000. More hyperparameter tuning could have been performed
- More ML models beyond CatBoost could have been tried. Generalized Linear models might be able to perform better seeing has a high number of dimensions and also a large number of training samples
