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

See the EDA notebook. One important point was removing all rows with zeroes for five numeric features (ie CPC), accounting for 42% of the dataset. All decisions in data cleaning and transformation are explained there

2. How did you perform NLP, if any?

I tried various word embedding algorithms and settled with ELMo as it performed better than word2vec, GloVe and also the CNN-based model of SpaCy. The embedding functions take in keyword strings then output embedding vectors, which are fed to the regressor. More details in the second notebook.

3. How did you model the problem, and why?

As discussed in the EDA notebook, we have a multi-output regression problem. The input features are Date, Market, CPC and the keyword as a vector embedding. The output features are CTR, clicks, impressions, cost and average position. Since the values are related by simple formulas, we only need to construct a regressor for three features: CTR, clicks and average position. Our chosen machine learning model for the regressor are CatBoost gradient boosted trees as there are a few categorical input features.

4. How did you evaluate your model? What were the results of the evaluation?

Since there are three regressors, three sets of explained variance scores and mean absolute errors were computed. The closer to 1.0 the explained variance score is, the better the model is. For model selection, 5-fold cross-validation was performed for each regressor after having chosen hyperparameters with RandomSearch and GridSearch. The constructed CatBoost regressor for CTR had a score of 0.87 and the regressor for Clicks had 0.94 which means the model predicts CTR and Clicks very well. The regressor for AveragePosition had a score of only 0.71, which is not as good as the other two. This may be a result of the fact that 95% of the data points had exactly 1.0 AveragePosition which will make it difficult to predict for any machine learning model, not to mention decision tree-based models.

5. If you had extra time, what would you do next?

Some points for improvement are:
- The embedding functions generate a vector for each word, but for each keyword (which usually consists of multiple words) I only took the averages over each of the 768 dimensions. A lot of information is lost when averaging. The biggest performance boost would probably come from constructing an attention-based LSTM recurrent neural network that receives a sequence of strings (the search keyword) and outputs a vector embedding. And then this could be connected to a simple multilevel-perceptron that converts that embedding into a prediction for CTR. Together the RNN plus the MLP is trained as a regressor using the keywords as input and the CTR values as targets, then the MLP is removed and the outputs of the RNN are used for generating vector embeddings.
- More vector embedding methods could have been tried. For example the larger ELMo embedding function could have been used, as it generates vectors of dimension greater than 2000.
- More ML models beyond CatBoost could have been tried. Generalized Linear models or Kernel-based methods might be able to perform better seeing that the problem has a high number of dimensions and also a large number of training samples
- Bayesian optimization has not yet been applied on the ML model
- Further feature engineering could be done beyond the choice of vector representation of the keyword. In a real-world situation, augmenting the predictor feature set would significantly help the predictive power of the resulting ML model. For example adding a column for time of day, season (winter/summer/etc). Other external events affect the trending keyword searches so perhaps extra features could be added that give the predictor extra context for a keyword.
