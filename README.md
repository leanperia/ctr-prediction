# ctr-prediction

First have Anaconda installed in your machine. Then install the requirements by:
```
$ conda create --name new_env --file requirements.txt
```

Then run the two notebooks in succession. Afterwards you can now use the function Predictor(Date, Market, Keyword, CPC) defined at the end of the second notebook.

Two utility scripts are provided. First is `prepare_data.py` which takes a csv file that is assumed to exactly follow the same format as the original provided dataset (ie CTR is a string like '2.54%' instead of a float). This script simply replicates the data cleaning and transformation actions taken in the EDA notebook. It prepares the data to be used in the second notebook. Usage is shown below.

``` python -m prepare_data input.csv output.csv ```

The second script ```evaluation.py``` uses the Predictor() function from the second notebook and applies it to a test dataset. The input csv file is assumed to follow the same format as the original provided dataset. An inference is performed for each data item in the input dataset, and then the explained variance score is computed for each of the five target variables and then printed to the screen. The resulting dataframe with the inputs, true targets and predicted targets is written to disk. Usage is shown below. 

``` python -m evaluation test.csv result.csv --modelsdir models --prefix with-date-season ```

It takes an extra two optional arguments providing the directory and filename prefix to the sklearn serialized models in disk. These models are the result of training in the second notebook, and defaults are provided - these defaults are the models that have performed the best so far.

## Problem Description

An advertising company sells a service of buying keywords in search engines on behalf of their customers. They’re trying to optimise their keyword and funds allocation. The first towards the optimal solution is to predict performance by keyword and fund.

Goal:

Predicting for any keyword (not necessarily the ones in the dataset file), CPC, and market (US/UK) the traffic a website would receive (I.e., the clicks).

Task Evaluation:

The evaluation of the task will use an input dataset of new keywords and CPC for each market (US/UK) at the date of 14/2/2013. The model's results will be compared to real results for that day.

## Questions

1. How did you manipulate the data, and why? Illustrate your answer with plots.

See the EDA notebook. One important decision was removing all rows with zeroes for five numeric features (ie CPC), accounting for 42% of the dataset. All decisions in data cleaning and transformation are explained there, but in summary:

| feature name        |    original format/type        |  transformation applied           |
|:--------------------|:--------------------------|:----------------------------------|
| Date                |    int: YYYYMMDD          |  3 new int features: Year, Month, Season  |
| Market              | str: 'US-Market' or 'UK-Market' |   int: 1 or 0 (respectively)  |
| Keyword             |    str                     |      take lowercase              |
| CPC                 |   float                   |      apply np.log2                |
| Clicks              |   float                   |      apply np.log10               |
| CTR                 |   float (0 to 100)        |      (none)                       |
| Impressions         |   float                   |      apply np.log10               |
| Cost                |   float                   |      apply np.log10               |
| AveragePosition     |   float  (0 to 12)        |      (none)                       |

2. How did you perform NLP, if any?

I tried various word embedding algorithms and settled with ELMo as it performed better than word2vec, GloVe and also the CNN-based model of SpaCy. The embedding functions take in keyword strings then output embedding vectors, which are fed to the regressor. More details in the second notebook.

3. How did you model the problem, and why?

As discussed in the EDA notebook, we have a multi-output regression problem. The input features are Date, Market, CPC and the keyword as a vector embedding. I added Year, Month and Season categorical features derived from Date. The output features are CTR, clicks, impressions, cost and average position. Five regressors are constructed for the five target variables. Our chosen machine learning model for the regressor are CatBoost gradient boosted trees as there are a few categorical input features.

4. How did you evaluate your model? What were the results of the evaluation?

Five sets of explained variance scores and mean absolute errors were computed for five regressors. The closer to 1.0 the explained variance score is, the better the model is. For model selection, 5-fold cross-validation was performed for each regressor after having chosen hyperparameters with RandomSearch. See the second notebook for example results of cross-validation. Also, the models folder contains two text file records of CV results for two different feature sets. Many more configurations were attempted, only these two got recorded. In addition, in the second notebook the cleaned dataset is divided into a training set and test set. Cross-validation is performed only on the training set. The test set is fed to ```evaluation.py```

The regressors (for all configurations/hyperparameters) for AveragePosition had a score of around 0.5- 0.7, which is not as good as the other regressors (usually around 0.85-95). This may be a result of the fact that 95% of the data points had exactly 1.0 AveragePosition which will make it difficult to predict for any machine learning model, not to mention decision tree-based models.

5. If you had extra time, what would you do next?

Some points for improvement are:
- The embedding functions generate a vector for each word, but for each keyword (which usually consists of multiple words) I only took the averages over each of the 768 dimensions. A lot of information is lost when averaging. The biggest performance boost would probably come from constructing an attention-based LSTM recurrent neural network that receives a sequence of strings (the search keyword) and outputs a vector embedding. And then this could be connected to a simple multilevel-perceptron that converts that embedding into a prediction for CTR. Together the RNN plus the MLP is trained as a regressor using the keywords as input and the CTR values as targets, then the MLP is removed and the outputs of the RNN are used for generating vector embeddings.
- More vector embedding methods could have been tried. For example the larger ELMo embedding function could have been used, as it generates vectors of dimension greater than 2000.
- More ML models beyond CatBoost could have been tried. Generalized Linear models or Kernel-based methods might be able to perform better seeing that the problem has a high number of dimensions and also a large number of training samples
- Bayesian optimization has not yet been applied on the ML model
- Better data will definitely lead to a better model. It would have been ideal to have similar amounts of data for each month, because the provided data is not balanced in that feature.
