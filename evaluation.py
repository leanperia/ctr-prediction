import pandas as pd
import numpy as np
from tqdm import tqdm

import flair
from flair.data import Sentence
from flair.embeddings import ELMoEmbeddings, DocumentPoolEmbeddings
import catboost
from sklearn.metrics import explained_variance_score

from joblib import dump, load
from os.path import join

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('inputfile', type=str, help='filename of input')
parser.add_argument('outputfile', type=str, help='filename of output')
parser.add_argument('-p', '--prefix', type=str, default='with-date-season', help='common prefix in filenames of pretrained models')
parser.add_argument('-d', '--modelsdir', type=str, default='models', help='directory containing pretrained models')

args = parser.parse_args()

test_df = pd.read_csv(args.inputfile)

# Throw away unwanted rows
test_df['AveragePosition'] = test_df['Average.Position']
test_df.drop('Average.Position', axis=1, inplace=True)
test_df.dropna(inplace=True)
test_df.CTR = test_df.CTR.apply(lambda x: x[:-1]).astype('float64')
test_df = test_df[(test_df.CPC != 0.0) | (test_df.AveragePosition != 0.0) | (test_df.Impressions != 0.0)]

# Compute embeddings
print('Loading ELMo model...', end='')
elmo_small = ELMoEmbeddings('small')
print('Done!')
document_embedding = DocumentPoolEmbeddings([elmo_small])
def compute_elmo_embedding(keyword):
    sentence = Sentence(keyword)
    document_embedding.embed(sentence)
    return sentence.get_embedding().detach().cpu().numpy()
vectors = []
print('\nNow computing embeddings for keywords...', end='')
for keyword in tqdm(test_df.Keyword.values, total=test_df.shape[0]):
    vectors.append(compute_elmo_embedding(keyword))
vectors = pd.DataFrame.from_records(np.array(vectors),index=test_df.index)
print('Done!')
test_df = pd.concat([test_df, vectors], axis=1)
    
# Load the TRAINED regressor models from disk
# Each file follows the filepath 'MODELS_DIR/<PREFIX>.<ctr/cost/impr/click/ap>.joblib'
print('\n'+('='*40)+ '\nLoading the pretrained models...', end='')
ctr_predictor = load(join(args.modelsdir, args.prefix + '.ctr.joblib'))
cost_predictor = load(join(args.modelsdir, args.prefix + '.cost.joblib'))
impression_predictor = load(join(args.modelsdir, args.prefix + '.impr.joblib'))
click_predictor = load(join(args.modelsdir, args.prefix + '.click.joblib'))
ap_predictor = load(join(args.modelsdir, args.prefix + '.ap.joblib'))
print('Done!')

def get_season(month):
    if month >= 3 and month < 6:
        return 1 # Spring
    elif month >= 6 and month < 9:
        return 2 # Summer
    elif month >= 9 and month < 11:
        return 3 # Fall
    else: 
        return 4 # Winter
def Predictor(Date, Market, Keyword, CPC, embedding_function=compute_elmo_embedding):
    # NOTE: this function only takes a single datapoint at a time
    # Each input must match the data type of the corresponding column in the original dataset, then 
    # the same data transformations are applied as in the EDA notebook
    year = int(str(Date)[:4])
    month = int(str(Date)[4:6])
    season = get_season(month)
    market = 1 if Market == 'US-Market' else 0
    cpc = np.log2(CPC)
    keyword = Keyword.lower()
    vector = list(embedding_function(keyword))
    input_vector = [Date, market, cpc, year, month, season, *vector]   ### WARNING: This must match the pretrained model's arguments
    ctr = ctr_predictor.predict(input_vector)
    clicks = click_predictor.predict(input_vector)
    averageposition = ap_predictor.predict(input_vector)
    impressions = impression_predictor.predict(input_vector)
    cost = cost_predictor.predict(input_vector)
    return ctr, 10**clicks, 10**impressions, 10**cost, averageposition

input_df = test_df[['Date', 'Market', 'Keyword', 'CPC']]    ### WARNING: This must match the pretrained model
result_raws = []
print('\n'+ ('='*40) + '\n')
print('Now performing inference on the test data...', end='')
for idx, item in tqdm(input_df.iterrows(), total=input_df.shape[0]):
    result_raws.append(Predictor(*list(item)))
print('Done!')
result_df = pd.DataFrame.from_records(result_raws, index=test_df.index, columns=['pred_ctr', 'pred_clicks', 'pred_impr', 'pred_cost', 'pred_ap'])
result_df = pd.concat([test_df,result_df], axis=1)

ctr = explained_variance_score(result_df.CTR, result_df.pred_ctr)
clicks = explained_variance_score(result_df.Clicks, result_df.pred_clicks)
impr = explained_variance_score(result_df.Impressions, result_df.pred_impr)
cost = explained_variance_score(result_df.Cost, result_df.pred_cost)
ap = explained_variance_score(result_df.CTR, result_df.pred_ctr)

print('\n'+('='*40)+'\n')
print('These are the obtained explained variance scores (1.0 is best) for the five regressors:')
print(f'CTR: {ctr}')
print(f'Clicks: {clicks}')
print(f'Impressions: {impr}')
print(f'Cost: {cost}')
print(f'Average Position: {ap}')
print(f'Mean score: {np.average([ctr, clicks, impr, cost, ap])}')

print('='*30 + '\nNow writing the resulting dataframe to disk.')
result_df.to_csv(args.outputfile, index_label=False)
print('All complete!')

print('Finished!')