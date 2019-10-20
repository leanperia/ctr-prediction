import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('inputfile', type=str, help='filename of input')
parser.add_argument('outputfile', type=str, help='filename of output')

args = parser.parse_args()

data = pd.read_csv(args.inputfile)

# Data cleaning and transformations
data['AveragePosition'] = data['Average.Position']
data.drop('Average.Position', axis=1, inplace=True)

data.dropna(inplace=True)
data.CTR = data.CTR.apply(lambda x: x[:-1]).astype('float64')
data = data[(data.CPC != 0.0) | (data.AveragePosition != 0.0) | (data.Impressions != 0.0)]
data['Year'] = data.Date.astype('str').apply(lambda x: x[:4]).astype('int')
data['Month'] = data.Date.astype('str').apply(lambda x: x[4:6]).astype('int')

def get_season(month):
    if month >= 3 and month < 6:
        return 1 # Spring
    elif month >= 6 and month < 9:
        return 2 # Summer
    elif month >= 9 and month < 11:
        return 3 # Fall
    else: 
        return 4 # Winter

data['Season'] = data.Month.apply(get_season)

data.CPC = np.log2(data.CPC.replace(0,0.0001))
data.Clicks = np.log10(data.Clicks.replace(0,0.001))
data.Impressions = np.log10(data.Impressions.replace(0,0.0001))
data.Cost = np.log10(data.Cost.replace(0,0.001))

data['Keyword'] = data.Keyword.apply(lambda x: x.lower())
data['Market'] = data.Market.map({'US-Market':1, 'UK-Market':0})

data.to_csv(args.outputfile, index_label=False)

print('Finished!')