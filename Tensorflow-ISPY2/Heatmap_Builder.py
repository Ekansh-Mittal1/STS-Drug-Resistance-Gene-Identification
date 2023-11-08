from typing import List, Any

import numpy as np
import matplotlib

import pandas as pd
from matplotlib.patches import Patch

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

print('Start')
data_short = []
data_full = []
genes = []

first = True

drugs = ['Paclitaxel', 'Paclitaxel+Ganetespib', 'Paclitaxel+MK-2206', 'Paclitaxel+MK-2206+Trastuzumab',
         'Paclitaxel+Neratinib', 'Paclitaxel+Pembrolizumab']

drugs = ['Paclitaxel']

top_20s = []
fulls = []

for drug in drugs:
    full = pd.read_csv('DrugData/'+drug+'/importances.tsv', delimiter='\t')['col_name'][:200]
    t20 = pd.read_csv('DrugData/' + drug + '/' + drug + '1.0_full.csv')['GENE'][:30]

    top_20s.append(t20)
    fulls.append(full)

# genelist = []
all_drugs_t20s = pd.concat(top_20s)
unique_genes = list(all_drugs_t20s.unique())
all_drugs_full = pd.concat(fulls)
all_genes = list(all_drugs_full.unique())
all_data = pd.read_csv('../ISPY-Data/X_df.txt', delimiter='\t')
y_df = pd.read_csv('../ISPY-Data/comp_y_df.csv')
meta = pd.read_csv('../ISPY-Data/cv_df.txt', delimiter='\t', usecols=['BP-subtype'])
all_data = all_data.join(y_df, how='inner')
all_data = all_data.join(meta, how='inner')

all_data = all_data.loc[all_data['Drug'].isin(drugs)]
all_data = all_data.dropna(axis=0)
dataframe_short = all_data[unique_genes + ['Drug', 'Response', 'BP-subtype']]
dataframe_short = dataframe_short.reindex(
    dataframe_short.sort_values(by=['Drug', 'BP-subtype', 'Response']).index)
dataframe_short.to_csv('DrugData/shortDataFrame.csv')

dataframe_full = all_data[all_genes + ['Drug', 'Response', 'BP-subtype']]
dataframe_full = dataframe_full.reindex(
    dataframe_full.sort_values(by=['Drug', 'BP-subtype', 'Response']).index)
dataframe_full.to_csv('DrugData/fullDataFrame.csv')

drug = dataframe_short['Drug']
response = dataframe_short['Response']
bp = dataframe_short['BP-subtype']
# ispy = dataframe_short['I-SPY2 Subtypes']

# reformatting data short
for index, row in dataframe_short.iterrows():
    if first:
        sample_names = row[1:]
        first = False

    else:
        genes.append(row[0])
        data_short.append(row[1:])

for index, row in dataframe_full.iterrows():
    if first:
        sample_names = row[1:]
        first = False

    else:
        genes.append(row[0])
        data_full.append(row[1:])

# creating dataframe
data_short = pd.DataFrame(dataframe_short)
data_full = pd.DataFrame(dataframe_full)

# getting gene columns
columns_short = data_short.columns
numerical_cols_short = list(columns_short)[:-3]

columns_full = data_full.columns
numerical_cols_full = list(columns_full)[:-3]

drug_lut = dict(zip(drug.unique(), ['Salmon', 'MediumAquamarine', 'Green', 'DeepSkyBlue', 'Yellow', 'Purple']))
drug_colors = pd.Series(drug).map(drug_lut)

response_lut = dict(zip(response.unique(), ['Blue', 'Green']))
response_colors = pd.Series(response).map(response_lut)

bp_lut = dict(zip(bp.unique(), sns.color_palette('hls', bp.unique().size)))
bp_colors = pd.Series(bp).map(bp_lut)

#ispy_lut = dict(zip(ispy.unique(), sns.color_palette('husl', ispy.unique().size)))
#ispy_colors = pd.Series(ispy).map(ispy_lut)

# drug_response_colors = pd.DataFrame(drug_colors).join(pd.DataFrame(response_colors))
# drb_colors = pd.DataFrame(drug_response_colors).join(bp_colors)
# drbi_colors = pd.DataFrame(drb_colors).join(ispy_colors)

db = pd.DataFrame(drug_colors).join(pd.DataFrame(bp_colors))
# dbi = db.join(ispy_colors)
# dbir = dbi.join(response_colors)

dbr = db.join(response_colors)
print('creating heatmap')
sns.set_context("paper", font_scale=1.3)
sns_plot = sns.clustermap(data_short[numerical_cols_short], xticklabels=True, yticklabels=False, row_colors=dbr,
                          row_cluster=False,
                          figsize=(15, 10))
print('creating legend')
# creating legend

handles = [Patch(facecolor=drug_lut[name]) for name in drug_lut]
drug_leg = plt.legend(handles, drug_lut, title='Drug',
                      bbox_to_anchor=(0.01, 0.4), bbox_transform=plt.gcf().transFigure, loc='lower left', handleheight=2,
                      handlelength=7,
                      fontsize=7, title_fontsize='xx-large')
plt.gca().add_artist(drug_leg)

handles = [Patch(facecolor=bp_lut[name]) for name in bp_lut]
bp_leg = plt.legend(handles, bp_lut, title='BP-subtype',
                    bbox_to_anchor=(0.01, 0.28), bbox_transform=plt.gcf().transFigure, loc='lower left', handleheight=2,
                    handlelength=7,
                    fontsize=7, title_fontsize='xx-large')
plt.gca().add_artist(bp_leg)
'''
handles = [Patch(facecolor=ispy_lut[name]) for name in ispy_lut]
ispy_leg = plt.legend(handles, ispy_lut, title='ISPY-subtype',
                      bbox_to_anchor=(0, 0.08), bbox_transform=plt.gcf().transFigure, loc='lower left', handleheight=2,
                      handlelength=7,
                      fontsize=7, title_fontsize='xx-large')
plt.gca().add_artist(ispy_leg)
'''
handles = [Patch(facecolor=response_lut[name]) for name in response_lut]
response_leg = plt.legend(handles, response_lut, title='Response',
                          bbox_to_anchor=(0.01, 0.01), bbox_transform=plt.gcf().transFigure, loc='lower left',
                          handleheight=2,
                          handlelength=7,
                          fontsize=7, title_fontsize='xx-large')
plt.gca().add_artist(response_leg)

sns_plot.savefig("DrugData/"+drugs[0]+"/"+drugs[0]+"_heatmap_30.pdf")

print('creating heatmap')
sns.set_context("paper", font_scale=1.3)
sns_plot = sns.clustermap(data_full[numerical_cols_full], xticklabels=False, yticklabels=False,
                          row_colors=dbr, row_cluster=False,
                          figsize=(15, 10))
print('creating legend')
# creating legend
handles = [Patch(facecolor=drug_lut[name]) for name in drug_lut]
drug_leg = plt.legend(handles, drug_lut, title='Drug',
                      bbox_to_anchor=(0.01, 0.4), bbox_transform=plt.gcf().transFigure, loc='lower left',
                      handleheight=2,
                      handlelength=7,
                      fontsize=7, title_fontsize='xx-large')
plt.gca().add_artist(drug_leg)

handles = [Patch(facecolor=bp_lut[name]) for name in bp_lut]
bp_leg = plt.legend(handles, bp_lut, title='BP-subtype',
                    bbox_to_anchor=(0.01, 0.28), bbox_transform=plt.gcf().transFigure, loc='lower left', handleheight=2,
                    handlelength=7,
                    fontsize=7, title_fontsize='xx-large')
plt.gca().add_artist(bp_leg)

'''
handles = [Patch(facecolor=ispy_lut[name]) for name in ispy_lut]
ispy_leg = plt.legend(handles, ispy_lut, title='ISPY-subtype',
                      bbox_to_anchor=(0, 0.08), bbox_transform=plt.gcf().transFigure, loc='lower left', handleheight=2,
                      handlelength=7,
                      fontsize=7, title_fontsize='xx-large')
plt.gca().add_artist(ispy_leg)
'''
handles = [Patch(facecolor=response_lut[name]) for name in response_lut]
response_leg = plt.legend(handles, response_lut, title='Response',
                          bbox_to_anchor=(0.01, 0.01), bbox_transform=plt.gcf().transFigure, loc='lower left',
                          handleheight=2,
                          handlelength=7,
                          fontsize=7, title_fontsize='xx-large')
plt.gca().add_artist(response_leg)
sns_plot.savefig("DrugData/"+drugs[0]+"/"+drugs[0]+"_heatmap_shap.pdf")
