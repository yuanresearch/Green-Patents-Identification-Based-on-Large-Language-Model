#pip3 install -U scikit-learn
#pip3 install hdbscan

# Importing necessary packages.
import sys
import os
import zipfile as zip
import pandas as pd
import csv
import numpy as np


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)

Desktop_Developing = True

# FeatureCounts = 25
# Min_cluster_size=25
# Min_samples = 5

FeatureCounts = 15
Min_cluster_size=25
Min_samples = 25
friendPairs_fine_tuning = FeatureCounts




if Desktop_Developing:
    Y_category ='Y02T'   ### ONly accept 1 element
    full_cpc_file_address = 'ResultsServer/0_df_V1.csv'

else:
    Y_category = sys.argv[1]
    full_cpc_file_address = 'Results/0_df_V1.csv'

## Read CSV file from Results/0_df_V1.csv

if Desktop_Developing:

    df_patent_All_CPC = pd.read_csv(full_cpc_file_address, header=0, dtype='unicode', low_memory=False, nrows=1000000)
else:
    df_patent_All_CPC = pd.read_csv(full_cpc_file_address, header=0, dtype='unicode', low_memory=False)

#### Open a TXT file that has the name as the Y_category.


import datetime

now = datetime.datetime.now()

text_file = open("Results/Step2/"+str(Y_category)+'_KeyInfo.txt', "w")

## write an empty line

text_file.write("\n")

text_file.write("Current date and time: %s\n" % now.strftime("%Y-%m-%d %H:%M:%S"))

text_file.write("\n")

### write information into same file

text_file.write("Y_category: %s\n" % Y_category)
text_file.write("\n")
text_file.write("FeatureCounts: %s\n" % FeatureCounts)
text_file.write("Min_cluster_size: %s\n" % Min_cluster_size)
text_file.write("Min_samples: %s\n" % Min_samples)
text_file.write("friendPairs_fine_tuning: %s\n" % friendPairs_fine_tuning)
text_file.write("\n")

##### I only need those ros with non-empty all_CPC column.

df_patent_All_CPC = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].notna()]

df_Current_Y_category  = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains(str(Y_category), regex=False)]

print (len(df_Current_Y_category.index))

df_Current_Y_category.to_csv('tmp/2_df_Current_Y_category.csv', index=False)

## Create a new column that contain the first CPC in all_CPC column.

df_Current_Y_category['first_CPC'] = df_Current_Y_category['all_CPC'].str.split('|').str[0]

## Print out the top FeatureCounts most frequent CPCs in this category.

df_Current_Y_category_head_frequency_List = df_Current_Y_category.groupby(['first_CPC']).size().reset_index(name='counts')

df_Current_Y_category_head_frequency_List = df_Current_Y_category_head_frequency_List.sort_values(by=['counts'], ascending=False)

print (df_Current_Y_category_head_frequency_List.head(FeatureCounts))



## Print out the top FeatureCounts most frequent CPCs in all_CPC column.

df_Current_Y_category_copy = df_Current_Y_category.copy()

df_Current_Y_category_copy['all_CPC'] = df_Current_Y_category_copy['all_CPC'].str.split('|')

df_Current_Y_category_copy = df_Current_Y_category_copy.explode('all_CPC')

df_Current_Y_category_copy = df_Current_Y_category_copy.groupby(['all_CPC']).size().reset_index(name='counts')

df_Current_Y_category_frequency_List = df_Current_Y_category_copy.sort_values(by=['counts'], ascending=False)

print (df_Current_Y_category_frequency_List.head(FeatureCounts))



## Convert df_Current_Y_category_head_frequency_List to a list

df_Current_Y_category_head_frequency_List = df_Current_Y_category_head_frequency_List['first_CPC'].head(FeatureCounts).tolist()

print (df_Current_Y_category_head_frequency_List)

## Convert df_Current_Y_category_frequency_List to a list

df_Current_Y_category_frequency_List = df_Current_Y_category_frequency_List['all_CPC'].head(FeatureCounts).tolist()

print (df_Current_Y_category_frequency_List)

## Compare these two lists. Which is availabe in the df_Current_Y_category_frequency_List but not the df_Current_Y_category_head_frequency_List?

print (list(set(df_Current_Y_category_head_frequency_List) - set(df_Current_Y_category_frequency_List)))

print (list(set(df_Current_Y_category_frequency_List) - set(df_Current_Y_category_head_frequency_List)))

text_file.write("df_Current_Y_category_head_frequency_List: %s\n" % df_Current_Y_category_head_frequency_List)

text_file.write("df_Current_Y_category_frequency_List: %s\n" % df_Current_Y_category_frequency_List)

## Now I am merging these two lists together. I am also removing any CPCs start with Y, because they are not useful.

df_Current_Y_category_frequency_List = df_Current_Y_category_head_frequency_List + list(set(df_Current_Y_category_frequency_List) - set(df_Current_Y_category_head_frequency_List))

df_Current_Y_category_frequency_List = [x for x in df_Current_Y_category_frequency_List if not x.startswith('Y')]

print (df_Current_Y_category_frequency_List)


text_file.write("df_Current_Y_category_frequency_List(Revised): %s\n" % df_Current_Y_category_frequency_List)

### Now I will be using df_Current_Y_category_frequency_List to create feature vectors, based on the frequencies of these CPCs in each patent.
### I will create a new column for each CPC in df_Current_Y_category_frequency_List.
### The value will be the counts of the CPC is in the all_CPC column, otherwise 0.

for CPC in df_Current_Y_category_frequency_List:

    df_Current_Y_category[CPC] = df_Current_Y_category['all_CPC'].str.count(CPC)

print (df_Current_Y_category.head(10))

### Now I am going to create a new column called 'feature_vector' to store the feature vector for each patent.

df_Current_Y_category['feature_vector'] = df_Current_Y_category[df_Current_Y_category_frequency_List].values.tolist()

print (df_Current_Y_category.head(10))

df_Current_Y_category.to_csv('tmp/2_df_Current_Y_category_feature.csv', index=False)

### I am going to normalize the feature vectors.

from sklearn import preprocessing

x = df_Current_Y_category[df_Current_Y_category_frequency_List].values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_Current_Y_category_normalized = pd.DataFrame(x_scaled)

df_Current_Y_category_normalized.columns = df_Current_Y_category_frequency_List

## Keep firve digits after decimal point.

df_Current_Y_category_normalized = df_Current_Y_category_normalized.round(5)

print (df_Current_Y_category_normalized.head(10))



### I am going to do HDBSCAN clustering on the feature vectors.
# 1) I need to find the center of each cluster.
# 2) I need to get the outliers of patents, which can not be clustered into any cluster.

import hdbscan

df_Current_Y_category_feature = df_Current_Y_category_normalized[df_Current_Y_category_frequency_List].values.tolist()

clusterer = hdbscan.HDBSCAN(min_cluster_size=Min_cluster_size, min_samples = Min_samples, gen_min_span_tree=True, prediction_data=True)

clusterer.fit(df_Current_Y_category_feature)

print (clusterer.labels_)

print (clusterer.labels_.max())

text_file.write("Cluster numbers: %s\n" % clusterer.labels_.max())

print (clusterer.labels_.min())

print (clusterer.labels_.shape)

print (len(df_Current_Y_category_feature))

df_Current_Y_category['cluster'] = clusterer.labels_

### Get the probability of each label, and save it into a new column.

probabilities = clusterer.probabilities_

df_Current_Y_category['probabilities'] = probabilities

df_Current_Y_category.to_csv('tmp/2_df_Current_Y_category_cluster.csv', index=False)

### Get the outlier counts. It means df_Current_Y_category['cluster'] equals to -1.

print ("This is the counts of all df_Current_Y_category")

print (len(df_Current_Y_category.index))

text_file.write("df_Current_Y_category: %s\n" % len(df_Current_Y_category.index))

df_Current_Y_category_non_outlier = df_Current_Y_category[df_Current_Y_category['cluster'] != -1]

print ("This is the counts of all clustered patents in this Y category")

print (len(df_Current_Y_category_non_outlier.index))

text_file.write("This is the counts of all clustered patents in this Y category: %s\n" % len(df_Current_Y_category_non_outlier.index))

df_Current_Y_category_outlier = df_Current_Y_category[df_Current_Y_category['cluster'] == -1]

print ("This is the counts of all outlier patents in this Y category")

print (len(df_Current_Y_category_outlier.index))

text_file.write("This is the counts of all outlier patents in this Y category: %s\n" % len(df_Current_Y_category_outlier.index))

### Predict the label of a new patent feature vector with hdbscan.approximate_predict

test_feature_vector = df_Current_Y_category_feature[0]

### Convert it into NP array

test_feature_vector = np.array(test_feature_vector)

print (test_feature_vector)

## wrap up it with a list

test_feature_vector = [test_feature_vector]

test_labels, strengths = hdbscan.approximate_predict(clusterer, test_feature_vector)

print (test_labels)


### Now I am going to find out all patents that relvant to this Y category, but not in this Y category, using df_Current_Y_category_frequency_List

df_Current_Y_category_frequency_List_COPY = df_Current_Y_category_frequency_List

print (df_Current_Y_category_frequency_List_COPY)

df_Current_Y_category_frequency_List_COPY = '|'.join(df_Current_Y_category_frequency_List_COPY)

print (df_Current_Y_category_frequency_List_COPY)

print (len(df_patent_All_CPC.index))

print (df_patent_All_CPC['all_CPC'].head(10))

df_Current_Y_category_friends = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains(str(df_Current_Y_category_frequency_List_COPY), regex=True)]

print ("This is the counts of all the friend patents in this Y category")
print (len(df_Current_Y_category_friends.index))

text_file.write("This is the counts of all the friend patents in this Y category: %s\n" % len(df_Current_Y_category_friends.index))


## remove any Y_category patents from df_Current_Y_category_friends

df_Current_Y_category_friends = df_Current_Y_category_friends[~df_Current_Y_category_friends['all_CPC'].str.contains(str(Y_category), regex=False)]

print ("This is the counts of all the friend patents in this Y category, without Y_category itself")
print (len(df_Current_Y_category_friends.index))

text_file.write("This is the counts of all the friend patents in this Y category, without Y_category itself: %s\n" % len(df_Current_Y_category_friends.index))


## I will create a similarly feature vector for each patent in df_Current_Y_category_friends.

for CPC in df_Current_Y_category_frequency_List:

    df_Current_Y_category_friends[CPC] = df_Current_Y_category_friends['all_CPC'].str.count(CPC)

#print (df_Current_Y_category_friends.head(10))

### Now I am going to create a new column called 'feature_vector' to store the feature vector for each patent.

df_Current_Y_category_friends['feature_vector'] = df_Current_Y_category_friends[df_Current_Y_category_frequency_List].values.tolist()

#print (df_Current_Y_category_friends.head(10))

df_Current_Y_category_friends.to_csv('tmp/2_df_Current_Y_category_friends_feature.csv', index=False)

### I am going to normalize the feature vectors.

from sklearn import preprocessing

x = df_Current_Y_category_friends[df_Current_Y_category_frequency_List].values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_Current_Y_category_friends_normalized = pd.DataFrame(x_scaled)

df_Current_Y_category_friends_normalized.columns = df_Current_Y_category_frequency_List

## Keep firve digits after decimal point.

df_Current_Y_category_friends_normalized = df_Current_Y_category_friends_normalized.round(5)

#print (df_Current_Y_category_friends_normalized.head(10))

### I am goint to Predict the label of all feature vector with hdbscan.approximate_predict

df_Current_Y_category_friends_feature = df_Current_Y_category_friends_normalized[df_Current_Y_category_frequency_List].values.tolist()

test_labels, strengths = hdbscan.approximate_predict(clusterer, df_Current_Y_category_friends_feature)

#print (test_labels)

df_Current_Y_category_friends['cluster'] = test_labels

### Get the probability of each label, and save it into a new column.

probabilities = strengths

df_Current_Y_category_friends['probabilities'] = probabilities

df_Current_Y_category_friends.to_csv('tmp/2_df_Current_Y_category_friends_cluster.csv', index=False)



### Now I am going to construct a big dataset, which contains all the patents in df_Current_Y_category and df_Current_Y_category_friends

## DRop the first_CPC column in df_Current_Y_category

df_Current_Y_category = df_Current_Y_category.drop(['first_CPC'], axis=1)

## Add a label column to df_Current_Y_category, if cluster is -1, then label is 0, otherwise 1.

df_Current_Y_category['label'] = np.where(df_Current_Y_category['cluster'] == -1, 0, 1)

## add a label column to df_Current_Y_category_friends, if cluster is -1, then label is 2, otherwise 3.

df_Current_Y_category_friends['label'] = np.where(df_Current_Y_category_friends['cluster'] == -1, 2, 3)

## Label 2 is negative patents, Label 3 is candidate patent

df_Current_Y_category_friends_candidates = df_Current_Y_category_friends[df_Current_Y_category_friends['label'] == 3]

df_Current_Y_category_friends_negative = df_Current_Y_category_friends[df_Current_Y_category_friends['label'] == 2]

print ("This is the counts of the candidate patents in this Y category")

print (len(df_Current_Y_category_friends_candidates.index))

text_file.write("This is the counts of the candidate patents in this Y category: %s\n" % len(df_Current_Y_category_friends_candidates.index))


print ("This is the counts of the negative patents in this Y category")

print (len(df_Current_Y_category_friends_negative.index))

text_file.write("This is the counts of the negative patents in this Y category: %s\n" % len(df_Current_Y_category_friends_negative.index))


## A) Fine turning the friends candidates dataset. If the candidate is too much, we will
## spend too much money on OPENAI, while get a small percentage of green patents.
## Ideally, the number of candidates should be the same as the number of raw green patents,
## maximum not more than 2 times of the number of raw green patents.
## This is done by count the frequency of high frequency CPC pairs from the raw green patents,
## and then use these high frequency CPC pairs to filter the candidates.


## create a copy of df_Current_Y_category

df_Current_Y_category_copy = df_Current_Y_category.copy()

## create a new column called all_CPC_non_duplicated, which contains non-duplicated CPCs in all_CPC column.

df_Current_Y_category_copy['all_CPC_non_duplicated'] = df_Current_Y_category_copy['all_CPC'].str.split('|').apply(set).str.join('|')

group_id_non_duplicated_list = df_Current_Y_category_copy.all_CPC_non_duplicated.tolist()
#print (group_id_non_duplicated_list)
group_id_non_duplicated_list_restructured = [i.split('|') for i in group_id_non_duplicated_list]
#print (group_id_non_duplicated_list_restructured)
group_id_non_duplicated_list_restructured_2 = [[ subelt for subelt in elt if subelt != Y_category ] for elt in group_id_non_duplicated_list_restructured]
#print (group_id_non_duplicated_list_restructured_2)

from collections import Counter
from itertools import combinations

a = group_id_non_duplicated_list_restructured_2
d  = Counter()
for sub in a:
    if len(a) < 2:
        continue
    sub.sort()
    for comb in combinations(sub,2):
        d[comb] += 1

### Now I got most frequent friend pairs from Y_category information. ## 10 most frequent pairs.
## For example, if we use Y02T as the Y_category, then the most frequent friend pairs are:
## [(('H01M', 'Y02E'), 3360), (('F01D', 'F05D'), 3350), (('B60L', 'H02J'), 2339), (('B60L', 'Y02E'), 2116), (('F02C', 'F05D'), 1906), (('B60K', 'B60Y'), 1702), (('B60L', 'H01M'), 1690), (('B60K', 'B60W'), 1635), (('B60K', 'B60L'), 1464), (('F01D', 'F02C'), 1375)]
## I keep Y02E here, because it is possible that some Y02E patents are not correctly assigned to Y02T,
## but they are still green patents.

print(d.most_common(friendPairs_fine_tuning))

most_common_pairs= [word for word, word_count in Counter(d).most_common(friendPairs_fine_tuning)]

print (most_common_pairs)

text_file.write("high-frequency pairs: %s\n" % d.most_common(friendPairs_fine_tuning))
text_file.write("high-frequency pairs CPC ONLY: %s\n" % most_common_pairs)

def Counts_Key_Pairs(group_id_non_duplicated,Most_Common_Words):
    count = 0
    for pairs in Most_Common_Words:
        if (pairs[0] in group_id_non_duplicated) and (pairs[1] in group_id_non_duplicated):
            count = count +1
    return count

df_Current_Y_category_friends_candidates['high_freq_pair_count'] = df_Current_Y_category_friends_candidates.apply(lambda row: Counts_Key_Pairs(row['all_CPC'], most_common_pairs), axis=1)

## Count the high_freq_pair_count column

print (df_Current_Y_category_friends_candidates['high_freq_pair_count'].value_counts().sort_index())
print (df_Current_Y_category_friends_candidates['high_freq_pair_count'].value_counts().sort_index().index[-1])
## Adaptive filtering from Value_counts. the maximum counts of df_Current_Y_category_friends_candidates
## should be not more than 2 times of the number of df_Current_Y_category.

max_index = df_Current_Y_category_friends_candidates['high_freq_pair_count'].value_counts().sort_index().index[-1]

starting_index = 0

print ("Starting index is: " + str(starting_index))
print ("max_index is: " + str(max_index))

while starting_index < max_index:

    df_Current_Y_category_friends_candidates_adaptive_selection = df_Current_Y_category_friends_candidates[
        df_Current_Y_category_friends_candidates['high_freq_pair_count'] >= starting_index]

    if len(df_Current_Y_category_friends_candidates_adaptive_selection.index) <= 2*len(df_Current_Y_category.index):
        break
    else:
        starting_index = starting_index +1

print ("Ending index is: " + str(starting_index))

text_file.write("high_freq_pair_count: %s\n" % df_Current_Y_category_friends_candidates['high_freq_pair_count'].value_counts().sort_index())
text_file.write("max_index is: %s\n" % str(max_index))
text_file.write("Ending index is: %s\n" % str(starting_index))


df_Current_Y_category_friends_candidates = df_Current_Y_category_friends_candidates[df_Current_Y_category_friends_candidates['high_freq_pair_count'] >= starting_index]

print ('This is the counts of the candidates after filtering by high_freq_pair_count')

print (len(df_Current_Y_category_friends_candidates.index))

text_file.write("This is the counts of the candidates after filtering by high_freq_pair_count: %s\n" % len(df_Current_Y_category_friends_candidates.index))




## B) Fine turning the negative dataset. I narrow the negative dataset by using the 100 common GreenKeywords provided by GPT4 as of 2023/12/18

green_keywords = [
    'sustainable', 'renewable', 'eco-friendly', 'biodegradable', 'recyclable',
    'solar', 'wind', 'geothermal', 'hydroelectric', 'biomass',
    'organic', 'clean energy', 'carbon footprint', 'zero emissions', 'low impact',
    'conservation', 'environmentally friendly', 'green technology', 'energy-efficient', 'sustainability',
    'climate change', 'greenhouse gas', 'natural resources', 'ecosystem', 'biodiversity',
    'carbon neutral', 'renewable resources', 'pollution reduction', 'waste management', 'recycling',
    'water conservation', 'air quality', 'green building', 'solar power', 'wind energy',
    'energy storage', 'electric vehicle', 'sustainable agriculture', 'sustainable development', 'environmental protection',
    'organic farming', 'green chemistry', 'clean technology', 'sustainable transport', 'green fuel',
    'carbon capture', 'solar panel', 'wind turbine', 'energy conservation', 'sustainable energy',
    'green design', 'environmental sustainability', 'ecological', 'clean air', 'green innovation',
    'sustainable materials', 'energy saving', 'renewable energy', 'ecofriendly products', 'climate action',
    'solar energy', 'wind farm', 'greenhouse effect', 'sustainable living', 'environmental impact',
    'carbon reduction', 'ecological footprint', 'green manufacturing', 'sustainable design', 'biofuel',
    'hydropower', 'green economy', 'eco-innovation', 'green practices', 'environmental conservation',
    'organic products', 'reusable', 'carbon offset', 'sustainable packaging', 'green initiative',
    'solar cells', 'wind power', 'eco-conscious', 'sustainable lifestyle', 'sustainable practices',
    'green business', 'ecology', 'green infrastructure', 'environmental stewardship', 'green energy',
    'sustainable solutions', 'recycled materials', 'energy transition', 'climate resilience', 'green policy',
    'energy efficiency', 'sustainable urban development', 'environmental technology', 'sustainable resource use', 'carbon sequestration'
]

## if any of the titile or abstract of the negative patents contain any of the green_keywords, then I will exclude it from the negative dataset.

df_Current_Y_category_friends_negative = df_Current_Y_category_friends_negative[~df_Current_Y_category_friends_negative['patent_title'].str.contains('|'.join(green_keywords), regex=True, na=False)]

df_Current_Y_category_friends_negative = df_Current_Y_category_friends_negative[~df_Current_Y_category_friends_negative['patent_abstract'].str.contains('|'.join(green_keywords), regex=True, na=False)]

print ("This is the counts of the negative patents in this Y category, after removing the green keywords")

print (len(df_Current_Y_category_friends_negative.index))

text_file.write("This is the counts of the negative patents in this Y category, after removing the green keywords: %s\n" % len(df_Current_Y_category_friends_negative.index))


## I will keep the counts of negative patents the same as the counts of clustered patents, df_Current_Y_category_non_outlier

df_Current_Y_category_friends_negative = df_Current_Y_category_friends_negative.sample(n=len(df_Current_Y_category_non_outlier.index), random_state=1, replace=True)

print ("This is the counts of the negative patents in this Y category, after sampling")

print (len(df_Current_Y_category_friends_negative.index))

text_file.write("This is the counts of the negative patents in this Y category, after sampling: %s\n" % len(df_Current_Y_category_friends_negative.index))


## Merge df_Current_Y_category and df_Current_Y_category_friends_candidates and df_Current_Y_category_friends_negative

df_Current_Y_category_all = pd.concat([df_Current_Y_category, df_Current_Y_category_friends_candidates, df_Current_Y_category_friends_negative])

print ("THis is the counts of all patents in this Y category plus all candidates from Y category friends")

print (len(df_Current_Y_category_all.index))

text_file.write("THis is the counts of all patents in this Y category plus all candidates from Y category friends: %s\n" % len(df_Current_Y_category_all.index))


## Save it into Step2 folder, with Y_category variable as the file name.

df_Current_Y_category_all.to_csv('Results/Step2/'+str(Y_category)+'_ML_Candidates.csv', index=False)

text_file.close()
























