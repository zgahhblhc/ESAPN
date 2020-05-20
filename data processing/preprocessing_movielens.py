# %matplotlib inline
import os, pickle
import pandas as pd
import numpy as np
# np.set_printoptions(threshold=30)

Train_Size = 22000000     # 5000000 10000000 15000000 27000000
Output_Dim = 5           # 1 5
DATA_SET_NAME = 'ml-latest' # 'ml-20m' 'ml-25m' 'ml-27m'
DATA_PATH     = '../data'
movies        = pd.read_csv(os.path.join(DATA_PATH, DATA_SET_NAME,'movies.csv' if DATA_SET_NAME == 'ml-20m' else 'movies.csv'),index_col=None, encoding='utf-8')
ratings       = pd.read_csv(os.path.join(DATA_PATH, DATA_SET_NAME,'ratings.csv' if DATA_SET_NAME == 'ml-20m' else 'ratings.csv'),index_col=None, encoding='utf-8')
tags          = pd.read_csv(os.path.join(DATA_PATH, DATA_SET_NAME,'tags.csv' if DATA_SET_NAME == 'ml-20m' else 'tags.csv'),index_col=None, encoding='utf-8')
genome_tags   = pd.read_csv(os.path.join(DATA_PATH, DATA_SET_NAME,'genome-tags.csv' if DATA_SET_NAME == 'ml-20m' else 'genome-tags.csv'),index_col=None, encoding='utf-8')
genome_scores = pd.read_csv(os.path.join(DATA_PATH, DATA_SET_NAME,'genome-scores.csv' if DATA_SET_NAME == 'ml-20m' else 'genome-scores.csv'),index_col=None, encoding='utf-8')

print('')
print('ratings[userId].max()+1',  ratings['userId'].max()+1)
print('ratings[movieId].max()+1', ratings['movieId'].max()+1)

print('')
print('The number of movies: {}'.format(movies.count()['movieId']))
print('The number of ratings: {}'.format(ratings.count()['movieId']))

print('')
print('min value of rating: {}'.format(ratings['rating'].min()))
print('max value of rating: {}'.format(ratings['rating'].max()))

print('')
ra = ratings.groupby(ratings['userId']).count()
print('The number of user in ratings.csv: {}'.format(ra.count()[0]))
print('The minimum number of ratings per user in ratings.csv: {}'.format(ra['movieId'].min()))
print('The maximun number of ratings per user in ratings.csv: {}'.format(ra['movieId'].max()))

print('')
ra = ratings.groupby(ratings['movieId']).count()
print('The number of movies in ratings.csv: {}'.format(ra.count()[0]))
print('The minimum number of ratings per movie in ratings.csv: {}'.format(ra['userId'].min()))
print('The maximun number of ratings per movie in ratings.csv: {}'.format(ra['userId'].max()))

print('The number of tags in tags.csv: {}'.format(tags.count()['userId']))
print('The number of tags in genome-tags.csv: {}'.format(genome_tags.count()['tagId']))
tags_in_genome_tags = genome_tags.count()['tagId']
print('')
ra = tags.groupby(tags['userId']).count()
print('The number of user in tags.csv: {}'.format(ra.count()[0]))
print('The minimum number of tags per user in tags.csv: {}'.format(ra['movieId'].min()))
print('The maximun number of tags per user in tags.csv: {}'.format(ra['movieId'].max()))

print('')
ra = tags.groupby(tags['movieId']).count()
print('The number of movies in tags.csv: {}'.format(ra.count()[0]))
print('The minimum number of tags per movie in tags.csv: {}'.format(ra['userId'].min()))
print('The maximun number of tags per movie in tags.csv: {}'.format(ra['userId'].max()))

print('')
tags_mer = pd.merge(tags, genome_tags, how='left', left_on='tag', right_on='tag')
print('The number of tags in tags.csv but not in genome-tags.csv: {}'.format(tags_mer[(tags_mer['tagId'].isnull())].count()[0]))

print('The length of genome_scores.csv: {}'.format(genome_scores.count()['movieId']))
print('max value of relevance from genome_scores.csv: {}'.format(genome_scores['relevance'].max()))
print('min value of relevance from genome_scores.csv: {}'.format(genome_scores['relevance'].min()))

print('')
ra = genome_scores.groupby(genome_scores['movieId']).count()
print('The number of movies in genome_scores.csv: {}'.format(ra.count()[0]))
movies_in_genome_scores = ra.count()[0]
print('The minimum number of tags per movie in genome_scores.csv: {}'.format(ra['tagId'].min()))
print('The maximun number of tags per movie in genome_scores.csv: {}'.format(ra['tagId'].max()))

genome_scores_group = genome_scores.groupby(genome_scores['movieId']).mean()
ratings_group = ratings.groupby(ratings['movieId']).mean()
rat_ge_merge = pd.merge(ratings_group, genome_scores_group, how='inner', left_on='movieId', right_on='movieId')
number = rat_ge_merge.count()[0]
print('Number of movies in both genome_scores.csv and ratings.csv: {}. Take up {}% of ratings.csv'.format(number, round(number/19545*100)))

ratings_genome_merge = pd.merge(ratings, genome_scores_group, how='inner', left_on='movieId', right_on='movieId')
number = ratings_genome_merge.count()[0]
print('Number of ratings where its movieId in genome_scores.csv: {}. Take up {}% of ratings.csv'.format(number, round(number/20000263*100)))

print('')
ra = ratings_genome_merge.groupby(ratings_genome_merge['userId']).count()
number = ra.count()[0]
print('{} users rate the movies appearing in both genome_scores.csv and ratings.csv. Take up {}% of ratings.csv'.format(number, round(number/138493*100)))
print('Minimum number of ratings per user for the movies appearing in both genome_scores.csv and ratings.csv: {}'.format(ra['movieId'].min()))

# Preprocess data (You should run last code cell to get 'ratings_genome_merge')
# The first column of features is userId, the next is movieId.
# The only one column of target is rating.

remove_fields = ['timestamp','tagId','relevance','rating']
# ratings_genome_merge.sort_values(by=['timestamp'], inplace=True)
target   = ratings_genome_merge['rating']
feature  = ratings_genome_merge.drop(remove_fields, axis=1)
# print('\nfeature\n', feature)
# print('\ntarget\n', target)
features = feature.values
targets  = target.values
# print('\nfeatures\n', features)
# print('\ntargets\n', targets)
targets   = 1 * np.vectorize(lambda i: i > 3.5)(targets) if Output_Dim == 1 else np.floor(targets-0.5)
# print('\ntargets1\n', targets, np.unique(targets), len(targets))

# Actually, using train_test_split in here is not best.
# The better method should split the data according the userId, which make sure every user is in the test set.
# But here, let us make it easier and quickly ( We have aLReady include 99.86% users).
from sklearn.model_selection import train_test_split
train_features, test_features, train_target, test_target = train_test_split(features, targets, train_size = Train_Size, random_state = 0, shuffle=True)
print('\ntrain_features\n', train_features, type(train_features), train_features.shape)
print('\ntrain_target\n',   train_target,   type(train_target),   train_target.shape)
print('\ntest_features\n',  test_features,  type(test_features),  test_features.shape)
print('\ntest_target\n',    test_target,    type(test_target),    test_target.shape)

user_count_dict   = dict()
user_rating_count = list()
for line in train_features[:,0]:
    if line in user_count_dict:
        user_count_dict[line] += 1
        user_rating_count.append([user_count_dict[line]])
    else:
        user_count_dict[line] = 1
        user_rating_count.append([1])

movie_count_dict = dict()
movie_rating_count = list()
for line in train_features[:,1]:
    if line in movie_count_dict:
        movie_count_dict[line] += 1
        movie_rating_count.append([movie_count_dict[line]])
    else:
        movie_count_dict[line] = 1
        movie_rating_count.append([1])

train_feature_data = pd.DataFrame(train_features, columns=list(feature.columns))
train_feature_data['user_frequency']  = pd.DataFrame(user_rating_count)
train_feature_data['movie_frequency'] = pd.DataFrame(movie_rating_count)
# train_feature_data.to_csv('./{}/{}_train_feature_data_{}_{}.csv'.format(DATA_PATH, DATA_SET_NAME, Train_Size, Output_Dim), index = None)
print('train_feature_data_{}_{}.csv done.\n'.format(Train_Size, Output_Dim))

user_count_dict   = dict()
user_rating_count = list()
for line in test_features[:,0]:
    if line in user_count_dict:
        user_count_dict[line] += 1
        user_rating_count.append([user_count_dict[line]])
    else:
        user_count_dict[line] = 1
        user_rating_count.append([1])

movie_count_dict   = dict()
movie_rating_count = list()
for line in test_features[:,1]:
    if line in movie_count_dict:
        movie_count_dict[line] += 1
        movie_rating_count.append([movie_count_dict[line]])
    else:
        movie_count_dict[line] = 1
        movie_rating_count.append([1])

test_feature_data = pd.DataFrame(test_features, columns=list(feature.columns))
test_feature_data['user_frequency']  = pd.DataFrame(user_rating_count)
test_feature_data['movie_frequency'] = pd.DataFrame(movie_rating_count)
# test_feature_data.to_csv('./{}/{}_test_feature_data_{}_{}.csv'.format(DATA_PATH, DATA_SET_NAME, len(test_feature_data), Output_Dim), index = None)
print('test_feature_data_{}_{}.csv done.\n'.format(len(test_feature_data),Output_Dim))

train_features = train_feature_data.values
test_features  = test_feature_data.values

print("train_features:", train_feature_data.head())
print("train_targets:", train_target[:10])

print('\ntrain_features\n', train_features, type(train_features), train_features.shape)
print('\ntrain_target\n',   train_target,   type(train_target),   train_target.shape)
print('\ntest_features\n',  test_features,  type(test_features),  test_features.shape)
print('\ntest_target\n',    test_target,    type(test_target),    test_target.shape)
pickle.dump((train_features, test_features, train_target, test_target), open('./{}/{}_TrainTest_{}_{}.data'.format(DATA_PATH, DATA_SET_NAME, Train_Size, Output_Dim), 'wb'))

dict_t = {}
dict_t['userId'] = test_features[:,0]
dict_t['movieId'] = test_features[:,1]
pd_data = pd.DataFrame.from_dict(dict_t)
user_test = pd_data.groupby(pd_data['userId']).count().count()[0]
print('{}% users in test set ({} users)'.format(round(user_test/138493*100, 2), user_test))

dict_t = {}
dict_t['userId'] = train_features[:,0]
dict_t['movieId'] = train_features[:,1]
pd_data = pd.DataFrame.from_dict(dict_t)
user_train = pd_data.groupby(pd_data['userId']).count().count()[0]
print('{}% users in training set ({} users)'.format(round(user_train/138493*100, 2), user_train))


print('Preprocessing train/test data done.\n')

###########################################################################################################

genome_scores_dict = {}
for i in range(movies_in_genome_scores):
    if i % 1000 == 0: print(i, movies_in_genome_scores)
    m_id = -1
    vec = []
    for j in range(tags_in_genome_tags):
        index = j + i * tags_in_genome_tags
        if m_id < 0:
            m_id = genome_scores['movieId'][index]
        assert genome_scores['movieId'][index] == m_id
        assert genome_scores['tagId'][index] == j + 1
        vec.append(genome_scores['relevance'][index])
    genome_scores_dict[str(m_id)] = vec

# Save preprocess data to './data/verify_assumption.data'
pickle.dump((genome_scores_dict), open('./{}/{}_GenomeScoresDict.data'.format(DATA_PATH, DATA_SET_NAME), 'wb'))
print('Preprocessing dictdata done.')