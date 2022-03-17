import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import pandas as pd
pd.options.display.width = 0

viral_tweets = pd.read_json('random_tweets.json', lines=True)

median_retweet_count = viral_tweets['retweet_count'].median()
viral_tweets['is_viral'] = viral_tweets['retweet_count'].apply(lambda x: 1 if x > median_retweet_count else 0)
viral_tweets['tweet_length'] = viral_tweets.apply(lambda x: len(x['text']), axis=1)
viral_tweets['followers_count'] = viral_tweets.apply(lambda x: x['user']['followers_count'], axis=1)
viral_tweets['friends_count'] = viral_tweets.apply(lambda x: x['user']['friends_count'], axis=1)
viral_tweets['number_of_hashtags'] = viral_tweets['text'].apply(lambda x: x.count('#'))

labels = viral_tweets['is_viral']
cols_to_keep = ['tweet_length', 'followers_count', 'friends_count', 'number_of_hashtags', 'favorite_count', 'is_viral']

lr_viral_tweets_df = viral_tweets[cols_to_keep]

X = lr_viral_tweets_df.iloc[:, :-1]
y = lr_viral_tweets_df.iloc[:, -1]
#print(X)
#print(y)

lr = LogisticRegression(max_iter=1000)
lr.fit(X, y)
print(lr.score(X, y))

sfs_forward = SFS(lr, k_features=3, forward=True, floating=False, scoring='accuracy', cv=0)
sfs_forward.fit(X, y)

plot_sfs(sfs_forward.get_metric_dict())
plt.show()

sfs_backward = SFS(lr, k_features=3, forward=False, floating=False, scoring='accuracy', cv=0)
sfs_backward.fit(X, y)
plot_sfs(sfs_backward.get_metric_dict())
plt.show()

sbfs = SFS(lr,
          k_features=3,
          forward=False,
          floating=True,
          scoring='accuracy',
          cv=0)
sbfs.fit(X, y)
print(sbfs.subsets_[3]['feature_names'])
