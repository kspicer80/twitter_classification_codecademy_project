import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
pd.options.display.width = 0

viral_tweets = pd.read_json('random_tweets.json', lines=True)
#print(viral_tweets.columns)
#print(viral_tweets.head())
#print(viral_tweets.shape)
#print(viral_tweets['text'][0])

retweet_counts = viral_tweets['retweet_count']
median_retweet_count = viral_tweets['retweet_count'].median()
mean_retweet_count = viral_tweets['retweet_count'].mean()

print(viral_tweets.agg(
    {
        "retweet_count": ["min", "max", "median", 'mean'],
        "favorite_count": ["min", "max", "median", "mean"],
    }
))

fig, ax = plt.subplots()
retweet_counts.plot.hist(bins=25, color='tab:cyan')
plt.axvline(x = median_retweet_count, color = 'red', label = f'median value of retweets in the dataset: {median_retweet_count: .0f}')
plt.axvline(x = mean_retweet_count, color = 'green', label = f'mean value of retweets in the dataset: {mean_retweet_count: .0f}')
plt.legend()
plt.show()

viral_tweets['is_viral'] = viral_tweets['retweet_count'].apply(lambda x: 1 if x > median_retweet_count else 0)
#print(viral_tweets['is_viral'].value_counts())

viral_tweets['tweet_length'] = viral_tweets.apply(lambda x: len(x['text']), axis=1)
viral_tweets['followers_count'] = viral_tweets.apply(lambda x: x['user']['followers_count'], axis=1)
viral_tweets['friends_count'] = viral_tweets.apply(lambda x: x['user']['friends_count'], axis=1)

labels = viral_tweets['is_viral']
cols_to_keep = ['tweet_length', 'followers_count', 'friends_count']

new_df = viral_tweets[cols_to_keep]
scaled_df = scale(new_df, axis=0)
#print(scaled_df[0])

train_data, test_data, train_labels, test_labels = train_test_split(scaled_df, labels, test_size = 0.2, random_state = 1)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(train_data, train_labels)
print(classifier.score(test_data, test_labels))

scores = []
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(train_data, train_labels)
    scores.append(classifier.score(test_data, test_labels))
plt.plot(range(1,200), scores)
plt.legend(['features: tweet length, # of followers, # of friends'])
#plt.title('KMeans Nearest Neighbors Accuracy Scores for Classifying "Viral Tweets" ("Viral" = Retweets > Median # of Retweets)')

viral_tweets['number_of_hashtags'] = viral_tweets['text'].apply(lambda x: x.count('#'))

hashtag_col_added = ['tweet_length', 'followers_count', 'friends_count', 'number_of_hashtags']
hashtag_df = viral_tweets[hashtag_col_added]
scaled_hashtag_df = scale(hashtag_df, axis=0)
h_train_data, h_test_data, h_train_labels, h_test_labels = train_test_split(scaled_hashtag_df, labels, test_size=0.2, random_state=1)
classifier.fit(h_train_data, h_train_labels)
print(classifier.score(h_test_data, h_test_labels))

h_scores = []
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(h_train_data, h_train_labels)
    h_scores.append(classifier.score(h_test_data, h_test_labels))
plt.plot(range(1, 200), h_scores)
#plt.legend(['features: tweet length, # of followers, # of friends', 'features: tweet length, # of followers, # of friends, # of hashtags'])
#plt.title('KMeans Nearest Neighbors Accuracy Scores for Classifying "Viral Tweets" ("Viral" Means > Median # of Retweets)')
#plt.show()

fav_col_added = ['tweet_length', 'followers_count', 'friends_count', 'number_of_hashtags', 'favorite_count']
fav_df = viral_tweets[fav_col_added]
scaled_fav_df = scale(fav_df, axis=0)
f_train_data, f_test_data, f_train_labels, f_test_labels = train_test_split(fav_df, labels, test_size=0.2, random_state=1)
classifier.fit(f_train_data, f_train_labels)
print(classifier.score(f_test_data, f_test_labels))

fav_scores = []
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(f_train_data, f_train_labels)
    fav_scores.append(classifier.score(f_test_data, f_test_labels))
plt.plot(range(1, 200), fav_scores)

length_friend_favorite = ['tweet_length', 'friends_count', 'favorite_count']
lff_df = viral_tweets[length_friend_favorite]
scaled_lff_df = scale(lff_df, axis=0)
lff_train_data, lff_test_data, lff_train_labels, lff_test_labels = train_test_split(lff_df, labels, test_size=0.2, random_state=1)
classifier.fit(lff_train_data, lff_train_labels)

lff_scores = []
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(lff_train_data, lff_train_labels)
    lff_scores.append(classifier.score(lff_test_data, lff_test_labels))

plt.plot(range(1, 200), lff_scores)
plt.legend(['features: tweet length, # of followers, # of friends', 'features: tweet length, # of followers, # of friends, # of hashtags', 'features: tweet length, # of followers, # of friends, # of hashtags, # of favorites', 'features: tweet_length, friends_count, favorite_count'])
plt.title('KMeans Nearest Neighbors Accuracy Scores for Classifying "Viral Tweets" ("Viral" = Retweets > Median # of Retweets)')
plt.show()
