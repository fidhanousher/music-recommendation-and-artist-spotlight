import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('spotify_millionsongdata.csv')

# Preprocess the lyrics
tfidf = TfidfVectorizer(stop_words='english')
df['lyrics'] = df['lyrics'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['lyrics'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define the recommendation function
def recommend(song_name, artist_name):
    index = df[(df['song_name'] == song_name) & (df['artist_name'] == artist_name)].index[0]
    distances = sorted(list(enumerate(cosine_sim[index])), reverse=True, key=lambda x: x[1])
    recommended_songs = []
    for i in distances[1:6]:
        recommended_songs.append(df.iloc[i[0]]['song_name'])

    return recommended_songs

# Test the recommendation function
print(recommend('Shape of You', 'Ed Sheeran'))