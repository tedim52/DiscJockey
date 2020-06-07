"""
Hack the Northeast
How to make a personal DJ in 5 simple steps
@author: tedimitiku, vaughncampos
"""
import os
import pandas as pd
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#Spotify API Setup
SPOTIPY_CLIENT_ID = os.environ["SPOTIPY_CLIENT_ID"]
SPOTIPY_CLIENT_SECRET = os.environ["SPOTIPY_CLIENT_SECRET"]
cc = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=cc)

#Create music dataframe of party songs and non party songs
columns = ["track_id", "song", "acousticness", "danceability", "energy",
            "instrumentalness", "liveness", "loudness", "speechiness",
            "valence", "tempo", "party"]
music = pd.DataFrame(columns=columns)

#Populating dataframe with song features
party_playlist_id = "5ge2YqUbZrmqd2Mve8Uezf?si=VVFB-RkdQMOpy1BffTeozQ"
non_party_playlist_id = "5hCRFgctanZE1v1XzTDim4?si=M3NmQZrwTJOElCUKVYCzZg"

def song_to_df(track_id, song_name, party):
    song_features = sp.audio_features(track_id)[0]
    song_data = {'track_id': track_id,
                 'song': song_name,
                 'acousticness': song_features.get("acousticness"),
                 'danceability': song_features.get("danceability"),
                 'energy': song_features.get("energy"),
                 'instrumentalness': song_features.get("instrumentalness"),
                 'liveness': song_features.get("liveness"),
                 'loudness': song_features.get("loudness"),
                 'speechiness': song_features.get("speechiness"),
                 'valence': song_features.get("valence"),
                 'tempo': song_features.get("tempo"),
                 'party': party
                }
    return song_data

def playlist_to_df(df, playlist_id, party):
    songs = sp.playlist_tracks(playlist_id).get("tracks").get("items")
    music_frame = df
    for song in songs:
        track = song.get("track")
        song_name = track.get("name")
        track_id = track.get("id")
        song_data = song_to_df(track_id, song_name, party)
        music_frame = music_frame.append(song_data, ignore_index=True)

        #Get five more songs Spotify says is like this song
        recommendations = sp.recommendations(seed_tracks=[track_id], limit = 5).get("tracks")
        for recommendation in recommendations:
            r_song_name = recommendation.get("name")
            r_track_id = recommendation.get("id")
            r_song_data = song_to_df(r_track_id, r_song_name, party)
            music_frame = music_frame.append(r_song_data, ignore_index=True)
    return music_frame

music = playlist_to_df(music, party_playlist_id, 1)
#Visualize audio features of party songs


music = playlist_to_df(music, non_party_playlist_id, 0)
music = music.sample(frac = 1)
music = music.reset_index();
music = music.drop(["index"], axis = 1)
music.to_csv (r'/Users/tewodrosmitiku/Desktop/Hack/music/music_data.csv', index = False, header=True)
print(music)

#Splitting dataframe into train and testing data
features = music.drop(["track_id", "song", "party"], axis = 1)
target = music["party"]
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42)
y_train = y_train.astype("int")
y_test = y_test.astype("int")

#Normalizing values
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#KNNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier(n_neighbors=5)
kNN.fit(x_train, y_train)
y_pred_kNN = kNN.predict(x_test)

print("Accuracy" + str(accuracy_score(y_test, y_pred_kNN)))
print(classification_report(y_test, y_pred_kNN))
print(confusion_matrix(y_test, y_pred_kNN))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

print("Accuracy" + str(accuracy_score(y_test, y_pred_lr)))
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

#Random Tree Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=2, random_state=0)
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)

print("Accuracy" + str(accuracy_score(y_test, y_pred_rfc)))
print(classification_report(y_test, y_pred_rfc))
print(confusion_matrix(y_test, y_pred_rfc))

#Visualize audio features of predicted party songs


song1 = song_to_df("1zi7xx7UVEFkmKfv06H8x0", "One Dance", 1)
song2 = song_to_df("1jaTQ3nqY3oAAYyCTbIvnM", "Whats Poppin", 1)
song3 = song_to_df("6u0dQik0aif7FQlrhycG1L", "The Morning", 0)
song4 = song_to_df("5IRLnB7JqTMcIlMtE0Rcuv", "Reverse Faults", 0)
song5 = song_to_df("5ehVOwEZ1Q7Ckkdtq0dY1W", "Lofi", 0)

test_song = pd.DataFrame(columns=columns)
test_song = test_song.append(song1, ignore_index=True)
test_song = test_song.append(song2, ignore_index=True)
test_song = test_song.append(song3, ignore_index=True)
test_song = test_song.append(song4, ignore_index=True)
test_song = test_song.append(song5, ignore_index=True)
test_song = test_song.drop(["track_id", "song", "party"], axis=1)
print(test_song)

test_pred_kNN = kNN.predict(test_song)
test_pred_lr = lr.predict(test_song)
test_pred_rfc = rfc.predict(test_song)
print(test_song)
print(test_pred_kNN)
print(test_pred_lr)
print(test_pred_rfc)
