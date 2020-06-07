"""
Hack the Northeast
How to make a personal DJ in 5 simple steps
@author: tedimitiku, vaughncampos
"""
import os
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

#example party playlist: https://open.spotify.com/playlist/5ge2YqUbZrmqd2Mve8Uezf?si=mvHSkH6_R4ydd8cn9Fh_6Q
party_playlist_id = "5ge2YqUbZrmqd2Mve8Uezf?si=VVFB-RkdQMOpy1BffTeozQ"

#non party playlist: https://open.spotify.com/playlist/1vviyyoqxJyVpnNL4Cf6Xz?si=17SoNRa-RNuMVNG60DlWEg
non_party_playlist_id = "1vviyyoqxJyVpnNL4Cf6Xz?si=FtKKu3ICSaWPOY7st55HbQ"

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
music = playlist_to_df(music, non_party_playlist_id, 0)
music = music.sample(frac = 1)
music = music.reset_index();
music = music.drop(["index"], axis = 1)
music.head()

#Visualizing the data
df = music
target = 'party'
cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo']
i = 0
nRows = math.ceil(len(cols) / 3)
nCols = 3
f = plt.figure(figsize=(17,15))

for col in cols:
    i += 1
    ax = f.add_subplot(nRows * 100 + nCols * 10 + i)
    ax.scatter(df[col],df[target], c = df[target]) #np.vectorize(color_chooser)(df[target]))
    ax.axvline(np.mean(df[df[target] == 1][col]), c='y')
    ax.axvline(np.mean(df[df[target] == 0][col]), c='m')

    ax.set_xlabel(col)
    ax.set_ylabel(target)
    ax.set_title(str(col) + " vs. " + target)

f.subplots_adjust(wspace = 0.2, hspace=0.3)
plt.show()

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

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

#Random Tree Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=2, random_state=0)
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)

# Metrics for K Nearest Neighbors
print("Accuracy" + str(accuracy_score(y_test, y_pred_kNN)))
print(classification_report(y_test, y_pred_kNN))
print(confusion_matrix(y_test, y_pred_kNN))

# Metrics for Logistic Regression
print("Accuracy" + str(accuracy_score(y_test, y_pred_lr)))
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

# Metrics for Random Tree Classifier
print("Accuracy" + str(accuracy_score(y_test, y_pred_rfc)))
print(classification_report(y_test, y_pred_rfc))
print(confusion_matrix(y_test, y_pred_rfc))
