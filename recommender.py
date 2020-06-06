"""
Hack the Northeast
Make your own DJ
@author: tedimitiku
"""

import os
#python data libraries
import numpy as np
import pandas as pd


#spotipy info
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#data visualization libraries
import matplotlib
import seaborn as sns

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

def song_to_json(track_id, song_name, party):
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

def song_features_to_df(df, playlist_id, party):
    songs = sp.playlist_tracks(playlist_id).get("tracks").get("items")
    music_frame = df
    for song in songs:
        track = song.get("track")
        song_name = track.get("name")
        track_id = track.get("id")
        song_data = song_to_json(track_id, song_name, party)
        music_frame = music_frame.append(song_data, ignore_index=True)

        #Get five more songs that spotify recommends of this song
        recommendations = sp.recommendations(seed_tracks=[track_id], limit = 5).get("tracks")
        for recommendation in recommendations:
            r_song_name = recommendation.get("name")
            r_track_id = recommendation.get("id")
            r_song_data = song_to_json(r_track_id, r_song_name, party)
            music_frame = music_frame.append(r_song_data, ignore_index=True)
    return music_frame

music = song_features_to_df(music, party_playlist_id, 1)
music = song_features_to_df(music, non_party_playlist_id, 0)
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
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred2 = model.predict(x_test)

print(accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))

#test one dance, nonstop, the morning
song1 = song_to_json("1zi7xx7UVEFkmKfv06H8x0", "One Dance", 1)
song2 = song_to_json("1jaTQ3nqY3oAAYyCTbIvnM", "Whats Poppin", 1)
song3 = song_to_json("6u0dQik0aif7FQlrhycG1L", "The Morning", 0)
song4 = song_to_json("5IRLnB7JqTMcIlMtE0Rcuv", "Reverse Faults", 0)
song5 = song_to_json("5ehVOwEZ1Q7Ckkdtq0dY1W", "Lofi", 0)

test_song = pd.DataFrame(columns=columns)
test_song = test_song.append(song1, ignore_index=True)
test_song = test_song.append(song2, ignore_index=True)
test_song = test_song.append(song3, ignore_index=True)
test_song = test_song.append(song4, ignore_index=True)
test_song = test_song.append(song5, ignore_index=True)
print(test_song)
test_song = test_song.drop(["track_id", "song", "party"], axis=1)
test_pred = classifier.predict(test_song)
print(test_pred)
print("for the memes")
#Visualize Data


#Feed into models with features and label

#Test accuracy of models

#Find most accurate DJ
