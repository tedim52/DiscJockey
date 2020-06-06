#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:26:46 2020

@author: vaughncampos
"""

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Spotipy API Setup
SPOTIPY_CLIENT_ID = "236e81909708434598e63e00fe671955"
SPOTIPY_CLIENT_SECRET = "5d574a1eb8f940b783b72b00c5eb4658"
cc = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
spotify = spotipy.Spotify(client_credentials_manager=cc)

def getIdsandNames(playlist):
    ids = []
    song_names = []
    playlist_id = playlist.get("id")
    songs_list = spotify.playlist_tracks(playlist_id).get("items")
    for track in songs_list:
        track_id = track.get("track").get("id")
        song_names.append(track.get("track").get("name"))
        ids.append(track_id)
        track_recs = spotify.recommendations(seed_tracks = [track_id], limit = 5)["tracks"]
        for rec_track in track_recs:
            ids.append(rec_track.get("id"))
            song_names.append(rec_track.get("name"))
    return ids[:100], song_names[:100]

def getAudioFeatures(ids, songNames, partystatus):
    songs_with_audio_features = spotify.audio_features(ids[:100])
    acousticness = []
    danceability = []
    energy = []
    instrumentalness = []
    liveness = []
    loudness = []
    speechiness = []
    valence = []
    tempo = []
    party = []
    for song in songs_with_audio_features:
        acousticness.append(song['acousticness'])
        danceability.append(song['danceability'])
        energy.append(song['energy'])
        instrumentalness.append(song['instrumentalness'])
        liveness.append(song['liveness'])
        loudness.append(song['loudness'])
        speechiness.append(song['speechiness'])
        valence.append(song['valence'])
        tempo.append(song['tempo'])
        if partystatus:
            party.append(1)
        else:
            party.append(0)
    
    df = pd.DataFrame()
    df['track id'] = ids
    df['name'] = songNames
    df['acousticness'] = acousticness
    df['danceability'] = danceability
    df['energy'] = energy
    df['instrumentalness'] = instrumentalness
    df['liveness'] = liveness
    df['loudness'] = loudness
    df['speechiness'] = speechiness
    df['valence'] = valence
    df['tempo'] = tempo
    df['party'] = party
    
    return df


#Vaughn's Spotify playlists
playlists = spotify.user_playlists("22ud4xauqzyrwyonwznwfscga").get("items")
    
non_party = playlists[0]
party = playlists[1]

party_ids, party_names = getIdsandNames(party)
party_df = getAudioFeatures(party_ids, party_names, True)

non_party_ids, non_party_names = getIdsandNames(non_party)
non_party_df = getAudioFeatures(non_party_ids, non_party_names, False)

random_df = party_df.append(non_party_df).sample(frac = 1)

x_train, x_test, y_train, y_test = train_test_split(random_df.drop(['track id', 'name', 'party'], axis = 1), random_df['party'], test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(5)
model.fit(x_train, y_train)
preds = model.predict(x_test)
print(accuracy_score(preds, y_test))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
preds = model.predict(x_test)
print(accuracy_score(preds, y_test))

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)
preds = svclassifier.predict(x_test)
print(accuracy_score(preds, y_test))

import matplotlib.pyplot as plt
df = x_test.copy()
df['party'] = preds
party_songs = df.loc[df['party'] == 1]
non_party_songs = df.loc[df['party'] == 0]

plt.scatter(party_songs['tempo'], party_songs['danceability'])
plt.scatter(party_songs['tempo'], party_songs['acousticness'])
plt.scatter(party_songs['tempo'], party_songs['energy'])
plt.scatter(party_songs['tempo'], party_songs['instrumentalness'])
plt.scatter(party_songs['tempo'], party_songs['speechiness'])
plt.legend(['Dance', 'Acou', 'Energy', 'Instr', 'Speech'])
plt.show()