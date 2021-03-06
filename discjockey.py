import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#Spotify API Setup
SPOTIPY_CLIENT_ID = "236e81909708434598e63e00fe671955"
SPOTIPY_CLIENT_SECRET = "5d574a1eb8f940b783b72b00c5eb4658"
cc = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=cc)


class DiscJockey():
    """ A class that imitates a DJ.

    You can tell this DJ what your favorite party songs are and it will learn your preferences
    so that when your party attendees suggest songs, this DJ knows which ones to add to the queue
    and which ones to throw out.

    Attributes:
    party_preference: A Spotify playlist id containing your favorite party songs.
    audio_features: Audio features this DJ uses to classify your preferences.
    music_data: The DJ's song library.
    model: The recommender.
    """
    def __init__(self, playlist_id):
        self.party_preference = playlist_id
        self.audio_features = ["track_id", "song", "acousticness", "danceability", "energy",
                    "instrumentalness", "liveness", "loudness", "speechiness",
                    "valence", "tempo", "party"]
        self.music_data = self.__datafy()
        self.model = self.__create_recommender()

    def ask_dj(self, track_id):
        """ Takes in a Spotify track and returns a yes or no from the DJ based on your preferences."""
        song_data = self.__song_to_df(track_id, "", -1)
        df = pd.DataFrame(song_data, index=[0])
        song_features = df.drop(["track_id", "song", 'party'], axis = 1)
        outcome = self.model.predict(song_features)
        song_features['party'] = outcome
        return (song_features['party'] == 1).bool()

    def __create_recommender(self):
        """ Creates the the DJ recommender """
        features = self.music_data.drop(["track_id", "song", "party"], axis = 1)
        target = self.music_data["party"]
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42)
        y_train = y_train.astype("int")

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        model = LogisticRegression()
        model.fit(x_train, y_train)
        return model

    def __datafy(self):
        music_data = pd.DataFrame(columns=self.audio_features)
        music_data = music_data.append(self.__playlist_to_df(self.party_preference, 1))
        non_party_data = pd.read_csv("nonpartymusicdata.csv")
        music_data = music_data.append(non_party_data)
        music_data = music_data.sample(frac = 1)
        music_data = music_data.reset_index()
        music_data = music_data.drop(["index"], axis = 1)
        return music_data

    def __playlist_to_df(self, playlist_id, party):
        music_data = pd.DataFrame(columns=self.audio_features)
        songs = sp.playlist_tracks(playlist_id).get("tracks").get("items")
        for song in songs:
            track = song.get("track")
            song_name = track.get("name")
            track_id = track.get("id")
            song_data = self.__song_to_df(track_id, song_name, party)
            music_data = music_data.append(song_data, ignore_index=True)

            #Get five more songs that spotify recommends of this song
            recommendations = sp.recommendations(seed_tracks=[track_id], limit = 5).get("tracks")
            for recommendation in recommendations:
                r_song_name = recommendation.get("name")
                r_track_id = recommendation.get("id")
                r_song_data = self.__song_to_df(r_track_id, r_song_name, party)
                music_data = music_data.append(r_song_data, ignore_index=True)
        return music_data

    def __song_to_df(self, track_id, song_name, party):
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
