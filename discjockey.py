import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#Spotify API Setup
SPOTIPY_CLIENT_ID = os.environ["SPOTIPY_CLIENT_ID"]
SPOTIPY_CLIENT_SECRET = os.environ["SPOTIPY_CLIENT_SECRET"]
cc = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=cc)


class DiscJockey():
    def __init__(self, playlist_id):
        self.model = self.createRecommender(playlist_id)
        self.party_preference = playlist_id
        self.id = playlist_id

    #Docstring needed
    def ask_dj(self, track_id):
        return

    #Docstring needed
    def curate(self, playlist, model):
        df = self.playlist_to_df(playlist)
        currated_playlist = df.loc[df['party'] == 1]
        return currated_playlist

    #Docstring needed
    def createRecommender(self):
        columns = ["track_id", "song", "acousticness", "danceability", "energy",
                    "instrumentalness", "liveness", "loudness", "speechiness",
                    "valence", "tempo", "party"]
        music_data = pd.DataFrame(columns=columns)
        party_playlist_id = self.party_preference
        non_party_playlist_id = "5hCRFgctanZE1v1XzTDim4?si=M3NmQZrwTJOElCUKVYCzZg"

        music_data = playlist_to_df(music_data, party_playlist_id)
        music_data = playlist_to_df(music_data, non_party_playlist_id)
        music_data = music_data.sample(frac = 1)
        music_data = music_data.reset_index();
        music_data = musi_data.drop(["index"], axis = 1)
        x_train, x_test, y_train, y_test = train_test_split(random_df.drop(['track id', 'name', 'party'], axis = 1), random_df['party'], test_size = 0.3)

        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(5)
        model.train(x_train, y_train)

        return model

    #Docstring needed
    def quality(self):
        return






    def playlist_to_df(self, df, playlist):
        songs = sp.playlist_tracks(playlist_id).get("tracks").get("items")
        music_frame = df
        for song in songs:
            track = song.get("track")
            song_name = track.get("name")
            track_id = track.get("id")
            song_data = song_to_df(track_id, song_name, party)
            music_frame = music_frame.append(song_data, ignore_index=True)

            #Get five more songs that spotify recommends of this song
            recommendations = sp.recommendations(seed_tracks=[track_id], limit = 5).get("tracks")
            for recommendation in recommendations:
                r_song_name = recommendation.get("name")
                r_track_id = recommendation.get("id")
                r_song_data = song_to_df(r_track_id, r_song_name, party)
                music_frame = music_frame.append(r_song_data, ignore_index=True)
        return music_frame

    def song_to_df(self, playlist):
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
