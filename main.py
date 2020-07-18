from discjockey import DiscJockey

def main():
    playlist_id = input("Enter a playlist id that has songs to your liking: ")

    while type(playlist_id) != str:
        playlist_id = input("invalid playlist id. Re-enter: ")

    # Creates party playlist off of your preferences
    dj = DiscJockey(playlist_id)

    track_id = input("Enter the track id you'd like to query: ")

    while type(track_id) != str:
        track_id = input("Invalid track id. Re-enter: ")

    # DJ takes your song request and decides if it's fit for the party or not.
    decision = dj.ask_dj(track_id)
    if decision:
        print("Your song has been accepted! Good choice!")
    else:
        print("Damn, straight up not a party song.  Learn some music taste, godamn")

if __name__ == "__main__":
    main()
