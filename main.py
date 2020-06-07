import discjockey

def main():
    playlist_id = input("Enter a playlist id that has songs to your liking: ")

    while type(playlist_id) != str:
        playlist_id = input("invalid playlist id. Re-enter: ")

    dj = discjockey.DiscJockey(playlist_id)

    track_id = input("Enter the track id you'd like to query: ")

    while type(track_id) != str:
        track_id = input("Invalid track id. Re-enter: ")

    b = dj.ask_dj(track_id)
    if b:
        print("Your song has been accepted! Good choice!")
    else:
        print("Damn, straight up not a party song.  Learn some music taste godamn")

if __name__ == "__main__":
    main()
