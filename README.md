# How to make a personal DJ in 5 simple steps

## Hack The Northeast hackathon submission.

### Created by @tedimitiku and @vaughncampos

We setout to build a content based recommender system that could act as your own personal DJ.

For information on how we created this DJ, take a look at models.ipynb.
To explore how we created the DJ, take a look take a discjockey.py
For all the data we used to train and test our recommender look at the musicdata folder.
To create your own DJ, take a look at and run main.py

To use:

1) Create environment using requirements.txt file on your computer.
2) Run main.py
3) Get your party playlist id and give it to the DJ.
Tog get a playlist id, navigate to your desired party playlist on spotify and copying the last part of the url.  For example my party playlist id looks like: 4RdNpG06Gzma4AuvfES6QR?si=PhZD_sK5T5C2kxkhQH9-lQ

The DJ will learn from you playlist and only accept songs that are similar to your party preferences. 

4) Test our your DJ by passing it a song track!
To get a track id, navigate to a song of your choice on spotify, then go to share --> Copy URI.
The URI should look something like this: spotify:track:2374M0fQpWi3dLnB54qaLX. Copy the last part after URI after track:
so your track id should look something like this: 2374M0fQpWi3dLnB54qaLX.

After inputing your party preferences and track, the DJ will tell you if he accepts your song choice or not!
