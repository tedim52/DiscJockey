Created by @tedimitiku and @vaughncampos

We setout to build a content based recommender system by training and selecting an ML classification model that could determine whether a song is fit enough for your party preferences. We encapsulated this recommender in a DiscJockey class that can take your party preferences and use them to determine whether other songs are fit enough for the party of your liking. 

For info on how we created and selected our model look at models.ipynb 
For the Disc Jockey class look at discjockey.py
For all the data we used to train and test our recommender look at the musicdata folder
To test out our recommender look at main.py

To use:
First input a playlist id of your playlist.  This can be done by navigating to the desired playlist and copying the last part of the url.  For example my party playlist id looks like: 4RdNpG06Gzma4AuvfES6QR?si=PhZD_sK5T5C2kxkhQH9-lQ

Then the DJ will learn from you playlist and only accept songs that are similar to your party preferences. 

To get a track id, navigate to a song of your choice on spotify, then go to share>copy uri.
The uri should look something like this: spotify:track:2374M0fQpWi3dLnB54qaLX. Then copy the last part after track:.
so you track id should look something like this: 2374M0fQpWi3dLnB54qaLX.

After inputing your party preferences and track, the DJ will tell you if he accepts your song choice!
