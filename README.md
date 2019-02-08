# Capillary-Hackathon
Capillary machine learning hackathon hosted on Analytics Vidhya

## Problem Statement:
Recommender system for fashion retail. Develop an algorithm which will recommend best suited items from inventory to a user in order to improve his/her shopping experience.

## Data:
train.csv - transactions. It contains userID and productID

product_attributes.csv - product features (Fit, Color, Fabric etc)

images - image of apparel

## Approach:

* Transactions by each user are grouped and sorted based on time.
* Data is created such that for each UserId - [ list of productIDs purchased ]
* A word2vec model is built to capture the relations among the sequence of products.(temporal relations)
* User similarity is found using CountVectorizer on User-Item matrix. SVD is performed on the matrix.
* Top 5 nearest users for each UserId are found using Cosine similarity of SVD feature vector.
* Feature vector for Image of each product is obtained from VGG-19.
* One hot encoding of product attributes is performed.
* Sequence of productIDs purchased by each userID are given as input to Recurrent Neural Network to predict the next probable item to be purchased by the user.
* Each product is represented by the vector - [ word2vec representation + image feature vector + one hot encoding of product attribute + feature vectors of 5 nearest users]

## Performance

* Hit rate at 10 is calculated.

LEADERBOARD RANK - 40
