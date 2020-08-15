# Customer-Review-Classification
RNN-LSTM model based of Word Vectors To Classify Customer Reviews

Instructions :

1.Download The "glove.6B.200d.txt" Word Vector Text File and place it in the Folder
2.Run the "Word_To_Num_To_Vector.py" file to generate Word-Num and Num-Vector Mapping which is saved as pickle files
3.Run the "Extract_Reviews_And_Rating.py" file to extract Reviews and ratings from the dataset (Amazon.txt,imdb.txt and yelp.txt)
4.Run the "Remove_Unknown_Words.py" file to remove unknown symbols and characters from the reviews
5.Run the "Review_To_Array_Of_Keys.py" file to generate the final dataset consisting of reviews represented by array of keys and their respective ratings. This script generates both train and test set.
6.Finally Run the "trainer.py" to train the model.
