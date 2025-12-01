Purpose : 

    Integrating perceptron into the empty file.

Stages : 

    I used pandas for reading excel file. And divide two part => Train & Test.
    First I prepared the code for train. => I took X’s as X_train Except Last & First columns (ID & Y).
    Secondly I scaled the X’s. Thirdly I took the Y’s & turned these values into -1 and +1.
    After that I prepared the test data. I took the test X values with the same column range as the training set. (Except ID). Then I scaled the test X’s with the same mean and std calculated from the training data. I didn’t calculate new scaling values for the test set.
    Next step was implementing the perceptron algorithm. I initialized all w’s as zero and b as zero. Then for each epoch I checked all rows one by one. For every row I calculated z = w.x + b. If z >= 0, prediction = 1, else = -1. If the prediction was wrong, then I updated the weights => w = w + y[i] * X[i], and I updated the bias => b = b + y[i].
    After training, I made the predictions for the test data using the final w and b. The output was in (-1, 1) format, so I converted these values back to the original labels 2 and 4. (-1 -> 2, 1 -> 4).
    Lastly, I created a dataframe with the ID column and the Prediction column. And I saved this dataframe as a CSV file named “Results.csv”.
    
Note :

    While writing the code, I also added explanations next to the lines.
