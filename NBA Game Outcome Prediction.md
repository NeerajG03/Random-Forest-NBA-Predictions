## NBA Game Outcome Prediction 
This document serves as a manual on the workings and the how to of the jupyter notebook given as the submission.

The jupyter notebook runs a ridge classifier model which is trained and works out predictions for each For and Against team in order to predict the outcome.

The final cell displays all the team's abbreviations. This can be given as input into the text box that pops up when running the second to last cell.

#### In order to run this you will need the following:
- SKLearn (installation dependency)
- Pandas (installation dependency)

#### The working:
The model outputs 3 values after the prediction function is run :
- **The Winning percentage**: This value shows how likely TeamA is to win against TeamB.
- **Accuracy metric**: This value shows how certain the prediction is.
- **Up/Down inconsistency**: This value shows how far off the accuracy metric can be.