# Bitcon-Price-Prediction

## GOAL
Predict the percentage change of Bitcoin one day into the future using the data collected from Messari
The gol of this project was to find, collect, clean and import data and find how accurate a simple feedforward linear neural network
could be in predicting price change. The model was suprisingly sucessful. The model typically underetimates price change when the percentage increases drastically but it also underestimates how drastic a significant drop will be. Obviously it struggles with the outliers where the price fluctuates greater than 10%. In the future I would like to update and perform with different paramiters, different currencies and use a 2d conv network. I would also like to add dropout layers in the future



## Libraries:
matplotlib
seaborn
pytorch
numpy
pandas
Path
