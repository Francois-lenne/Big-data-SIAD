from model import *

model_test = TweetCheckerModel()
model_test.importData("https://raw.githubusercontent.com/Francois-lenne/Big-data-SIAD/main/train.csv")
model_test.prepare()
model_test.train_model()