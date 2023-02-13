from model import *

model_test = TweetCheckerModel()
model_test.importData("train.csv")
model_test.prepare()
model_test.train_model()
