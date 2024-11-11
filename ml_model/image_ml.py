### imports ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    train_test_split,
)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier
from lightgbm.sklearn import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC



examples = np.load("ml_model/examples.npy")

data = {
    "image": [examples]
}

image_df = pd.DataFrame(examples, columns=len(examples[0])*["beign"]).T

