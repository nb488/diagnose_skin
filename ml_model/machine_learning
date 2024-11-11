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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

example1 = np.load("example.npy")

### load in dataset ###
skin_description = pd.read_csv("data/Symptom2Disease.csv")
skin_description.shape # 1200 text and labels

# Train into testing and training data
skin_train, skin_test = train_test_split(skin_description, train_size = 0.7, random_state=123)

### Exploratory data analysis ###
skin_train["label"].unique()

counts = skin_train["label"].value_counts().nlargest(20)
plt.bar(counts.index, counts.values)
plt.xlabel("label")
plt.ylabel("count")
plt.xticks(rotation=90)
plt.title("label" + " Histogram")
plt.show()
# approximately 35 to 40 cases of each skin type (24 skin types)


### Separate traning and testing data ###
train_X = skin_train["text"]
train_y = skin_train["label"]

test_X = skin_test["text"]
test_y = skin_train["label"]

### Define transformers ###
bag_feats = train_X
cv_transformer = CountVectorizer(max_features = 10, stop_words='english', dtype = "float64")


### Make dummy model ###
dummy_pipe1 = make_pipeline(cv_transformer, DummyClassifier(strategy="most_frequent"))
dummy_pipe2 = make_pipeline(cv_transformer, DecisionTreeClassifier())

results = {}

results["Dummy Classifier"] = cross_validate(dummy_pipe1, train_X, train_y, cv =5, scoring = "recall_macro")
results["Decision Tree"] = cross_validate(dummy_pipe2, train_X, train_y, cv =5, scoring = "recall_macro")

results_df = pd.DataFrame(results) #TODO: make the test_score a mean

### Make real models ###
pipe_catboost = make_pipeline(
    cv_transformer,
    CatBoostClassifier(verbose=0, random_state=123),
)

pipe_randomforest = make_pipeline(
    cv_transformer,
    RandomForestClassifier(n_estimators=100, random_state=123)
)

pipe_multinomialnb = make_pipeline(
    cv_transformer,
    MultinomialNB()
)

pipe_adaboost = make_pipeline(
    cv_transformer,
    AdaBoostClassifier(n_estimators=50)
)

pipe_scv = make_pipeline(
    cv_transformer,
    SVC(kernel='linear')
)

pipe_scv_optimized = make_pipeline(
    cv_transformer,
    SVC(C=100, gamma= 0.1)
)

pipe_knn = make_pipeline(
    cv_transformer,
    KNeighborsClassifier()
)

results["CatBoost"] = cross_validate(pipe_catboost, train_X, train_y, cv=5, return_train_score= True, scoring="recall_macro")

results["random forest"] = cross_validate(pipe_randomforest, train_X, train_y, cv=5, return_train_score= True, scoring="recall_macro")

results["MultinomialNB"] = cross_validate(pipe_multinomialnb, train_X, train_y, cv=5, return_train_score= True, scoring="recall_macro")

results["adaboost"] = cross_validate(pipe_adaboost, train_X, train_y, cv=5, return_train_score= True, scoring="recall_macro")

results["scv"] = cross_validate(pipe_scv, train_X, train_y, cv=5, return_train_score= True, scoring="recall_macro")

results["pipe_scv_optimized"] = cross_validate(pipe_scv, train_X, train_y, cv=5, return_train_score= True, scoring="recall_macro")

results["pipe_knn"] = cross_validate(pipe_knn, train_X, train_y, cv=5, return_train_score= True, scoring="recall_macro")




ensemble_model = VotingClassifier(
    estimators=[
        ('random_forest', RandomForestClassifier(n_estimators=100, random_state=123)),
        ('multinomial_nb', MultinomialNB()),
        ('adaboost', AdaBoostClassifier(n_estimators=50)),
        ('svc', SVC(C=100, gamma=0.1, probability=True)),
        ('knn', KNeighborsClassifier()),
        ('catboost', CatBoostClassifier(verbose=0, random_state=123))
    ],
    voting='hard'
)

# Create a pipeline with a transformer and the ensemble model
pipe_monster = make_pipeline(
    cv_transformer,
    ensemble_model
)

results["monster"] = cross_validate(pipe_knn, train_X, train_y, cv=5, return_train_score= True, scoring="recall_macro")




param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
}

results_dict = {"C": [], "gamma": [], "mean_cv_score": []}
best_score = 0


new_example = ["I have a rash on my arm, it is red and itchy"]
pipe_scv.fit(train_X, train_y)
pipe_scv.predict(new_example)