import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv("./data/data.csv")

# The current UFC rules have been implemented after this date.
# The fights before, therefore, need to be excluded.

limit_date = "2001-04-01"
df = df[(df["date"] > limit_date)]

# Let's take a look at the NA's.

na = []
for index, col in enumerate(df):
    na.append((index, df[col].isna().sum() / len(df.index)) )
na_sorted = na.copy()
na_sorted.sort(key = lambda x: x[1], reverse = True)

for i in range(len(df.columns)):
    print(df.columns[na_sorted[i][0]], ":", na_sorted[i][1], "NaN %")

# Most NA's stem from missing fighters' fight stats.
# Let's remove the corresponding fights (rows).

na_features = ["B_avg_BODY_att", "R_avg_BODY_att"]
df.dropna(subset = na_features, inplace = True)

# We also remove features that have no bearing on the outcome
# or are irrelevant for other reasons (eg a factor variable that has
# a continuous parallel).

feat_rm = ["Referee", "location", "weight_class", "R_draw", "B_draw",
               "R_fighter", "B_fighter", "date"]

df.drop(feat_rm, axis = 1, inplace = True)

# Let's also exclude fights that ended in a draw.

df = df[df["Winner"].isin(["Red", "Blue"])]

# Transform the dependent feature into binary

df["Winner"] = df["Winner"].map({"Red":1, "Blue":0})



# To prevent data leakage in the preprocessing step,
# let's use the pipeline.

# Preprocessing
# Imputate:
# Median for age
# Multiple imputation for height, reach, weight
# Most frequent for stance
# Transform:
# categorical to one hot encoding

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

feat_med = ["R_age", "B_age"]
feat_freq = ["R_Stance", "B_Stance"]
feat_mult = ["R_Height_cms", "B_Height_cms",
             "R_Reach_cms", "B_Reach_cms"]

med_transformer = \
    Pipeline(steps = [("imputer", SimpleImputer(strategy = "median"))])

freq_transformer = \
    Pipeline(steps = [("imputer", SimpleImputer(strategy = "most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown ="ignore"))])

mult_transformer = \
    Pipeline(steps = [("imputer", IterativeImputer(estimator = ExtraTreesRegressor(), max_iter = 10, random_state = 123))])

preprocessor = ColumnTransformer(transformers = [("med", med_transformer, feat_med),
                                                 ("freq", freq_transformer, feat_freq),
                                                 ("mult", mult_transformer, feat_mult)])

# Train / test split

X = df.drop(["Winner"], axis = 1)
y = df["Winner"]

X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size = 0.25, random_state = 123)


encoder = preprocessor.fit(X_train)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

# xgBoost classifier

# Let's start with hyperparameter tuning using cross validation via hyperopt.

from hyperopt import fmin, tpe, hp, anneal, Trials
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Creat 5-fold for CV

fold_cv = StratifiedKFold(n_splits = 5)

# Define the metric to score the performance

def xgb_cv(params, random_state = 123, cv = fold_cv, X = X_train, y = y_train):
    # the function gets a set of variable parameters in "param"
    params = {"max_depth": int(params["max_depth"]),
              "gamma": int(params["gamma"]),
              "reg_alpha": int(params["reg_alpha"]),
              "min_child_weight": int(params["min_child_weight"]),
              "eta": params["eta"],
              "colsample_bytree": params["colsample_bytree"]
              }

    #xgboost model
    model = XGBClassifier(random_state = 123, **params)

    # cross validation score metric
    score = cross_val_score(model, X, y, cv = cv, scoring = "roc_auc", n_jobs = -1).mean()

    return score

# Define space for parameter search

space = {"max_depth": hp.quniform("max_depth", 5, 50, 1),
         "gamma": hp.uniform("gamma", 1, 10),
         "reg_alpha": hp.quniform("reg_alpha", 20, 180, 5),
         "min_child_weight": hp.quniform('min_child_weight', 0, 10, 1),
         "eta": hp.loguniform("eta", -5, -1),
         "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
         }

trials = Trials()

# Perform search

best = fmin(fn = xgb_cv, # function to optimize
            space = space,
            algo = tpe.suggest,
            max_evals = 50, # max iterations
            trials = trials,
            rstate = np.random.RandomState(123) # fixing random state for the reproducibility
         )

# Let's evaluate the best model on test data

model = XGBClassifier(random_state = 123,
                      max_depth = int(best["max_depth"]),
                      gamma = int(best["gamma"]),
                      reg_alpha = int(best["reg_alpha"]),
                      min_child_weight = int(best["min_child_weight"]),
                      eta = best["eta"],
                      colsample_bytree = best["colsample_bytree"])
model.fit(X_train, y_train)
test_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

# Confusion matrix

conf_mat = confusion_matrix(y_test, y_pred)
ax = plt.subplot()
sns.heatmap(conf_mat, annot = True, ax = ax, fmt = "d")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Blue", "Red"])
ax.yaxis.set_ticklabels(["Blue", "Red"])
###############################################################