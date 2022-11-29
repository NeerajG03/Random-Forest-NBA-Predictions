from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit

def accuracy_forupset(value):
    return value + ((value)*(37/100))

df = pd.read_csv("./raw_data/nba_games.csv", index_col=0)
df.sort_values("date",inplace=True)
df.reset_index(drop=True,inplace=True)
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group

df = df.groupby("team", group_keys=False).apply(add_target)

df["target"][pd.isnull(df["target"])] = 2 # mark the future games in target as null
df["target"] = df["target"].astype(int, errors="ignore")

nulls = pd.isnull(df).sum()
nulls = nulls[nulls > 0]
valid_columns = df.columns[~df.columns.isin(nulls.index)]
df = df[valid_columns].copy()

# models

rcl = RidgeClassifier(alpha=1) #initialize classifier
split = TimeSeriesSplit(n_splits=3) # split based of time to keep the value for prediction sequential
sfs = SequentialFeatureSelector(rcl, n_features_to_select=35, direction="forward",cv=split,n_jobs=1)


removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]


scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])
sfs.fit(df[selected_columns], df["target"])

predictors = list(selected_columns[sfs.get_support()])
print(predictors,"\nno. of predicted columns =",len(predictors))

def modelstart(data, team, opposition, model, predictors, start=5, step=1):
    all_predictions = []
    
    seasons = sorted(data["season"].unique())
    
    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data.loc[(data["team"] == team) & (data["team_opp"]==opposition)]
        # print(test)
        
        model.fit(train[predictors], train["target"])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        
        all_predictions.append(combined)
    predictions = pd.concat(all_predictions)
    return predictions["prediction"].mean()