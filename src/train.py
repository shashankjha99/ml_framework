from sys import displayhook
from numpy.lib.function_base import disp
import pandas as pd
import os
from sklearn import preprocessing
from sklearn import ensemble,metrics
import dispatcher
import joblib

TRAINING_DATA = os.environ.get('TRAINING_DATA')
FOLD =int(os.environ.get('FOLD'))
MODEL  = os.environ.get('MODEL')

fold_mapping = {
    0 : [1,2,3,4],
    1 : [0,2,3,4],
    2 : [0,1,3,4],
    3 : [0,2,1,4],
    4 : [0,2,3,1],
}


if __name__=='__main__':
    df = pd.read_csv(TRAINING_DATA)
    df_train = df[df.kfold.isin(fold_mapping.get(FOLD))]
    df_test = df[df.kfold == FOLD]

    ytrain = df_train.target.values
    yval = df_test.target.values

    df_train = df_train.drop(['id','target','kfold'],axis=1)
    df_test = df_test.drop(['id','target','kfold'],axis=1)

    df_test= df_test[df_train.columns]

    label_encoders =[]
    for c in df_train.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df_train[c].values.tolist()+df_test[c].values.tolist())
        df_train.loc[:,c]=lbl.transform(df_train[c].values.tolist())
        df_test.loc[:,c]=lbl.transform(df_test[c].values.tolist())
        label_encoders.append((c,lbl))
    
    # clf = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,verbose=2)
    clf = dispatcher.MODELS[MODEL]
    clf.fit(df_train,ytrain)
    preds = clf.predict_proba(df_test)[:,1]
    print(metrics.roc_auc_score(yval, preds))

    joblib.dump(label_encoders,f"../models/{MODEL}_label_encoder.pkl")
    joblib.dump(clf, f"../models/{MODEL}.pkl")



