from sklearn import ensemble

MODELS = {
    'randomforest': ensemble.RandomForestClassifier(n_estimators=20,n_jobs=-1, verbose=2),
    'extratree': ensemble.ExtraTreesClassifier(n_estimators=20,n_jobs=-1, verbose=2)
}