from train_test import load_X_y, Churn_Model
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
font = {'weight': 'bold'
       ,'size': 16}
plt.rc('font', **font)
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score
from collections import OrderedDict

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = load_X_y('food-delivery-churn', 30)

    churn_model = Churn_Model(GradientBoostingClassifier(), 30, (X_train, X_test, y_train, y_test))
    churn_model.convert_cat_to_int()
    X_train, X_test, y_train, y_test = churn_model.X_train, churn_model.X_test, churn_model.y_train, churn_model.y_test
    
    best_model = GradientBoostingClassifier(subsample=0.25
                                           ,n_estimators=200
                                           ,min_samples_leaf=2
                                           ,max_features=None
                                           ,max_depth=8
                                           ,learning_rate=0.05)

    best_model.fit(X_train, y_train)
    
    feature_dict = {}
    for feature, importance in zip(X_test.columns, best_model.feature_importances_):
        feature_dict[feature] = importance

    print("| Feature                        | Importance % |  ")
    print("|--------------------------------|--------------| ")
    for feature in sorted(feature_dict, key=feature_dict.__getitem__, reverse=True):
        print(f"| {feature.replace('_', ' ').title()} | {feature_dict[feature]:.1%} |")