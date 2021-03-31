from train_test import load_X_y, Churn_Model
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
font = {'weight': 'bold'
       ,'size': 16}
plt.rc('font', **font)
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score
from collections import OrderedDict


if __name__ == '__main__':

    bucket_name = 'food-delivery-churn'
    filename = 'boolean_churn'
    is_feature_selection = False
    feature_list = []

    X_train, X_test, y_train, y_test = load_X_y(bucket_name, filename, is_feature_selection, feature_list)

    churn_model = Churn_Model(GradientBoostingClassifier(), (X_train, X_test, y_train, y_test))
    churn_model.convert_cat_to_int()
    X_train, X_test, y_train, y_test = churn_model.X_train, churn_model.X_test, churn_model.y_train, churn_model.y_test
    
    best_lr = LogisticRegression(class_weight='balanced'
                                ,fit_intercept=True
                                ,max_iter=100
                                ,penalty='l2'
                                ,solver='liblinear')
    
    best_lr.fit(X_train, y_train)
    X_train['lr_predictions'] = best_lr.predict_proba(X_train)[:,1]
    X_test['lr_predictions'] = best_lr.predict_proba(X_test)[:,1]

    best_model = GradientBoostingClassifier(subsample=0.25
                                           ,n_estimators=200
                                           ,min_samples_leaf=2
                                           ,max_features=None
                                           ,max_depth=8
                                           ,learning_rate=0.05)

    best_model.fit(X_train, y_train)

    print(best_model.score(X_test, y_test))

    
    # feature_dict = {}
    # for feature, importance in zip(X_test.columns, best_model.feature_importances_):
    #     feature_dict[feature] = importance

    # print("| Feature                        | Importance % |  ")
    # print("|--------------------------------|--------------| ")
    # for feature in sorted(feature_dict, key=feature_dict.__getitem__, reverse=True):
    #     print(f"| {feature.replace('_', ' ').title()} | {feature_dict[feature]:.1%} |")

    # my_plot = plot_partial_dependence(best_model
    #                                 ,features=[i for i in range(0,len(X_train.columns.tolist()))]
    #                                 ,X=X_train
    #                                 ,n_cols=4
    #                                 ,feature_names=X_train.columns.tolist())
    # plt.show()