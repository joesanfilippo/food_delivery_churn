import io
import os
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
font = {'weight': 'bold'
       ,'size': 16}
plt.rc('font', **font)
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

def convert_cat_to_int(data, col):
    """ Converts string objects in categorical data to integers to use for training models.
    Args:
        data (Pandas DF): A pandas DF with a column 'col' to convert string/boolean values to integers
        col (str) = The name of the column to convert from categorical to integer
        
    Returns:
        data (Pandas DF): A pandas DF with the string values in 'col' converted to integers
    """
    col_dict = {}
    for idx, cat in enumerate(data[col].unique()):
        col_dict[cat] = idx

    data[col] = data[col].map(lambda x: col_dict[x])

    return data

def plot_model_aoc(classifier, X_test, y_test, ax, plot_kwargs={}):
    """ Plots the AOC Curve for a classifier given the test data and axis
    Args:
        classifier (Sklearn Classifier): The best classifier to test against other classifiers
        X_test (numpy array): Values to use when predicting with the classifier
        y_test (numpy array): True target values to evaluate the predictors against
        ax (matplotlib axis): An axis to plot the AOC curve on.

    Returns: 
        None
        Modifies ax by plotting the AOC curve from the best classifier.
    """
    y_preds = classifier.predict_proba(X_test)[:,1] 
    auc_score = roc_auc_score(y_test, y_preds)
    fpr, tpr, threshold = roc_curve(y_test, y_preds)
    roc_df = pd.DataFrame(zip(fpr, tpr, threshold), columns = ['fpr', 'tpr', 'threshold'])
    ax.plot(roc_df.fpr, roc_df.tpr, label=f"{classifier.__class__.__name__} AUC={auc_score:.3f}", **plot_kwargs)

if __name__ == '__main__':
    
    ## Load training data from AWS
    aws_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret = os.environ['AWS_SECRET_ACCESS_KEY']
    client = boto3.client('s3'
                         ,aws_access_key_id=aws_id
                         ,aws_secret_access_key=aws_secret)

    train_obj = client.get_object(Bucket='food-delivery-churn', Key='churn_train.csv')
    churn_train_X = pd.read_csv(io.BytesIO(train_obj['Body'].read()), encoding='utf8')

    ## Remove target column from X and assign to y
    churn_train_y = churn_train_X.pop('churned_user')
    
    ## Dropping columns that aren't going to be relevant predictors
    churn_train_X_drop = churn_train_X.drop(['user_id', 'signup_time_utc', 'last_order_time_utc'], axis=1)

    ## Convert string columns to integers.
    churn_train_X_drop = convert_cat_to_int(churn_train_X_drop, 'city_name')
    churn_train_X_drop = convert_cat_to_int(churn_train_X_drop, 'city_group')
    
    ## Only use first order columns
    # column_tuple = ("first_30", "city", "signup_to")
    # churn_train_X_drop = churn_train_X_drop.loc[:, churn_train_X_drop.columns.to_series().str.startswith(column_tuple)]
    
    ## Only use select columns
    # select_columns = ['city_name'
    #                  ,'signup_to_order_hours'
    #                  ,'first_order_discount_percent'
    #                  ,'first_30_day_orders'
    #                  ,'first_30_day_avg_meal_rating'
    #                  ,'first_30_day_avg_driver_rating'
    #                  ,'first_30_day_discount_percent'
    #                  ,'first_30_day_subscription_user']
    # churn_train_X_drop = churn_train_X_drop[select_columns]
    
    X_train = churn_train_X_drop.values
    y_train = churn_train_y.values

    ## Random Search to find best hyperparams of each model
    logistic_regression_grid = {'penalty': ['l1', 'l2']
                               ,'fit_intercept': [True, False]
                               ,'class_weight': [None, 'balanced']
                               ,'solver': ['liblinear']
                               }

    random_forest_grid = {'max_depth': [2, 4, 8]
                         ,'max_features': ['sqrt', 'log2', None]
                         ,'min_samples_leaf': [1, 2, 4]
                         ,'min_samples_split': [2, 4]
                         ,'bootstrap': [True, False]
                         ,'n_estimators': [5,10,25,50,100,200]}

    gradient_boosting_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.25]
                             ,'max_depth': [2, 4, 8]
                             ,'subsample': [0.25, 0.5, 0.75, 1.0]
                             ,'min_samples_leaf': [1, 2, 4]
                             ,'max_features': ['sqrt', 'log2', None]
                             ,'n_estimators': [5,10,25,50,100,200,250]}

    logistic_gridsearch = GridSearchCV(LogisticRegression()
                                        ,logistic_regression_grid
                                        ,n_jobs=-1
                                        ,verbose=False
                                        ,scoring='accuracy')

    random_forest_randomsearch = RandomizedSearchCV(RandomForestClassifier(warm_start=True)
                                                   ,random_forest_grid
                                                   ,n_jobs=-1
                                                   ,verbose=False
                                                   ,scoring='accuracy')

    gradient_boosting_randomsearch = RandomizedSearchCV(GradientBoostingClassifier(warm_start=True)
                                                   ,gradient_boosting_grid
                                                   ,n_jobs=-1
                                                   ,verbose=False
                                                   ,scoring='accuracy')

    logistic_gridsearch.fit(X_train, y_train)
    random_forest_randomsearch.fit(X_train, y_train)
    gradient_boosting_randomsearch.fit(X_train, y_train)
    
    # print(f"Best Logistic Parameters: {logistic_gridsearch.best_params_}")
    # print(f"Best Random Forest Parameters: {random_forest_randomsearch.best_params_}")
    # print(f"Best Gradient Boosting Parameters: {gradient_boosting_randomsearch.best_params_}")

    # print(f"Best Logistic Score: {logistic_gridsearch.best_score_:.4f}")
    # print(f"Best Random Forest Score: {random_forest_randomsearch.best_score_:.4f}")
    # print(f"Best Gradient Boosting Score: {gradient_boosting_randomsearch.best_score_:.4f}")

    logistic_best_model = logistic_gridsearch.best_estimator_
    random_forest_best_model = random_forest_randomsearch.best_estimator_
    gradient_boosting_best_model = gradient_boosting_randomsearch.best_estimator_ 

    ## Load and clean test data to match training data
    test_obj = client.get_object(Bucket='food-delivery-churn', Key='churn_test.csv')
    churn_test_X = pd.read_csv(io.BytesIO(test_obj['Body'].read()), encoding='utf8')
    
    ## Remove target column from X and assign to y
    churn_test_y = churn_test_X.pop('churned_user')

    ## Dropping columns that aren't going to be relevant predictors
    churn_test_X_drop = churn_test_X.drop(['user_id', 'signup_time_utc', 'last_order_time_utc'], axis=1)

    ## Convert string columns to integers.
    churn_test_X_drop = convert_cat_to_int(churn_test_X_drop, 'city_name')
    churn_test_X_drop = convert_cat_to_int(churn_test_X_drop, 'city_group')

    ## Only use first order columns
    # churn_test_X_drop = churn_test_X_drop.loc[:, churn_test_X_drop.columns.to_series().str.startswith(column_tuple)]
    
    ## Only use select columns
    # churn_test_X_drop = churn_test_X_drop[select_columns]
    
    X_test = churn_test_X_drop.values
    y_test = churn_test_y.values

    fig, ax = plt.subplots(figsize=(25,10))

    plot_model_aoc(logistic_best_model, X_test, y_test, ax, plot_kwargs={'linestyle':'-'
                                                                        ,'linewidth': 3
                                                                        ,'color': '#F8766D'})
    plot_model_aoc(random_forest_best_model, X_test, y_test, ax, plot_kwargs={'linestyle':'--'
                                                                        ,'linewidth': 3
                                                                        ,'color': '#00BA38'})
    plot_model_aoc(gradient_boosting_best_model, X_test, y_test, ax, plot_kwargs={'linestyle':':'
                                                                        ,'linewidth': 3
                                                                        ,'color': '#619CFF'})
    ax.legend(loc='lower right')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("False Positivity Rate")
    ax.set_ylabel("True Positivity Rate")
    ax.set_title("AOC Curve for Best Classifiers")                                                                    

    plt.show()