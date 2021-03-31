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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score

class Churn_Model(object):

    def __init__(self, classifier, split_data):
        """ Initialize an instance of the Churn_Model class given a classifier and number of days to use to 
            classify a user as "churned"
        Args:
            classifier (sklearn model): The type of classifier to use in the model. Examples include Logisitic 
                                        Regression, RandomForestClassifier, and GradientBoostingClassifier.
            split_data (tuple): Training and Test data split into their predictors (X) and targets (y)

        Returns:
            None
            Instantiates a Churn_Model class 
        """
        self.classifier = classifier 
        self.X_train, self.X_test, self.y_train, self.y_test = split_data
        
    def convert_cat_to_int(self):
        """ Converts string objects in categorical data to integers to use for training models.
        Args:
            None

        Returns:
            None
            Converts categorical object columns in self.X_train to categorical integer columns.
        """
        full_data = pd.concat([self.X_train, self.X_test], axis=0)
        object_cols = full_data.select_dtypes(include=['object']).columns.tolist()

        for col in object_cols:
            col_dict = {}

            for idx, category in enumerate(full_data[col].unique()):
                col_dict[category] = idx

            self.X_train[col] = self.X_train[col].map(lambda x: col_dict[x])
            self.X_test[col] = self.X_test[col].map(lambda x: col_dict[x])

    def fit_model(self, grid, selection_type, scoring_type):
        """ Using GridSearchCV or RandomSearchCV, find the optimal hyperparameters of the classifier and 
            then fit it to the training data.
        Args: 
            grid (dict): A dictionary of hyperparameters and their associated values to test the classifier with.
            selection_type (sklearn selection_model): The type of hyperparameter selection to use, either 
                                                      GridSearchCV or RandomizedSearchCV.
            scoring_type (str): The metric used to score the fit, examples include 'accuracy', 'precision', 'recall',
                                or 'roc_auc'.
        Returns:
            None
            Performs a RandomSearchCV or GridsearchCV on a classifier and stores the best_estimator_ to the class.
        """
        self.model_search = selection_type(self.classifier
                                     ,grid
                                     ,n_jobs=-1
                                     ,verbose=False
                                     ,scoring=scoring_type)

        self.model_search.fit(self.X_train, self.y_train)

        classifier_name = self.classifier.__class__.__name__

        print(f"Best Parameters for {classifier_name}: {self.model_search.best_params_}")
        print(f"Best {scoring_type} Score for {classifier_name}: {self.model_search.best_score_:.4f}")

        self.best_model = self.model_search.best_estimator_

class Churn_Plot(object):

    def __init__(self, classifier, X_test, y_test):
        """ Initialize an instance of the Churn_Plot class given a classifier and number of days to use to 
            classify a user as "churned"
        Args:
            classifier (sklearn model): The type of classifier to use in the model. Examples include Logisitic 
                                        Regression, RandomForestClassifier, and GradientBoostingClassifier.
            X_test (Pandas DataFrame): The predictor values to use in performance evaluation
            y_test (Pandas Series): The target values to use to measure performance

        Returns:
            None
            Instantiates a Churn_Model class 
        """
        self.classifier = classifier
        self.X_test = X_test
        self.y_test = y_test
        self.y_probs = self.classifier.predict_proba(self.X_test)[:,1] 

    def plot_model_roc(self, ax, plot_kwargs={}):
        """ Plots the ROC Curve for a classifier given the test data and axis
        Args:
            classifier (Sklearn Classifier): The best classifier to test against other classifiers
            ax (matplotlib axis): An axis to plot the ROC curve on.
            plot_kwargs (dict): Keyword arguments to pass to the plot for formatting

        Returns: 
            None
            Modifies ax by plotting the ROC curve from the best classifier.
        """
        auc_score = roc_auc_score(self.y_test, self.y_probs)
        fpr, tpr, threshold = roc_curve(self.y_test, self.y_probs)
        roc_df = pd.DataFrame(zip(fpr, tpr, threshold), columns = ['fpr', 'tpr', 'threshold'])
        ax.plot(roc_df.fpr, roc_df.tpr, label=f"{self.classifier.__class__.__name__} AUC={auc_score:.3f}", **plot_kwargs)
    
    def calc_profit_curve(self, tp_cost, fp_cost):
        """ Calculates the profit for a range of thresholds given the true and false positive costs.
        Args:
            tp_cost (float): The cost of correctly predicted a positive class
            fp_cost (float): The cost a incorrectly predicting a positive class

        Returns:
            threshold_list (list): List from 0% to 100% of each threshold
            profit_list (list): List of profits calculated at the corresponding threshold
        """
        thresholds = np.linspace(0,1,101)
        profit_list = []
        threshold_list = []

        for threshold in thresholds:
            y_preds = (self.y_probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_preds).ravel()
            profit = tp * tp_cost + fp * fp_cost
            profit_list.append(round(profit, 2))
            threshold_list.append(round(threshold, 2))
        
        return threshold_list, profit_list

    def plot_profit_curve(self, ax, plot_kwargs={}, tp_cost=8.42, fp_cost=-5.43):
        """ Plots the Profit Curve for a classifier given an ax and plot_kwargs
        Args:
            ax (matplotlib axis): The axis to plot the best classifier's F1 score curve.
            plot_kwargs (dict): Keyword arguments to visually separate the best classifiers.
            tp_cost (float): The cost of correctly predicted a positive class
            fp_cost (float): The cost a incorrectly predicting a positive class

        Returns:
            None
            Modifies the ax to show the Profit curve for best classifier
        """
        x_axis, y_axis = self.calc_profit_curve(tp_cost, fp_cost)

        max_profit = max(y_axis) / len(self.X_test)
        max_profit_line = x_axis[y_axis.index(max(y_axis))]
        classifier_name = self.classifier.__class__.__name__

        ax.plot(x_axis, y_axis, label=f"{classifier_name} Profit per User: ${max_profit:.2f}", **plot_kwargs)
        ax.axvline(x=max_profit_line, label=f"Max Profit Threshold: {max_profit_line:.0%}", **plot_kwargs)

    def plot_f1_curve(self, ax, plot_kwargs={}):
        """ Plots the F1 Score Curve for a classifier given an ax and plot_kwargs
        Args:
            ax (matplotlib axis): The axis to plot the best classifier's F1 score curve.
            plot_kwargs (dict): Keyword arguments to visually separate the best classifiers.
            
        Returns:
            None
            Modifies the ax to show the F1 Score curve for best classifier
        """
        thresholds = np.linspace(0,1,101)
        threshold_list = []
        f1_list = []

        for threshold in thresholds:
            y_preds = (self.y_probs >= threshold).astype(int)
            f1 = f1_score(self.y_test, y_preds)
            f1_list.append(f1)
            threshold_list.append(threshold)

        max_f1_score = max(f1_list)
        max_threshold = threshold_list[f1_list.index(max(f1_list))]
        classifier_name = self.classifier.__class__.__name__

        ax.plot(threshold_list, f1_list, label=f"{classifier_name} F1 Score: {max_f1_score:.1%}", **plot_kwargs)
        ax.axvline(x=max_threshold
                  ,label=f'Ideal Threshold: {max_threshold:.0%}', **plot_kwargs)


def load_X_y(bucket_name, filename, is_feature_selection=False, feature_list=[]):
    """ Loads the X & y training & test data from AWS Bucket
    Args: 
        bucket_name (str): The AWS S3 bucket to pull the training and test data from.
        is_feature_selection (bool): Whether or not to use specific features to train the model. Default is False.
        feature_list (list): The names of the features to use if `is_feature_selection` is True. Default is an 
                             empty list.

    Returns:
        X_train (Pandas DataFrame): Predictor values for training dataset
        X_test (Pandas DataFrame): Predictor values for test dataset
        y_train (Pandas Series): Target values for training dataset
        y_test (Pandas Series): Target values for test dataset
    """
    aws_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret = os.environ['AWS_SECRET_ACCESS_KEY']
    client = boto3.client('s3'
                          ,aws_access_key_id=aws_id
                          ,aws_secret_access_key=aws_secret)

    train_obj = client.get_object(Bucket=bucket_name, Key=f"{filename}_train.csv")
    test_obj = client.get_object(Bucket=bucket_name, Key=f"{filename}_test.csv")

    X_train = pd.read_csv(io.BytesIO(train_obj['Body'].read()), encoding='utf8')
    X_train = X_train.drop(['user_id', 'signup_time_utc', 'last_order_time_utc'], axis=1)
    if is_feature_selection:
        X_train = X_train[feature_list]
    y_train = X_train.pop('churned_user')

    X_test = pd.read_csv(io.BytesIO(test_obj['Body'].read()), encoding='utf8')
    X_test = X_test.drop(['user_id', 'signup_time_utc', 'last_order_time_utc'], axis=1)
    if is_feature_selection:
        X_test = X_test[feature_list]
    y_test = X_test.pop('churned_user')

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    
    bucket_name = 'food-delivery-churn'
    filename = 'original_churn'
    is_feature_selection = False
    feature_list = []
    ## Only use first order columns
    # column_tuple = ("first_30", "city", "signup_to")
    # first_order_columns = rf_model.X_train.columns.to_series().str.startswith(column_tuple)]
    
    ## Only use select columns 
    # feature_list = ['days_since_signup'
    #                ,'first_30_day_orders'
    #                ,'signup_to_order_hours'
        
    split_data = load_X_y(bucket_name, filename, is_feature_selection, feature_list)

    lr_model = Churn_Model(LogisticRegression(), split_data)
    rf_model = Churn_Model(RandomForestClassifier(), split_data)
    gb_model = Churn_Model(GradientBoostingClassifier(), split_data)

    lr_model.convert_cat_to_int()
    rf_model.convert_cat_to_int()
    gb_model.convert_cat_to_int()
    
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
    
    lr_model.fit_model(logistic_regression_grid, GridSearchCV, 'roc_auc')
    rf_model.fit_model(random_forest_grid, RandomizedSearchCV, 'roc_auc')
    gb_model.fit_model(gradient_boosting_grid, RandomizedSearchCV, 'roc_auc')

    X_test = split_data[1]
    y_test = split_data[3]

    lr_plot_kwargs = {'linestyle':'-', 'linewidth': 3, 'color': '#F8766D'}
    rf_plot_kwargs = {'linestyle':'--', 'linewidth': 3, 'color': '#00BA38'}
    gb_plot_kwargs = {'linestyle':':', 'linewidth': 3, 'color': '#619CFF'}
    
    classifiers = [lr_model.best_model, rf_model.best_model, gb_model.best_model]
    classifier_plot_kwargs = [lr_plot_kwargs, rf_plot_kwargs, gb_plot_kwargs]

    fig, ax = plt.subplots(figsize=(15,15))
    
    for classifier, kwargs in zip(classifiers, classifier_plot_kwargs):
        model_plot = Churn_Plot(classifier, X_test, y_test)
        model_plot.plot_model_roc(ax, kwargs)             

    ax.legend(loc='lower right')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("False Positivity Rate")
    ax.set_ylabel("True Positivity Rate")
    ax.set_title(f"ROC Curve for Best Classifiers")                                                                    
    
    plt.savefig(f"images/original_roc_curves.png")

    fig, axs = plt.subplots(nrows=2, figsize=(15,25))
    
    for classifier, kwargs in zip(classifiers, classifier_plot_kwargs):
        
        model_plot = Churn_Plot(classifier, X_test, y_test)
        model_plot.plot_f1_curve(axs[0], kwargs)             
        model_plot.plot_profit_curve(axs[1], kwargs, tp_cost=2.99)
    
    axs[0].legend(loc='lower left')
    axs[0].set_title(f"F1 Curves for Best Classifiers")                                                                    
    axs[0].set_xlabel(f"Threshold Percent")
    axs[0].set_ylabel(f"F1 Score")
    axs[0].xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axs[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    axs[1].legend(loc='lower left')
    axs[1].set_title("Profit Curves for Best Classifiers")                                                                    
    axs[1].set_xlabel(f"Threshold Percent")
    axs[1].set_ylabel(f"Profit ($) on {round(len(y_test)/1000,0)}k Users")
    axs[1].xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    
    plt.suptitle("Comparing Best Classifiers", y=0.99, fontsize=30)
    plt.tight_layout()
    plt.savefig(f"images/original_profit_and_f1_curves.png")