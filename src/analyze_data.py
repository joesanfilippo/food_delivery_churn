import io
import os
import boto3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
font = {'weight': 'bold'
       ,'size': 16}
plt.rc('font', **font)
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

class EDA_Plot(object):

    def __init__(self, bucket_name, train_filename):
        """ Initialize an instance of the EDA_Plot class that will be used to plot categorical and continuous 
           predictors to use in the machine learning models.
        
        Args: 
            bucket_name (str): The name of the AWS S3 bucket to pull the training data from.
            train_filename (str): The name of the training data csv to pull from the AWS S3 Bucket.

        Returns:
            None
            Instantiates a EDA_Plot class
        """
        aws_id = os.environ['AWS_ACCESS_KEY_ID']
        aws_secret = os.environ['AWS_SECRET_ACCESS_KEY']
        client = boto3.client('s3'
                            ,aws_access_key_id=aws_id
                            ,aws_secret_access_key=aws_secret)

        train_obj = client.get_object(Bucket=bucket_name, Key=train_filename)

        self.X = pd.read_csv(io.BytesIO(train_obj['Body'].read())
                            ,encoding='utf8'
                            ,parse_dates=['signup_time_utc', 'last_order_time_utc']
                            ,date_parser=pd.to_datetime)

        self.y = self.X.pop('churned_user')

    def kde_continuous_plot(self, continuous_values, ax):
        """ Plots a Kernel Density Estimate of Churned vs Active users using the values passed.
            Args:
                continuous_values (Pandas Series): The predictor that you want to use for the X axis on the KDE plot
                ax (matplotlib axis): An axis to plot the KDE plot

            Returns:
                None
                Modifies ax (matplotlib axis): An axis with the KDE plot
        """
        churn = continuous_values[self.y == True]
        active = continuous_values[self.y == False]

        sns.kdeplot(churn, fill=True, bw_method=0.1, color='#F8766D', label='Churned Users', ax=ax)
        sns.kdeplot(active, fill=True, bw_method=0.1, color='#619CFF', label='Active Users', ax=ax)
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1e}'))
        ax.set_title(continuous_values.name.replace('_', ' ').title())
        ax.legend(loc='upper center')

    def plot_continuous_features(self):
        """ Visualize continuous predictors using KDE plots
            Args: 
                None

            Returns:
                None
                Saves the KDE plots from the continuous features into a .png file for viewing
        """
        continuous_data = self.X.select_dtypes(include=[np.number, 'boolean']).drop('user_id', axis=1)
        continuous_columns = continuous_data.columns.tolist()
        
        fig, axs = plt.subplots(nrows=np.ceil(len(continuous_columns)/2).astype(int), ncols=2, figsize=(20,50))
        
        for col, ax in zip(continuous_columns, axs.flatten()):
            self.kde_continuous_plot(continuous_data[col], ax)
        
        if len(continuous_columns) % 2 == 1:
            fig.delaxes(axs[np.ceil(len(continuous_columns)/2).astype(int)-1, 1])

        plt.tight_layout(rect=(0,0,1,0.98))
        plt.suptitle(f"KDE Plots for Continuous Predictors", y=0.99, fontsize=35)
        plt.savefig(f"images/original_kde_plots.png")

    def bar_categorical_plot(self, cat_df, cat_col, ax):
        """ Plots a 100% Fill Stacked Barchart of Churned vs Active users using the values passed.
            Args:
                cat_df (Pandas Dataframe): The predictors and target that you want to use for the 
                                        X axis on the barchart
                cat_col (str): The column name of the categorical data to plot
                ax (matplotlib axis): An axis to plot the barchat

            Returns:
                None
                Modifies ax (matplotlib axis): An axis with the barchart plot
        """
        group_data = cat_df.groupby([cat_col, 'churned_user']).size().unstack()
        group_data.columns = ['Active Users', 'Churned Users']
        group_data = group_data.fillna(value=0)
        group_data['Total Users'] = group_data['Active Users'] + group_data['Churned Users']
        group_data['Active Percent'] = group_data['Active Users'] / group_data['Total Users']
        group_data['Churned Percent'] = group_data['Churned Users'] / group_data['Total Users']
        total_users = group_data.sort_values('Active Percent', ascending=False)['Total Users']
        city_churn = group_data[['Active Percent', 'Churned Percent']].sort_values('Active Percent', ascending=False)
        city_churn.plot.bar(stacked=True, ax=ax, color=['#619CFF', '#F8766D'], alpha = 0.5)
        ax2 = ax.twinx()
        total_users.plot(color='black', ax=ax2)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        ax.set_xlabel('')
        ax.set_title(cat_col.replace('_', ' ').title())
        ax.set_ylabel('% of Total Users')
        ax2.set_ylabel('# of Total Users')
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    def plot_categorical_features(self):
        """ Visualize categorical predictors using stacked 100% fill barcharts
            Args: 
                None 
            
            Returns:
                None:
                Saves the barchart plots from the categorical features into a .png file for viewing
        """
        categorical_data = pd.concat([self.X.select_dtypes(include=['object']), self.y],axis=1)
        categorical_columns = categorical_data.columns.tolist()[:-1]
        
        fig, axs = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(20,35))

        for col, ax in zip(categorical_columns, axs.flatten()):
            self.bar_categorical_plot(categorical_data, col, ax)

        plt.tight_layout(rect=(0,0,1,0.98))
        plt.suptitle(f"100% Fill Barchart for Categorical Predictors", y=0.99, fontsize=35)
        plt.savefig(f"images/original_barchart_plots.png")

if __name__ == '__main__':
    
    bucket_name = 'food-delivery-churn'
    train_filename = 'original_churn_train.csv'
    
    churn_eda = EDA_Plot(bucket_name, train_filename)
    
    print(f"Plotting Continuous Features...")
    churn_eda.plot_continuous_features()
    
    print(f"Plotting Categorical Features...")
    churn_eda.plot_categorical_features()