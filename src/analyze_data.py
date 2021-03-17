import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
font = {'weight': 'bold'
       ,'size': 16}
plt.rc('font', **font)
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from clean_data import Query_results

def kde_continuous_plot(values, target, ax):
    """ Plots a Kernel Density Estimate of Churned vs Active users using the values passed.
        Args:
            values (Pandas Series): The predictor that you want to use for the X axis on the KDE plot
            target (Pandas Series): The target that you want to use to group the KDE plot
            ax (matplotlib axis): An axis to plot the KDE plot

        Returns:
            None
            Modifies ax (matplotlib axis): An axis with the KDE plot
    """
    churn = values[target == True]
    active = values[target == False]

    sns.kdeplot(churn, fill=True, bw_method=0.1, color='#F8766D', label='Churned Users', ax=ax)
    sns.kdeplot(active, fill=True, bw_method=0.1, color='#619CFF', label='Active Users', ax=ax)
    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1e}'))
    ax.set_title(values.name.replace('_', ' ').title())
    ax.legend(loc='upper center')

def bar_categorical_plot(cat_df, cat_col, ax):
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
    ax.set_title(col.replace('_', ' ').title())
    ax.set_ylabel('% of Total Users')
    ax2.set_ylabel('# of Total Users')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')


if __name__ == '__main__':

    ## Pull data and clean it for analysis
    params = {'p_param': ''}
    churn_query_id = 714507
    api_key = os.environ['REDASH_API_KEY']
    query_url = os.environ['REDASH_LINK']
    got_cities = ["King's Landing"
                 ,"Braavos"
                 ,"Qarth"
                 ,"Old Valyria"
                 ,"Volantis"
                 ,"Asshai"
                 ,"Meereen"
                 ,"Astapor"
                 ,"Old Ghis"
                 ,"Oldtown"
                 ,"Pentos"
                 ,"Qohor"
                 ,"Sathar"
                 ,"Sunspear"
                 ,"Lannisport"
                 ,"Vaes Dothrak"
                 ,"White Harbor"
                 ,"Yunkai"
                 ,"The Wall"
                 ,"Ghozi"
                 ,"Gulltown"
                 ,"Lys"
                 ,"Mantarys"
                 ,"Tyria"
                 ,"Tolos"
                 ,"Samyrian"
                 ,"Oros"
                 ,"Norvos"]
    clean_dict = {'datetime_cols': ['signup_time_utc', 'last_order_time_utc']
                 ,'boolean_cols': ['first_order_delivered_on_time']
                 ,'target_column': 'last_order_time_utc'
                 ,'days_to_churn': 30
                 ,'city_column': 'city_name'
                 ,'fake_cities': got_cities
                 }
    
    churn_data = Query_results(query_url, churn_query_id, api_key, params)
    churn_data.clean_data(clean_dict)
    
    ## Visualize continuous predictors using KDE plots
    continuous_data = churn_data.df.select_dtypes(include=[np.number, 'boolean']).drop('user_id', axis=1)
    continuous_columns = continuous_data.columns.tolist()

    fig, axs = plt.subplots(nrows=np.ceil(len(continuous_columns)/2).astype(int), ncols=2, figsize=(20,50))
    
    for col, ax in zip(continuous_columns, axs.flatten()):
        kde_continuous_plot(continuous_data[col], churn_data.target, ax)
    
    plt.tight_layout(rect=(0,0,1,0.98))
    plt.suptitle("KDE Plots for Continuous Predictors", y=0.99, fontsize=35)
    plt.savefig('images/kde_plots.png')

    ## Visualize categorical predictors using stacked 100% fill barcharts
    categorical_data = pd.concat([churn_data.df.select_dtypes(include=['object']), churn_data.target],axis=1)
    categorical_columns = categorical_data.columns.tolist()[:-1]
    # print(categorical_data)
    fig, axs = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(20,35))

    for col, ax in zip(categorical_columns, axs.flatten()):
        bar_categorical_plot(categorical_data, col, ax)

    plt.tight_layout(rect=(0,0,1,0.98))
    plt.suptitle("100% Fill Barchart for Categorical Predictors", y=0.99, fontsize=35)
    plt.savefig('images/barchart_plots.png')
    
