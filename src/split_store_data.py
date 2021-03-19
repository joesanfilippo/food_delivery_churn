import os 
import boto3
import pandas as pd
from clean_data import Query_results
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    ## Pull data and clean it for analysis
    params = {'p_param': ''}
    churn_query_id = 714507
    api_key = os.environ['REDASH_API_KEY']
    query_url = os.environ['REDASH_LINK']
    
    churn_data = Query_results(query_url, churn_query_id, api_key, params)
    
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
                 ,'target_column': 'last_order_time_utc'
                 ,'days_to_churn': 30
                 ,'city_column': 'city_name'
                 ,'fake_cities': got_cities
                 }
    
    churn_data.clean_data(clean_dict)

    churn_data_Xy = pd.concat([churn_data.df, churn_data.target], axis=1)
    churn_train, churn_test = train_test_split(churn_data_Xy, test_size=0.2, shuffle=True, stratify=churn_data.target)

    churn_train.to_csv('churn_train.csv', index=False)
    churn_test.to_csv('churn_test.csv', index=False)

    aws_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret = os.environ['AWS_SECRET_ACCESS_KEY']
    client = boto3.client('s3'
                         ,aws_access_key_id=aws_id
                         ,aws_secret_access_key=aws_secret)
    client.upload_file(Filename='churn_train.csv', Bucket='food-delivery-churn', Key='churn_train.csv')
    client.upload_file(Filename='churn_test.csv', Bucket='food-delivery-churn', Key='churn_test.csv')