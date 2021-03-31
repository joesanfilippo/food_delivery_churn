import os 
import boto3
import pandas as pd
from clean_data import Query_results
from sklearn.model_selection import train_test_split

def clean_store_data(churn_data, clean_dict, bucket_name, filename):
    """ Clean and store data in AWS bucket for later analysis
    Args:
        churn_data (Query_results): A Query_results object ready to be cleaned
        clean_dict (dict): A dictionary containing keyword arguments to clean and store query results including:
                            1. datetime_cols: Any columns containing datetime information
                            2. target_column: Column to calculate churn based off of
                            3. days_to_churn: The number of days to use before a user is considered "churned"
                            4. city_column: Column which contains city names
                            5. fake_cities: A list of fake city names to replace city_column with. Should be equal
                                            or greater to the number of unique cities in city_column.
        bucket_name (str): The name of the AWS S3 Bucket to store the training and test data in.
    
    Returns: None
             Stores the cleaned query results into the AWS bucket
    """
    churn_data.clean_data(clean_dict)

    churn_data_Xy = pd.concat([churn_data.df, churn_data.target], axis=1)
    churn_train, churn_test = train_test_split(churn_data_Xy, test_size=0.2, shuffle=True, stratify=churn_data.target)

    train_filename = f"{filename}_train.csv"
    test_filename = f"{filename}_test.csv"

    churn_train.to_csv(train_filename, index=False)
    churn_test.to_csv(test_filename, index=False)

    aws_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret = os.environ['AWS_SECRET_ACCESS_KEY']
    client = boto3.client('s3'
                         ,aws_access_key_id=aws_id
                         ,aws_secret_access_key=aws_secret)
                         
    client.upload_file(Filename=train_filename, Bucket=bucket_name, Key=train_filename)
    client.upload_file(Filename=test_filename, Bucket=bucket_name, Key=test_filename)

if __name__ == '__main__':
    
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

    params = {}
    api_key = os.environ['REDASH_API_KEY']
    query_url = os.environ['REDASH_LINK']

    original_churn_query_id = 714507
    original_churn_data = Query_results(query_url, original_churn_query_id, api_key, params)

    boolean_churn_query_id = 730923
    boolean_churn_data = Query_results(query_url, boolean_churn_query_id, api_key, params)


    clean_dict = {'datetime_cols': ['signup_time_utc', 'last_order_time_utc']
                ,'target_column': 'last_order_time_utc'
                ,'days_to_churn': 30
                ,'city_column': 'city_name'
                ,'fake_cities': got_cities
                }

    print(f"Cleaning and storing data...")
    clean_store_data(original_churn_data, clean_dict, 'food-delivery-churn', 'original_churn')
    clean_store_data(boolean_churn_data, clean_dict, 'food-delivery-churn', 'boolean_churn')
