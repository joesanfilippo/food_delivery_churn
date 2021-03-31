import pandas as pd 
import query_pull

class Query_results(object):

    def __init__(self, query_url, query_id, api_key, params):
        """ Initialize an instance of the Query_results class that will be used to analyze user churn.

        Args:
            query_url (str): The URL of the query software to use. This is stored in the user's .bash_profile
                       or .zshrc file to avoid any identifying information of the company.
            query_id (int): The unique identifier of the query to use. This is needed since the query is 
                      prewritten to avoid any identifying table or column names.
            api_key (str): API key stored in the user's .bash_profile or .zshrc file.
            params (dict): Any optional parameters to be used in the query like number of churn days, dates, or cities.

        Returns:
            None
            Stores the results of the query into self.results
        """
        self.query_url = query_url 
        self.query_id = query_id 
        self.api_key = api_key 
        self.params = params 
        self.results = query_pull.get_fresh_query_result(self.query_url, self.query_id, self.api_key, self.params)

    def to_dataframe(self):
        """ Converts the results of a query to a Pandas Dataframe
        Args:
            self (Query_results): A class of Query_results with self.results populated

        Returns:
            None
            Creates self.df (Pandas DF): A Pandas Dataframe with the columns cleaned
        """
        self.df = pd.DataFrame(self.results)

    def clean_columns(self):
        """ Replaces column names that have ' ' with _ and changes all the column names to all lowercase letters.
        Args:
            self (Query_results): A class of Query_results with self.df populated

        Returns:
            None
            Modifies self.df (Pandas DF): A Pandas Dataframe with the columns cleaned
        """
        cleaned_cols = self.df.columns.tolist()
        cleaned_cols = [col.replace(' ', '_').lower() for col in cleaned_cols]
        self.df.columns = cleaned_cols

    def convert_booleans(self):
        """ Converts any boolean columns to Boolean data type for further analysis
        Args:
            self (Query_results): A class of Query_results with self.df populated
            boolean_cols (list): A list of boolean column names to convert to a Boolean data type

        Returns:
            None
            Modifies self.df (Pandas DF): A Pandas Dataframe with the boolean columns converted
        """
        boolean_cols = self.df.select_dtypes(include=['bool']).columns.tolist()

        for col in boolean_cols:
            self.df[col] = self.df[col].astype(bool)

    def convert_datetimes(self, datetime_cols):
        """ Converts any datetime columns to a Pandas DateTime64 data type for further analysis
        Args:
            self (Query_results): A class of Query_results with self.df populated
            datetime_cols (list): A list of datetime column names to convert to a Pandas DataTime64 object

        Returns:
            None
            Modifies self.df (Pandas DF): A Pandas Dataframe with the datetime columns converted
        """
        for col in datetime_cols:
            self.df[col] = pd.to_datetime(self.df[col])

    def calculate_churned_user(self, target_column, days_to_churn=30):
        """ Calculates whether or not a user is considered churn based on the days_to_churn
        Args:
            self (Query_results): A class of Query_results with self.df populated
            target_column (str): The target column to evaluate whether or not a user has churned.
                                  In this case, the target column is date of the user's last order.
            days_to_churn (int): The number of days since a user's last order after which a user is 
                                 considered to have churned. Default is 30 days.

        Returns:
            None
            Creates self.target (Pandas Series): A Pandas series with values that determine whether or
                                                 not a user has churned.
        """
        churn_date = pd.to_datetime('today').floor('D') - pd.Timedelta(days=days_to_churn)
        self.target = self.df[target_column].map(lambda x: True if x < churn_date else False)
        self.target = self.target.rename("churned_user")

    def convert_cities(self, city_column, fake_cities):
        """ Converts the real city names to fake city names to hide any identifying company information.
        Args:
            self (Query_results): A class of Query_results with self.df populated
            city_column (str): The column that includes city names to convert to fake names.
            fake_cities (lst): A list of fake city names to use for the real city names. Length of the list
                               should be equal to the length of unique real city names.

        Returns:
            None
            Modifies self.df (Pandas DF): A Pandas Dataframe with the real city names replaced by fake ones.
        """
        real_cities = self.df[city_column].sort_values().unique().tolist()
        fake_cities.sort()
        city_dict = dict(zip(real_cities, fake_cities))
        self.df.city_name = self.df.city_name.map(city_dict)

    def clean_data(self, kwargs_dict):
        """ Perform all cleaning methods on the Query_results class object.
        Args:
            kwargs_dict (dict): A dictionary containing keyword arguments for: 
                                    1. convert_datetimes: datetime_cols
                                    2. convert_booleans: boolean_cols
                                    3. calculate_churned_user: target_column & days_to_churn (optional)
                                    4. convert_cities: city_column & fake_cities
        Returns:
            None
            Modifies self.df (Pandas DF) and creates self.target for use in later analysis.
        """
        self.to_dataframe()
        self.clean_columns()
        self.convert_booleans()
        self.convert_datetimes(kwargs_dict['datetime_cols'])
        try:
            self.calculate_churned_user(kwargs_dict['target_column'], kwargs_dict['days_to_churn'])
        except:
            self.calculate_churned_user(kwargs_dict['target_column'])
        self.convert_cities(kwargs_dict['city_column'], kwargs_dict['fake_cities'])

if __name__ == '__main__':

    print("I'll show you.")