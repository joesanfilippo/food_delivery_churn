# Predicting User Churn

## Background
Food Delivery Company X saw substantial user growth as lockdowns started and restaurants shutdown in-person dining beginning mid-March in 2013. More and more people turned to app based delivery platforms to enjoy their favorite food from local restaurants. One year later, with vaccinations rolling out and more restaurants opening in person dining, their growth in active users has started to plateau and the CEO wants to dig in to what's causing it and what they can do to prevent it.

Using Supervised Machine Learning models, I will attempt to predict the probability a user will churn based of their signup data, first order data, and first 30 days on the platform. 

$$\Large \text{Churned User (+ Class)} = \text{A user who's last order was > 30 days ago}$$
$$\Large \text{Active User (- Class)} = \text{A user who's last order was <= 30 days ago}$$

---

## Dataset
My dataset is pulled from Company X's database using their preferred query editor. In order to protect their data and identity, I converted any identifying city names and stored the data in a private AWS S3 bucket to use for later analysis.

### Features
The features I pulled included:

| Feature Name                   | Data Type | Description |
|--------------------------------|-----------|-------------|
| city_name                      | String    | The city the user placed their first order in. Changed to protect the company's identity and data. |
| city_group                     | String    | One of `college town` or `city` based on the city's population size and presence of a university. |
| signup_time_utc                | DateTime  | The date and time the user signed up on the platform in Coordinated Universal Time. |
| is_foreign_user                | Boolean   | Whether or not the user is domestic to the US. |
| acquisition_category           | String    | The acquisition channel a user signed up through (Referrals, Digital Ads, Social Media, etc). |
| acquisition_subcategory        | String    | The acquisition channel's subcategory a user signed up through (Facebook, Snapchat, In-restaurant promotions, etc). |
| language                       | String    | The user selected language they prefer. Can be Unknown is not-selected. |
| last_order_time_utc            | DateTime  | The date and time of the user's last order on the platform in Coordinated Universal Time. **This is used to calculate the positive and negative class.**|
| signup_to_order_hours          | Float     | The number of hours between signing up and a user's first order. |
| days_since_signup              | Integer   | The number of days since a user has signed up. |
| first_order_driver_rating      | Integer   | The optional rating a user gave their courier on their first order. Can be -1 if there was no rating. |
| first_order_avg_meal_rating    | Float     | The optional average rating a user gave their meals on their first order. Can be -1 if there were no ratings. An order can have multiple meals included. |
| first_order_median_meal_rating | Float     |The optional median rating a user gave their meals on their first order. Can be -1 if there were no ratings. An order can have multiple meals included.|
| first_order_delivered_on_time  | Boolean   | Whether or not a user's first order was delivered by the time promised in the app. |
| first_order_hours_late         | Float     | The number of hours late a user's first order was delivered. If an order was delivered on time, this equals 0. |
| first_order_gmv                | Float     | The Gross Merchandise Value (in USD) of a user's first order. This is the total cost of an order, including things like taxes, fees, and tips. |
| first_order_payment            | Float     | The amount a user paid the Company on their first order (in USD). This is net of the promos or discounts a user was offered. |
| first_order_discount_amount    | Float     | The difference between the GMV and an order payment (in USD). This is equal to any promos or discounts a user was offered. |
| first_order_discount_percent   | Float     | The % of the GMV that was discounted on a user's first order |
| first_order_meal_reviews       | Integer   | The number of meal reviews that a user rated during their first order. An order can have multiple meals included. |
| first_30_day_orders            | Integer   | The number of orders a user placed during their first 30 days after their first order. |
| first_30_day_avg_meal_rating   | Float     | The average meal rating for all the orders a user placed during their first 30 days after their first order. |
| first_30_day_avg_driver_rating | Float     | The average driver rating for all the orders a user placed during their first 30 days after their first order. |
| first_30_day_avg_gmv           | Float     | The average GMV (in USD) for all the orders a user placed during their first 30 days after their first order. |
| first_30_day_discount_percent  | Float     | The average discount percent for all the orders a user placed during their first 30 days after their first order. |
| first_30_day_subscription_user | Boolean   | Whether or not a user was a subscription user during their first 30 days. |

Before starting my EDA, I needed to separate out a holdout dataset that I would use to test the performance of my best classifiers. I shuffled the data, stratified based on my target class, and then split out 20% to hold back for testing purposes.

### Data Cleaning
Most of my data cleaning occurred when I was writing the query itself, which included:
* Grouping cities into `College Town` town or `City`
* Converting local currencies to USD
* Filtering for only completed orders
* Converting NaNs to -1 for ratings and 0 for late orders.
* Calculating 30 day and last order metrics

## EDA

### KDE Plots for Continuous Features

### 100% Bar Charts for Categorical Features

### Comparing 30, 60, 90 days for Churn Prediction

![ROC AUC Score for 30,45,60,90](images/roc_curves.png)

## Model Tuning & Performance

After determining my best model, I wanted to look at which features it determined were most important:

| Feature                        | Importance % |
|--------------------------------|--------------|
| Days Since Signup | 19.8% |
| First 30 Day Orders | 19.3% |
| Signup To Order Hours | 16.8% |
| First 30 Day Discount Percent | 6.4% |
| First 30 Day Avg Gmv | 6.1% |
| First Order Payment | 4.5% |
| First Order Gmv | 4.2% |
| First Order Discount Percent | 4.0% |
| City Name | 3.8% |
| First Order Discount Amount | 3.8% |
| First 30 Day Avg Meal Rating | 2.2% |
| Acquisition Subcategory | 1.5% |
| First Order Hours Late | 1.5% |
| First Order Meal Reviews | 1.2% |
| Language | 0.9% |
| Acquisition Category | 0.8% |
| First Order Avg Meal Rating | 0.6% |
| Foreign User | 0.6% |
| First 30 Day Avg Driver Rating | 0.5% |
| First 30 Day Subscription User | 0.4% |
| City Group | 0.4% |
| First Order Driver Rating | 0.3% |
| First Order Median Meal Rating | 0.2% |
| First Order Delivered On Time | 0.1% |

## Model and Threshold Selection

## Conclusions

### Lessons Learned

## Future Work