# Rossman-Project- Sales Prediction
**Business Problem:**  Rossmann is one of the largest drug store chains in Europe with more than 4000 stores. The company wants to settle a budget to renovate the stores, based on the sales that the stores will make in the next weeks. This project was developed to predict the sales from the stores in the next 6 weeks and it was inspired in a competition from Kaggle (https://www.kaggle.com/c/rossmann-store-sales). Data was downloaded from CSVs files, and it contains the information below. Sales is the response variable, and the goal of the project is to predict them, considering the other features.

**Id** - an Id that represents a (Store, Date) duple within the test set

**Store** - a unique Id for each store

**Customers** - the number of customers on a given day

**Open** - an indicator for whether the store was open: 0 = closed, 1 = open

**StateHoliday** - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public 
holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

**SchoolHoliday** - indicates if the (Store, Date) was affected by the closure of public schools

**StoreType** - differentiates between 4 different store models: a, b, c, d

**Assortment** - describes an assortment level: a = basic, b = extra, c = extended

**CompetitionDistance** - distance in meters to the nearest competitor store

**CompetitionOpenSince [Month/Year]** - gives the approximate year and month of the time the nearest competitor was opened

**Promo** - indicates whether a store is running a promo on that day

**Promo2** - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating

**Promo2Since[Year/Week]** - describes the year and calendar week when the store started participating in Promo2

**PromoInterval** - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb, May, Aug, Nov" means each round starts in February, May, August, November of any given year for that store

**Sales** - the turnover for any given day (this is what you are predicting) 

**Assumptions:** Stores that have “Na” in “competition_distance”, don’t have any competitors nearby.

**Solution Steps:**

**Loading data:** Three files that contain information from the stores, training and testing data were downloaded from the Kaggle platform. 

**Descriptive analyses and data cleaning:** The purpose of this step was to get familiarized with the data, and fill blank fields based on the business assumptions. Data was divided into categorical and numerical features.  To the numerical features, statistical numbers such as mean, median, standard deviation were calculated.  To the categorical features, boxplots were plotted to describe the data.

**Feature Engineering:** The features "year", "month", "day", "week_of_year", "year_week", "competition_time_days" and "competition_time_month" were created from the original features. These features can be useful to explain the sales. 

**Exploratory Data Analysis:** Univariate, bivariate and multivariate analysis were done to understand how features are related to the sales and to each other. Some hypotheses were created to validate how some of the features are related to the sales:
1)Stores that have competition nearby sell less - **False**
2)Stores sell more in the second semester of the year - **False**
3)Stores sell more in the last days of the month - **False**

**Data Preparing:**  Features that have a large range were rescaled. This step is important so that the model doesn’t provide a larger weight to these features. The features "competition_distance" and "competition_time_month" were rescaled using RobustScaler, that is robust to outliers. The features "promo_time_weeks" and "year" were rescaled using MinMaxScaler.

Categorical features were transformed into numerical features, so that they can also be used in the model. The features "state_holiday", "store_type" and "assortment" were transformed respectively by the "one hot encoding", "label encoding" and "ordinal encoding" algorithms.
Features that have cyclical nature, such as "day_of_week", "month, day" and "week_of_year" were also transformed using sine and cosine, so that the machine learning model could better understand them.

The response variable "sales" was normalized applying the logarithm to the original values. Machine Learning models work better with features that are normalized.

**Feature Selection:** Features that were considered relevant to explain the response variable "sales" were selected to be used in the machine learning model.  The selection was made using Boruta, that is an algorithm designed to automatically perform feature selection on a dataset. The features selected were: "store", "promo", "store_type", "assortment", "competition_distance", "competition_open_since_month", "competition_open_since_year", "promo2","promo2_since_week", "promo2_since_year", "competition_time_month", "promo_time_weeks", "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos", "day_sin",  "day_cos", "week_of_year_sin", "week_of_year_cos"

**Machine Learning Modeling:** Four models were implemented, and the error from these models were calculated, to compare which model should be used.

**-Average Model:** It considers the mean sales value from each store, to predict the sales.

**-Linear Regression Model:**  Linear regression models use a straight line to predict the sales. During the training stage, if this model feels like one feature is particularly important, the model may place a large weight to the feature. 

**-Linear Regression Regularized (Lasso):** Lasso is a modification of linear regression, where the model is penalized for the sum of absolute values of the weights.

**-XGBoost Regression:** XGBoost is an implementation of the gradient boosted trees algorithm. Gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target variable.
For each model the errors MAE, MAPE and RMSE were calculated.  The errors consider the difference between what the model is predicting and the actual sales value.

**Cross validation:** To verify the error in the different data partitions, five different data groups were created to be the testing data, and the error was calculated to each of these parts.  The average error was calculated in the table below:
|Model Name|	MAE CV|	MAPE CV	|RMSE CV|
|----------|--------|---------|-------|
|XGBoost|	1044.24 +/- 167.49|	0.14 +/- 0.02|	1504.4 +/- 229.6|
|Linear Regression|	2082.37 +/- 295.54|	0.3 +/- 0.02|	2953.01 +/- 467.76|
|Lasso|	2116.38 +/- 341.5|	0.29 +/- 0.01|	3057.75 +/- 504.26|

**Hyperparameter fine tuning:** Machine learning models have hyperparameters that you must set in order to customize the model to your dataset. In order to optimize these parameters, the model was run using random values of parameter, and the error was calculated to each set of parameters.  The set of parameters that presented the smallest error was chosen.

**Interpreting the error:** MAE is the mean absolute error. It uses the same scale as the data being measured 
The best scenario considered the prediction of sales + MAE 
The worst scenario considered the prediciton of saled - MAE 
The table below shows 10 stores and the predictions of sales, worst scenarios, best scenarios, MAE and MAPE 
|store|	predictions|best_scenario|worst_scenario|	MAE|MAPE|
|-----|------------|-------------|--------------|----|----|
|292|103509.578125|106806.039217|100213.117033|3296.461092|0.539909|
|909|241141.578125|248520.668181|233762.488069|7379.090056|0.503596|
|876|202002.296875|205950.086521|198054.507229|3947.789646|0.312363|
|722|352810.000000|354796.865089|350823.134911|1986.865089|0.267348|
|274|197443.562500|198782.167151|196104.957849|1338.604651|0.234801|
|334|227514.703125|228663.345334|226366.060916|1148.642209|0.226468|
|657|234500.734375|235565.616884|233435.851866|1064.882509|0.223573|
|578|341358.968750|343721.455805|338996.481695|2362.487055|0.217949|
|595|413537.625000|416663.986170|410411.263830|3126.361170|0.212382|
|1073|283675.062500|285643.469120|281706.655880|1968.406620|0.211750|

**Total Performance:**  In the table below, it can be seen the total sales predictions considering all stores, the best scenario and the worst scenario of sales:

|Scenario|	Total Sales|
|--------|-------------|
|Predictions|R$284,191,744.00|
|Best Scenario|R$284,975,306.72|
|Worsts Scenario|R$283,408,168.78|


**Deploy model to production:** The model was deployed in Heroku cloud, and a bot in telegram was created to access the results. The user needs to enter / + the number of the store, and a message with the sales predictions will be sent back to him. 
Image
