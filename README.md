 ![Cover](/images/cover.png)
# Machine Learning Project: Predicting Airbnb Listing Prices


## Overview

This project focused on building a machine learning pipeline to predict Airbnb listing prices in New York City.  The goal was to develop an accurate pricing model while uncovering key features that influence Airbnb prices. It demonstrated skills in exploratory data analysis (EDA), regression modeling, feature engineering, and model evaluation, while maintaining a strong business-driven mindset.


## Authors

- Jiamin (Josh) Pai  &nbsp;<a href="https://www.linkedin.com/in/josh-pai"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="height: 1em; width:auto;"/></a> &nbsp; <a href="https://github.com/Josh-Pai"> <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="height: 1em; width: auto;"/></a>



## Business Motivation

For hosts and property managers, setting the right price is critical to maximizing occupancy and revenue. Overpricing can drive guests away, while underpricing leaves money on the table. Understanding which factors (e.g., room type, neighborhood, property features) drive price differences can also offer actionable insights to optimize listing strategies.



## Data Source

- Listings dataset from Inside Airbnb (New York City snapshot)
- Over 30,000 records with detailed information on listing attributes, availability, location, and historical reviews



## Technologies Used

- Python (pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost)
- Jupyter Notebook



## Project Workflow

### Exploratory Data Analysis (EDA)

Analyzed the distribution of price and applied log-transformation to address heavy right-skewness.

  ![Price Distribution](/images/price_distribution.png)
  
  > The original price distribution was highly right-skewed, with most listings priced under $500.

  ![Log Price Distribution](/images/log_price_distribution.png)
  
  > Log-transforming prices resulted in a more symmetric, approximately normal distribution, making the data more suitable for regression modeling.

Investigated feature relationships using selected scatter plots and box plots, highlighting Log Price vs. Accommodates and Log Price by Neighbourhood Group.

  ![Log Price vs Accommodates](/images/log_price_vs_accommodates.png)
  
  > A positive but diminishing relationship between the number of accommodates and log price, suggesting limited marginal returns.

  ![Log Price by Neighbourhood Group](/images/log_price_by_neighbourhood.png)
  
  > Listings in Manhattan and Brooklyn generally command higher prices compared to other boroughs.

Identified key numerical and categorical predictors using a correlation heatmap.

  ![Correlation Heatmap](/images/correlation_matrix_with_log_price.png)
  
  > Accommodation size, number of bedrooms, and bathrooms showed a moderate positive correlation with log-transformed price, while other features exhibited weak or negligible relationships.


### Data Preprocessing

Removed price outliers using the Interquartile Range (IQR) method.
Performed one-hot encoding on categorical variables like room type and neighborhood group.
Created clean train, exploration (validation), and test sets (60%-20%-20% split).


### Modeling

Trained and evaluated three models:
  1. Linear Regression
  2. Random Forest Regressor
  3. XGBoost Regressor

For Linear Regression, the model learned the following equation for predicting the log-transformed price:

```text
Log(Price) = 4.573 + 0.045 * bedrooms - 0.023 * bathrooms + 0.091 * accommodates
           + 0.182 * neighbourhood_group_cleansed_Brooklyn
           + 0.416 * neighbourhood_group_cleansed_Manhattan
           + 0.025 * neighbourhood_group_cleansed_Queens
           - 0.015 * neighbourhood_group_cleansed_Staten Island
           + 0.090 * room_type_Hotel room
           - 0.512 * room_type_Private room
           - 0.524 * room_type_Shared room
```

Each coefficient reflects how much the log price is expected to change with a one-unit increase in the corresponding feature, holding others constant.



## Evaluation Metrics

**R² (coefficient of determination)** and **RMSE (Root Mean Squared Error)** were used to evaluate model performance on both log-transformed and original prices.



## Feature Importance Analysis

Extracted and visualized feature importance scores from **Linear Regression**, **Random Forest**, and **XGBoost** models to interpret key pricing drivers and understand the relative impact of different features.

  ![Linear Regression Top 10 Features](/images/top_10_features_lr.png)

  ![Random Forest Top 10 Features](/images/top_10_features_rf.png)

  ![XGBoost Top 10 Features](/images/top_10_features_xgb.png)

  > Room type, accommodation size, and neighborhood group consistently emerged as key factors across different models.



## Results

Performance comparison of three regression models:

| Model               | R² (Log Price) | RMSE (Log Price) | R² (Original Price) | RMSE (Original Price) |
|---------------------|----------------|------------------|---------------------|-----------------------|
| Linear Regression   | 0.411           | 0.485            | 0.301               | 82.78                 |
| Random Forest       | 0.493           | 0.450            | 0.398               | 76.81                 |
| XGBoost             | 0.491           | 0.451            | 0.393               | 77.13                 |

🏆 **Best Model**  
Random Forest achieved the best overall performance with the highest R² and the lowest RMSE.

> Alongside the quantitative evaluation, visualizations such as scatter plots of predicted vs. actual prices, residual plots, and feature importance charts were incorporated during modeling stages to enhance analysis and interpretation.
