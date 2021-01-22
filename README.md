# Project 2 - Ames Housing Data and Kaggle Challenge

Kelly Slatery
US-DSI-10
01.17.2020

## Project Directory
```
project-2
|__ code
|   |__ 01_EDA_Cleaning_Models_Submissions.ipynb   
|   |__ 02_Clean_Test_Data.ipynb   
|__ data
|   |__ data_dictionary.csv
|   |__ train.csv
|   |__ train_clean.csv
|   |__ train_final.csv
|   |__ test.csv
|   |__ test_clean.csv
|   |__ 01_submission.csv
|   |__ 02_submission.csv
|   |__ 03_submission.csv
|   |__ 04_submission.csv
|   |__ 05_submission.csv
|   |__ 06_submission.csv
|   |__ 07_submission.csv
|   |__ 08_submission.csv
|   |__ 09_submission.csv
|   |__ 10_submission.csv
|   |__ 11_submission.csv
|   |__ 12_submission.csv
|   |__ 13_submission.csv
|   |__ 14_submission.csv
|   |__ 15_submission.csv
|   |__ 16_submission.csv
|   |__ 17_submission.csv
|   |__ 18_submission.csv
|   |__ 19_submission.csv
|   |__ 20_submission.csv
|   |__ 21_submission.csv
|__ presentation.pdf
|__ README.md
```



## Problem Statement

Real estate and property appraisal are fields based on expertise knowledge. All existing Automated Valuation Models (AVMs) market themselves as a reference point, but advise home buyers and sellers to consult an agent for accurate, up-to-date property value estimates. While in the scope of this project, we are only looking at data for individual residential properties sold between 2006 and 2010 in the city of Ames, Iowa, most prominent AVMs estimate tens to hundreds of millions of home values in a wide range of cities around the U.S. or even the world. Considering the features that optimize sale price predictions on this dataset, what features most considerably impact sale price? What features would be useful to have here to make predictions for sale prices in Ames, Iowa? And, ultimately, just what makes for a good property value estimator?



## Executive Summary

In this project, we explore the Ames Housing Dataset, originally provided by the Ames Assessor’s Office. The original dataset contains 2051 observations and 81 columns containing variables describing features of residential properties sold in Ames, Iowa from 2006 to 2010. Based on extensive data cleaning, exploratory data analysis (EDA), and visualization, we will train, evaluate, and finetune various machine learning linear regression models to predict the price of a house for sale in the Ames Housing Market (between 2006-2010). 

Through initial exploration, it became clear that there were many erroneous data types and null values to be addressed. Upon cleaning the data, some major patterns emerged. The first was the strength of various measures of quality in predicting sale price. Our final model includes 6 features explicitly labeled as "quality" ('Overall Qual', 'Exter Qual', 'Kitchen Qual', 'Bsmt Qual', 'Fireplace Qu', 'Heating QC') as well as a feature labeled "condition" ('Garage Cond') and another feature with three possible appreciations ('Paved Drive'). A few models performed very highly on the train data, but were overfit and received low scores in Kaggle. Additionally, standardized linear regression, lasso, and ridge variants used with the final set of features chosen did not perform well, indicating that these features are most likely optimized for a linear regression model, but not necessarily the strongest predictors of price. Because the model chosen for evaluation in this model did not use standardized variables, the coefficients are not highly interpretable. However, through the process of model tuning and evaluation, I was able to draw insights into what makes for a good property value estimator, as described in [Conclusions and Recommendations](#Conclusions-and-Recommendations).



## Contents:
- [Import and Explore Data](#Import-and-explore-data)
- [Data Cleaning](#Clean-data)
- [Exploratory Data Analysis](#Plot,-explore,-&-interpret-data)
- [Model Exploration](#Create-Machine-Learning-models)
- [Model Evaluation](#Visualize-results)
- [Model Finetuning and Reevaluation](#Model-exploration-and-evaluation)
- [Conclusions and Recommendations](#Conclusions-and-Recommendations)



## Data Dictionary

The data dictionary for the original 81 columns can be found at:<br /> 
[Data Dictionary: Ames Housing Dataset (Source: Journal of Statistics Education)](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt)

In this project, 17 of the original categorical columns were adjusted to numerical values. In addition to the original 81 columns, 65 additional columns were added for data analysis purposes. These new and adjusted columns are described below:

| |**variable**|**type**|**description**|**component columns**|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|0|zone\_C (all)|uint8|Dummy for 'MS Zoning'|MS Zoning|
|1|zone\_FV|uint8|Dummy for 'MS Zoning'|MS Zoning|
|2|zone\_I (all)|uint8|Dummy for 'MS Zoning'|MS Zoning|
|3|zone\_RH|uint8|Dummy for 'MS Zoning'|MS Zoning|
|4|zone\_RL|uint8|Dummy for 'MS Zoning'|MS Zoning|
|5|zone\_RM|uint8|Dummy for 'MS Zoning'|MS Zoning|
|6|lot\_config\_CulDSac|uint8|Dummy for 'Lot Config'|Lot Config|
|7|lot\_config\_FR2|uint8|Dummy for 'Lot Config'|Lot Config|
|8|lot\_config\_FR3|uint8|Dummy for 'Lot Config'|Lot Config|
|9|lot\_config\_Inside|uint8|Dummy for 'Lot Config'|Lot Config|
|10|neighborhood\_Blueste|uint8|Dummy for 'Neighborhood'|Neighborhood|
|11|neighborhood\_BrDale|uint8|Dummy for 'Neighborhood'|Neighborhood|
|12|neighborhood\_BrkSide|uint8|Dummy for 'Neighborhood'|Neighborhood|
|13|neighborhood\_ClearCr|uint8|Dummy for 'Neighborhood'|Neighborhood|
|14|neighborhood\_CollgCr|uint8|Dummy for 'Neighborhood'|Neighborhood|
|15|neighborhood\_Crawfor|uint8|Dummy for 'Neighborhood'|Neighborhood|
|16|neighborhood\_Edwards|uint8|Dummy for 'Neighborhood'|Neighborhood|
|17|neighborhood\_Gilbert|uint8|Dummy for 'Neighborhood'|Neighborhood|
|18|neighborhood\_Greens|uint8|Dummy for 'Neighborhood'|Neighborhood|
|19|neighborhood\_GrnHill|uint8|Dummy for 'Neighborhood'|Neighborhood|
|20|neighborhood\_IDOTRR|uint8|Dummy for 'Neighborhood'|Neighborhood|
|21|neighborhood\_Landmrk|uint8|Dummy for 'Neighborhood'|Neighborhood|
|22|neighborhood\_MeadowV|uint8|Dummy for 'Neighborhood'|Neighborhood|
|23|neighborhood\_Mitchel|uint8|Dummy for 'Neighborhood'|Neighborhood|
|24|neighborhood\_NAmes|uint8|Dummy for 'Neighborhood'|Neighborhood|
|25|neighborhood\_NPkVill|uint8|Dummy for 'Neighborhood'|Neighborhood|
|26|neighborhood\_NWAmes|uint8|Dummy for 'Neighborhood'|Neighborhood|
|27|neighborhood\_NoRidge|uint8|Dummy for 'Neighborhood'|Neighborhood|
|28|neighborhood\_NridgHt|uint8|Dummy for 'Neighborhood'|Neighborhood|
|29|neighborhood\_OldTown|uint8|Dummy for 'Neighborhood'|Neighborhood|
|30|neighborhood\_SWISU|uint8|Dummy for 'Neighborhood'|Neighborhood|
|31|neighborhood\_Sawyer|uint8|Dummy for 'Neighborhood'|Neighborhood|
|32|neighborhood\_SawyerW|uint8|Dummy for 'Neighborhood'|Neighborhood|
|33|neighborhood\_Somerst|uint8|Dummy for 'Neighborhood'|Neighborhood|
|34|neighborhood\_StoneBr|uint8|Dummy for 'Neighborhood'|Neighborhood|
|35|neighborhood\_Timber|uint8|Dummy for 'Neighborhood'|Neighborhood|
|36|neighborhood\_Veenker|uint8|Dummy for 'Neighborhood'|Neighborhood|
|37|bldg\_type\_2fmCon|uint8|Dummy for 'Bldg Type'|Bldg Type|
|38|bldg\_type\_Duplex|uint8|Dummy for 'Bldg Type'|Bldg Type|
|39|bldg\_type\_Twnhs|uint8|Dummy for 'Bldg Type'|Bldg Type|
|40|bldg\_type\_TwnhsE|uint8|Dummy for 'Bldg Type'|Bldg Type|
|41|total\_full\_bath|int64|Added 'Full Bath' and 'Bsmt Full Bath'|'Full Bath', 'Bsmt Full Bath'|
|42|total\_half\_bath|int64|Added 'Half Bath' and 'Bsmt Half Bath'|'Half Bath', 'Bsmt Half Bath'|
|43|lot\_size|float64|Interaction term|'Lot Area', 'Lot Frontage'|
|44|year\_built\_remod/add|int64|Interaction term|'Year Built', 'Year Remod/Add'|
|45|area\_rms\_abvgrd|int64|Interaction term|'TotRmsAbvGrd', 'Gr Liv Area'|
|46|bathrooms|int64|Interaction term|'Full Bath', 'Half Bath'|
|47|garage\_size|float64|Interaction term|'Garage Cars', 'Garage Area'|
|48|garage\_cond\_qual|int64|Interaction term|'Garage Cond', 'Garage Qual'|
|49|has\_condition|int64|Made 'Condition1' and 'Condition2' binary|'Condition1', 'Condition2'|
|50|bsmtfin\_sf1\_total\_sf|float64|Interaction term|'BsmtFin SF 1', 'Total Bsmt SF'|
|51|beds\_baths|int64|Interaction term|'total\_full\_bath', 'Bedroom AbvGr'|
|52|func\_overall\_qual|int64|Interaction term|'Functional', 'Overall Qual'|
|53|lot\_area\_zone\_C (all)|int64|Interaction term|'Lot Area' and 'MS Zoning' dummies|
|54|lot\_area\_zone\_FV|int64|Interaction term|'Lot Area' and 'MS Zoning' dummies|
|55|lot\_area\_zone\_I (all)|int64|Interaction term|'Lot Area' and 'MS Zoning' dummies|
|56|lot\_area\_zone\_RH|int64|Interaction term|'Lot Area' and 'MS Zoning' dummies|
|57|lot\_area\_zone\_RL|int64|Interaction term|'Lot Area' and 'MS Zoning' dummies|
|58|lot\_area\_zone\_RM|int64|Interaction term|'Lot Area' and 'MS Zoning' dummies|
|59|house\_style\_bldg\_type\_2fmCon|float64|Interaction term|'House Style' and 'Bldg Type' dummies|
|60|house\_style\_bldg\_type\_Duplex|float64|Interaction term|'House Style' and 'Bldg Type' dummies|
|61|house\_style\_bldg\_type\_Twnhs|float64|Interaction term|'House Style' and 'Bldg Type' dummies|
|62|house\_style\_bldg\_type\_TwnhsE|float64|Interaction term|'House Style' and 'Bldg Type' dummies|
|63|SalePrice\_preds|float64|Sale Price predictions from most accurate model|All features in final\_features|
|64|errors|float64|Actual sale prices - predicted sale prices (errors)|'SalePrice', 'SalePrice_preds'|



## Conclusions & Recommendations

After evaluating the data, features, limitations, and median error rate of four of the most popular AVMs on the market, it became clear that there are three major items that would enhance this model for the Ames Housing Market 2006-2010:

1. Better data (i.e. market conditions, location-specific criteria such as schools, crime rates, proximity to public transport and stores, etc.)
2. Neighborhood-specific models (all AVMs analyzed use some sort of model comparing house sale prices in similar areas and determining weight of different features based on location)
3. Expertise knowledge (as a data scientist, I can interpret the significance of the numbers and make inferences, but, as mentioned above, real estate and property appraisal are incredibly nuanced)

In addition, I make four recommendations going forward for anyone hoping to create an AVM / property value estimator, as illustrated in my process below:

1. Simplicity is key for generalization
2. Determine if outliers are part of “irreducible error” noise or the “Joe Biden Effect”
3. A good model starts with good data
4. In real estate, nuance is key

In conclusion, the models built in this notebook illustrate a process of data cleaning, EDA, data visualization, and model tuning that led me to draw the conlusions above. The final model that is used for comparison with larger AVMs uses 23 features, 4 engineered in this project to filter for predictive categories and reflect real-life predictors' interactions. However, without more comprehensive data and more nuanced models informed by expertise knowledge, any sale price predictions will be more for interest than practical market use.