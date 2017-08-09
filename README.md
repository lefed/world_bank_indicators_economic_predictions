# World Bank Development Indicators - Use for Economic Predictions

This project was done to practice machine learning classification modeling. 
My model was optimized to predict whether an economy would be growing or shrinking in the next measured year. 
It uses the World Bank Open Data Initiative's world development indicators published in July 2017.

### World Bank Data Explorations and Model Selection
The data used in this modeling and exploration process was taken from the [World Bank's Open Data Initiative Data Catalog](http://data.worldbank.org/data-catalog/).

The goal of this exercise was to use development indicators such as those related to education, health, environment, and economic industry to create a classification model to predict whether a economy would be likely to grow or shrink in any particular upcoming year.

Through an iterative process of feature engineering that involved finding development indicators that correlated to an increased or decreased likelihood of an economy growing or shrinking, I chose to use features related to 32 of a possible 140+ development indicators. This provided the best balance of preservation of data size and quality as well as giving interesting and useful features to predict economic growth.

By choosing this number of features, and by doing minimal data replacement (whereby for a NaN value I replaced the value with the mean value for that country), I was able to use data from 5,556 economy instances between 1960 and 2015.

The features (not necessarily in order of importance) found to best predict whether an economy was about to be in a period of growth or decline turned out to be:

* Annual Inflation Rate
* Carbon Dioxide Damage (% of GNI)
* Industry, value added (% of GDP)
* Total Reserves in Months of Imports
* School enrollment, primary (% gross)
* Mobile cellular subscriptions (per 100 people)
* Fuel imports (% of merchandise imports)
* Domestic credit to private sector (% of GDP)
* Trade (% of GDP)
* Services, etc., value added (% of GDP)
* Imports of goods and services (% of GDP)
* Gross domestic savings (% of GDP)
* Gross savings (% of GNI)
* CO2 emissions (metric tons per capita)
* Deviation from mean - Food production index (2004-2006 = 100)
* Change from previous year in Crop production index (2004-2006 = 100)
* Forest rents (% of GDP)
* Rate of Change of Military expenditure (% of GDP)
* Death rate, crude (per 1,000 people)
* Life expectancy at birth, female (years)
* Life expectancy at birth, male (years)
* Survival to age 65, male (% of cohort)
* Fertility rate, total (births per woman)
* Mortality rate, infant (per 1,000 live births)
* Mortality rate, under-5 (per 1,000 live births)
* Rural population (% of total population)
* Population ages 20-24, female (% of female population)
* Deviation from mean - Population ages 20-24, female (% of female population)
* Population ages 30-34, female (% of female population)
* Population ages 55-59, female (% of female population)
* Age dependency ratio, old (% of working-age population)
* Rate of Change of Age dependency ratio (% of working-age population)

The following heatmap shows the correlation of the various features of my model with eachother based on this data set. The target variable, 'shrinking', is 0 if the economy instance grew in the measurement period following the indicator measurements and 1 if the economy instance was shrinking in the measurement period following the indicator measurements.

![](https://github.com/lefed/world_bank_indicators_economic_predictions/blob/master/graphs/corr_heatmap.png)

In order to better understand model performance of possible classification models such as Logistic Regression, Random Forest Classification, Gradient Boosted Trees, K Nearest Neighbors and others, I plotted each model's precision-recall curve and ROC curve to understand where different models may be performing well and to help with model or ensemble selection.

The following curves show how these models performed over various thresholds for classification as compared to eachother. Note that these models were all done using the final selected 33 features related to 32 development indicators.

![](https://github.com/lefed/world_bank_indicators_economic_predictions/blob/master/graphs/PR_ROC_regression_curves.png)

Based on the results of these curves, I chose to use a Gradient Boosted Trees classification model with tuned learning rate, exponential loss function, and 75 trees.

The precision-recall curve and ROC curve for this specific model can be seen below.

![](https://github.com/lefed/world_bank_indicators_economic_predictions/blob/master/graphs/grad_boosted_trees_pr_roc.png)

The precision-recall tradeoff for various model thresholds for this particular model can be seen in the following graphic. 

![](https://github.com/lefed/world_bank_indicators_economic_predictions/blob/master/graphs/grad_boost_pr_tradeoff.png)

Given that I was thinking of a use case of my model as being used to find economies at risk of decline in order to try to provide intervention, I chose to use a threshold where my model could achieve a recall of 83% - meaning it correctly classified 83% of economies that were about to go into decline - with 25.5% precision. 

Using this threshold resulted in the following scores for my model:

* Precision - 25.5%
* Recall - 82.6%
* F1 Score - 0.389
* Log-Loss - 0.48 (Note that I chose to prioritize good ranking performance over good prediction of probabilities.)
* ROC AUC - 0.66
* Accuracy - 48%

