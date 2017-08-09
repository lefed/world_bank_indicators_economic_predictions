# world_bank_indicators_economic_predictions
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

[[https://github.com/lefed/world_bank_indicators_economic_predictions/graphs/corr_heatmap.png]|alt=feature_correlation_heatmap]
