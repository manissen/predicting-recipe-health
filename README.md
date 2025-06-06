# Predicting Recipe Health

By: Mieko Chun and Margot Nissen

## Overview
This DSC 80 project focuses on predicting how healthy a recipe is through a number of features such as the nutrition ratings and the key heath tags associated with each recipe.

## Introduction
As heath and nutrition becomes a greater issue, it is important for people to understand the nutritional content of a recipe. This project aims to help users make better decisions about their diet by investigating the healthiness of recipes based on their nutritioal ratings and health tags.

The two datasets we used were 'Recipes' and 'Interactions'. The 'Recipe' dataframe has 83782 rows and 12 columns. Here are the relevant columns and a brief description:

| Column | Desciption |
| :------ | :--------------------------- |
| 'name' | Name of the recipe |
| 'id' | ID of the recipe |
| 'minutes' | How long it took to make the recipe |
| 'submitted' | When the recipe was added to the wesbite |
| 'tags' | Different descriptors about the recipe |
| 'nutrition' | Nutritional facts about the rating including calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates (PDV)|
| 'ingredients' | Ingredients needed for the recipe |

The 'Interactions' dataframe contains 731927 rows and 5 columns. Here are the relevant columns for our project:

| Column | Description |
| :------ | :--------------------------- |
| 'recipe_id' | ID of the recipe |
| 'date' | When the user submitted their rating |
| 'rating' | Rating of the recipe made by the specific user |
| 'review'| Comment made by the user about the recipe |

The 'Interactions' dataframe has more observations because there are multiple ratings per recipe. Some recipes don't contain any ratings.

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning
1. To make the data easier to read and use, we merged the two datasets.
     ```py
     merged_data = recipes.merge(ratings, left_on='id', right_on='recipe_id', how='left')
     merged_data['rating'] = merged_data['rating'].replace(0.0, np.nan)
     ```
     - This merge kept all the recipes with ratings.

2. Cells without any ratings were replaced with np.NaN so that the missing ratings would not affect the average rating.
3. Added a column called 'avg_rating' which computed the average rating of each recipe
    - This gave us a generalized rating of each recipe we could baseline the individual ratings on
4. Converted the items in the nutrition list from strings to floats
     - This allows us to manipulate the data and create health scores described below
5. Healthy vs Unhealthy recipes can be a bit arbitrary, so we incorporated different ways of measuring health. We first analyzed the 'tags' created by food.com and categorized some of them into 'healthy', 'medium healthy', and 'unhealthy'. Below, we have included all of the keywords for the three categories:
     ```py
     healthy_keywords = {'healthy-2', 'healthy', 'salads', 'chard', 'vegan', 'very-low-carbs', 'vegetarian', 'high-fiber', 'spinach', 'low-carb', 'low-sodium', 'low-calorie', 'vegetables', 'low-fat', 'low-saturated-fat'}
     midhealthy_keywords = {'high-protein', 'pork-sausage', 'smoothies', 'desserts-fruit', 'low-in-something', 'pot-pie', 'dairy-free', 'gluten-free', 'casseroles', 'tex-mex'}
     unhealthy_keywords = {'drop-cookies', 'desserts', 'super-bowl', 'brownies', 'cakes', 'cake-fillings-and-frostings', 'fudge', 'rolled-cookies', 'cookies-and-brownies', 'cupcakes', 'desserts-easy', 'pies-and-tarts', 'sugar-cookies', 'fillings-and-frostings-chocolate', 'chocolate-chip-cookies', 'ice-cream'}
    ```
    - We created a new column called 'health_rating' and applied the 'healthy', 'medium healthy', or 'unhealthy' categorizer if the recipes' tags included any of the respective keywords.
6. The second measure of healthiness was by using the nutritional facts of each recipe. We weighted the different nutritional facts based on their nutritional impact. Sugar has a higher impact because it is a strong negative health indicator. Meanwhile, protein is generally considered to be a positive nutrient, especially when it comes from unprocessed sources.

     | Nutritional Fact | Nutritional Impact |
     | :--------------- | :------------------ |
     | calories (#) | 0.15 |
     | total fat (PVD) | 0.15 |
     | sugar (PVD) | 0.3 |
     | sodium (PVD) | 0.1 |
     | protein (PVD) | -0.2 |
     | saturated fat (PVD) | 0.25 |
     | carbohydrates (PVD) | 0.1 |
 
     - Since protein can be beneficial to one's health, we made the impact of high protein negative.

    - We added a column called health_score which was the sum of all the nutritional facts multiplied by their nutritional impact.
The higher the health_score, the more unhealthy the recipe is.
7. We grouped the data by the name of the recipe.
    - This made sure that recipes with more ratings wouldn't have as big of an impact on our analysis

Our final dataframe contains 83628 rows and 20 columns. He is an example of what one row looks like. We included the columns that are most important for our project.

| name | id | tags | nutrition | n_ingredients | rating | avg_rating | difficulty| health_rating | health_score |
| :----| :- | :----| :-------- | :------------ | :------| :----------| :-------- | :-------------| :----------- |
|0 carb 0 cal gummy worms|45|['60-minutes-or-less', 'time-to-make', ...]|[384.7, 0.0, 0.0, 70.0, 159.0, 0.0, 6.0|3|5|4.75|intermediate|unhealthy|33.5|


### Univariate Analysis
This histogram shows the distribution of Health Scores with the outliers removed.

Mean: 97.712
Median: 64.999

<iframe
  src="assets/univariate_health_scores.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Assessment of Missingness

## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis

