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
| 'tags' | Different descriptors about the recipe |
| 'nutrition' | Nutritional facts about the rating including calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates (PDV)|
| 'ingredients' | Ingredients needed for the recipe |

The 'Interactions' dataframe contains 731927 rows and 5 columns. Here are the relevant columns for our project:

| Column | Description |
| :------ | :--------------------------- |
| 'recipe_id' | ID of the recipe |
| 'rating' | Rating of the recipe made by the specific user |
| 'review'| Comment made by the user about the recipe |

The 'Interactions' dataframe has more observations because there are multiple ratings per recipe. Some recipes don't contain any ratings.

## Data Cleaning and Exploratory Data Analysis
1. To make the data easier to read and use, we merged the two datasets.
 ```py
 merged_data = recipes.merge(ratings, left_on='id', right_on='recipe_id', how='left')
 merged_data['rating'] = merged_data['rating'].replace(0.0, np.nan)
 ```
 This merge kept all the recipes with ratings. Cells without any ratings were replaced with 0.0 so the dataframe could be easily manipulated   during analysis and testing.

2. We added a few columns to accurately reflect a ratings health. Healthy vs Unhealthy recipes can be a bit arbitrary, so we incorporated different ways of measuring health. We first analyzed the 'tags' created by food.com and categorized some of them into 'healthy', 'medium healthy', and 'unhealthy'. Tags such as 'healthy', 'salads', and 'low-sodium' were keywords for 'healthy', while 'desserts', 'chocolate', and 'super-bowl' were categorized as 'unhealthy'. Below, we have included all of the keywords for the three categories:
 ```py
 healthy_keywords = {'healthy-2', 'healthy', 'salads', 'chard', 'vegan', 'very-low-carbs', 'vegetarian', 'high-fiber', 'spinach', 'low-carb', 'low-sodium', 'low-calorie', 'vegetables', 'low-fat', 'low-saturated-fat'}
 midhealthy_keywords = {'high-protein', 'pork-sausage', 'smoothies', 'desserts-fruit', 'low-in-something', 'pot-pie', 'dairy-free', 'gluten-free', 'casseroles', 'tex-mex'}
 unhealthy_keywords = {'drop-cookies', 'desserts', 'super-bowl', 'brownies', 'cakes', 'cake-fillings-and-frostings', 'fudge', 'rolled-cookies', 'cookies-and-brownies', 'cupcakes', 'desserts-easy', 'pies-and-tarts', 'sugar-cookies', 'fillings-and-frostings-chocolate', 'chocolate-chip-cookies', 'ice-cream'}
```

 We created a new column called 'health_rating' and applied the 'healthy', 'medium healthy', or 'unhealthy' if the recipes' tags included any of the keywords.

3. The second measure of healthiness was by using the nutritional facts of each recipe. We weighted the different nutritional facts based on their nutritional impact

 calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates (PDV)

 | Nutritional Fact | Nutritional Impact |
 | :--------------- | :------------------ |
 | calories (#) | 0.15 |
 | total fat (PVD) | 0.15 |
 | sugar (PVD) | 0.3 |
 | sodium (PVD) | 0.1 |
 | protein (PVD) | -0.2 |
 | saturated fat (PVD) | 0.25 |
 | carbohydrates (PVD) | 0.1 |

 Since protein can be beneficial to one's health, we made the impact of high protein negative. 
 
 We added a column called 'health_score' which was the sum of all the nutritional facts multiplied by their nutritional impact. **The higher the 'health_score', the more unhealthy the recipe is.**

## Assessment of Missingness

## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis

