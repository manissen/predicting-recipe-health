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
To make the data easier to read and use, we merged the two datasets.
'''py
merged_data = recipes.merge(ratings, left_on='id', right_on='recipe_id', how='left')
merged_data['rating'] = merged_data['rating'].replace(0.0, np.nan)
'''

## Assessment of Missingness

## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis

