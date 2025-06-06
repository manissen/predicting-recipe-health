# Predicting Recipe Health

By: Mieko Chun and Margot Nissen

## Overview
This DSC 80 project focuses on predicting how healthy a recipe is using features such as nutrition ratings and health tags associated with each recipe.

## Introduction
As health and nutrition become a greater concern, it's important for people to understand the nutritional content of a recipe. This project helps users make better dietary decisions by analyzing recipes through their nutritional ratings and tags.

We used two datasets: **Recipes** and **Interactions**.

### Recipe Dataset (83,782 rows, 12 columns)

| Column        | Description                                                                  |
|---------------|------------------------------------------------------------------------------|
| `name`        | Name of the recipe                                                           |
| `id`          | ID of the recipe                                                             |
| `minutes`     | How long it took to make the recipe                                          |
| `submitted`   | When the recipe was added to the website                                     |
| `tags`        | Descriptors about the recipe                                                 |
| `nutrition`   | Nutritional facts: calories, fat, sugar, sodium, protein, sat. fat, carbs    |
| `ingredients` | Ingredients needed for the recipe                                            |

### Interactions Dataset (731,927 rows, 5 columns)

| Column       | Description                                       |
|--------------|---------------------------------------------------|
| `recipe_id`  | ID of the recipe                                  |
| `date`       | When the user submitted their rating              |
| `rating`     | Rating given by the user                          |
| `review`     | Comment made by the user about the recipe         |

There are more ratings than recipes because some recipes have multiple ratings, and some have none.

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning
1. **Merged datasets**:
    
    ```py
    merged_data = recipes.merge(ratings, left_on='id', right_on='recipe_id', how='left')
    merged_data['rating'] = merged_data['rating'].replace(0.0, np.nan)
    ```
    - This kept all recipes, and treated `0.0` ratings as missing.

2. **Handled missing ratings**:
    - Replaced with `np.NaN` to avoid skewing averages.

3. **Created `avg_rating`**:
    - Shows the average rating per recipe.

4. **Converted nutrition strings to floats**:
    - Enabled health score computation.

5. **Created `health_rating` column**:
    - Based on tags categorized into "healthy", "medium healthy", and "unhealthy":

    ```py
    healthy_keywords = {'healthy-2', 'healthy', 'salads', 'chard', 'vegan', 'very-low-carbs', 'vegetarian', 'high-fiber', 'spinach', 'low-carb', 'low-sodium', 'low-calorie', 'vegetables', 'low-fat', 'low-saturated-fat'}
    midhealthy_keywords = {'high-protein', 'pork-sausage', 'smoothies', 'desserts-fruit', 'low-in-something', 'pot-pie', 'dairy-free', 'gluten-free', 'casseroles', 'tex-mex'}
    unhealthy_keywords = {'drop-cookies', 'desserts', 'super-bowl', 'brownies', 'cakes', 'cake-fillings-and-frostings', 'fudge', 'rolled-cookies', 'cookies-and-brownies', 'cupcakes', 'desserts-easy', 'pies-and-tarts', 'sugar-cookies', 'fillings-and-frostings-chocolate', 'chocolate-chip-cookies', 'ice-cream'}
    ```
**Computed `health_score`**:
    - Weighted nutritional values to quantify health:

    | Nutritional Fact         | Impact |
    | ------------------------ | -------|
    | calories (#)             | 0.15   |
    | total fat (PDV)          | 0.15   |
    | sugar (PDV)              | 0.3    |
    | sodium (PDV)             | 0.1    |
    | protein (PDV)            | -0.2   |
    | saturated fat (PDV)      | 0.25   |
    | carbohydrates (PDV)      | 0.1    |

    - Protein gets a negative weight due to its generally positive health impact.

7. **Grouped by recipe name**:
    - Prevented duplicate recipe names from skewing the results.
  
---

Our final DataFrame has 83,628 rows and 20 columns. Here's a sample row:

| name | id | tags | nutrition| n_ingredients | rating | avg_rating | difficulty | health_rating | health_score |
|------|----|------|----------|---------------|--------|------------|------------|---------------|--------------|
| 0 carb 0 cal gummy worms | 45 | [...] | [384.7, 0.0, 0.0, ...]| 3 | 5 | 4.75 | intermediate | unhealthy | 33.5 |

---

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

