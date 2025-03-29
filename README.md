PROBLEM STATEMENT:
Most existing recipe platforms offer generic search and filtering options, lacking the ability to provide personalized meal suggestions based on detailed nutritional requirements. This project addresses the need for a data-driven solution that can intelligently recommend recipes tailored to individual nutritional profiles. By leveraging machine learning techniques on a comprehensive recipe dataset, the system aims to bridge the gap between user-specific dietary needs and the vast pool of available food options.

DATASET:
The dataset used for this project is sourced from Kaggle's Food.com Recipes Data, which contains structured information on over 5,20,000 recipes. Key attributes include Name, CookTime, PrepTime, TotalTime, RecipeIngredientParts, RecipeInstructions, and several nutritional attributes such as Calories, FatContent, SaturatedFatContent, SugarContent, SodiumContent, ProteinContent, and CarbohydrateContent. The dataset was cleaned to remove missing or malformed entries, and additional features such as Calories_per_protein, Fat_to_Carb_Ratio, and Sodium_per_calorie were engineered to provide more nuanced insights into nutritional composition.

SOLUTION:
To prepare the data, preprocessing steps included handling missing values using median imputation, winsorizing outliers in key nutritional fields, and converting cooking/preparation times from ISO 8601 to minutes. Ingredient and instruction texts were vectorized using TF-IDF with dimensionality reduction applied through PCA to handle high-dimensional features. Clustering was performed using KMeans to group similar recipes, with the optimal number of clusters determined via the elbow method and silhouette score analysis. A LightGBM classifier was then trained to predict cluster membership based on normalized nutritional values. The final prediction step involved using KNN to identify recipes most similar to a user’s predicted cluster, thereby enabling the delivery of tailored recommendations.



