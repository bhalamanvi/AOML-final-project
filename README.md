# Nutrition-Based Recipe Recommender System

## Problem Statement
Develop an intelligent recipe recommendation system that suggests recipes based on user-specified nutritional requirements. The system should:
- Accept target values for calories, protein, fat, and carbohydrates
- Find recipes that closely match these nutritional requirements
- Consider dietary preferences and restrictions
- Provide detailed recipe information including ingredients and instructions
- Prioritize matching calories and protein content while being flexible with fats and carbs

## Dataset Source
The system uses the Spoonacular API (https://spoonacular.com/food-api) which provides:
- Over 5000+ recipes with detailed nutritional information
- Recipe details including:
  - Full ingredient lists with amounts
  - Step-by-step cooking instructions
  - Cooking time and servings
  - Cuisine types and dietary categories
  - High-quality recipe images
  - Comprehensive nutritional analysis

## Solution Approach

### 1. Machine Learning Pipeline
- **Data Preprocessing**: 
  - MinMaxScaler for normalizing nutritional values
  - Ensures all nutritional components contribute proportionally
  - Stored as `models/scaler.pkl`

- **Feature Engineering**:
  - Weighted importance for nutritional components
  - Calories (40%), Protein (40%), Fat (10%), Carbs (10%)
  - Prioritizes matching calories and protein content

- **Similarity Matching**:
  - K-Nearest Neighbors (KNN) algorithm
  - Finds 5 closest matches to target nutritional values
  - Model stored as `models/knn_model.pkl`

- **Dimensionality Reduction**:
  - Autoencoder neural network for encoding nutritional profiles
  - Helps in better similarity matching
  - Model stored as `models/encoder_model.h5`

### 2. Technical Implementation
- **Framework**: Streamlit for interactive web interface
- **API Integration**: Spoonacular API for recipe data
- **Backend**: Python with scikit-learn and TensorFlow
- **Data Storage**: Local JSON for saved recipes

### 3. Features
- **Nutritional Target Input**:
  - Calories (kcal)
  - Protein (g)
  - Fat (g)
  - Carbohydrates (g)

- **Filtering Options**:
  - Cuisine Type (American, Italian, Asian, etc.)
  - Meal Type (Breakfast, Lunch, Dinner, etc.)
  - Dietary Restrictions (Vegetarian, Vegan, Gluten-Free, etc.)
  - Serving Size

- **Recipe Display**:
  - Recipe title and image
  - Nutritional information per serving
  - Cooking time and difficulty
  - Ingredients list
  - Step-by-step instructions
  - Recipe summary
  - Save recipe functionality

## Installation & Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with your Spoonacular API key:
```
SPOONACULAR_API_KEY=your_api_key_here
```

3. Run the application:
```bash
python -m streamlit run app.py
```

## Model Details

### MinMaxScaler
- Purpose: Normalize nutritional values to [0,1] range
- Features scaled: calories, protein, fat, carbohydrates
- Ensures balanced contribution from all nutritional components

### KNN Model
- Algorithm: K-Nearest Neighbors
- n_neighbors: 5
- Distance metric: Euclidean
- Purpose: Find recipes with similar nutritional profiles

### Autoencoder
- Architecture: Dense neural network
- Input dimension: 4 (nutritional values)
- Encoding dimension: 2
- Purpose: Dimensionality reduction for better similarity matching

## Usage Example

1. Enter target nutritional values:
   - Calories: 2000 kcal
   - Protein: 50g
   - Fat: 70g
   - Carbs: 250g

2. Select filters (optional):
   - Cuisine: Italian
   - Meal Type: Dinner
   - Diet: None
   - Servings: 4

3. Click "Find Recipes" to get top 5 matching recipes

4. View detailed recipe information and save favorites

## Future Improvements
1. Implement user profiles and personalized recommendations
2. Add meal planning functionality
3. Include more advanced filtering options
4. Add recipe rating and review system
5. Implement nutritional goal tracking

## Dependencies
- streamlit==1.32.0
- pandas==2.2.1
- numpy==1.26.4
- scikit-learn==1.3.2
- tensorflow==2.16.1
- requests==2.31.0
- python-dotenv==1.0.1
- pillow==10.2.0

## Note
The system requires an active internet connection to fetch recipe data from the Spoonacular API. Make sure you have sufficient API credits for continuous usage. 



