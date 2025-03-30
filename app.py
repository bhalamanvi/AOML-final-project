import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import pickle
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import re
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Nutrition Recipe Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .nutrition-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .summary-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e1e4e8;
    }
    .recipe-card {
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: white;
    }
    .recipe-meta {
        color: #666;
        font-size: 0.9em;
        margin-bottom: 10px;
    }
    .difficulty-easy {
        color: #28a745;
    }
    .difficulty-medium {
        color: #ffc107;
    }
    .difficulty-hard {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CUISINES = [
    "All", "American", "Italian", "Asian", "Mexican", "Mediterranean", 
    "Indian", "French", "Greek", "Japanese", "Thai", "Chinese"
]

MEAL_TYPES = [
    "All", "Breakfast", "Lunch", "Dinner", "Snack", "Dessert"
]

DIETS = [
    "None", "Vegetarian", "Vegan", "Gluten Free", "Ketogenic", 
    "Paleo", "Low FODMAP", "Whole30"
]

# Load models
@st.cache_resource
def load_models():
    try:
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/knn_model.pkl', 'rb') as f:
            knn_model = pickle.load(f)
        encoder_model = tf.keras.models.load_model('models/encoder_model.h5')
        return scaler, knn_model, encoder_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please ensure all model files are present and not corrupted.")
        st.stop()

# Function to load saved recipes
@st.cache_data
def load_saved_recipes():
    try:
        with open('saved_recipes.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Function to save recipe
def save_recipe(recipe):
    saved_recipes = load_saved_recipes()
    recipe['saved_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saved_recipes.append(recipe)
    with open('saved_recipes.json', 'w') as f:
        json.dump(saved_recipes, f)
    st.cache_data.clear()

# Function to fetch recipes from Spoonacular
def fetch_recipes(api_key, target_calories, target_protein, target_fat, target_carbs, cuisine="", meal_type="", diet=""):
    base_url = "https://api.spoonacular.com/recipes/complexSearch"
    
    params = {
        "apiKey": api_key,
        "number": 100,
        "addRecipeNutrition": True,
        "instructionsRequired": True,
        "fillIngredients": True,
        "sort": "random"
    }
    
    # Add filters if selected
    if cuisine and cuisine != "All":
        params["cuisine"] = cuisine
    if meal_type and meal_type != "All":
        params["type"] = meal_type
    if diet and diet != "None":
        params["diet"] = diet.lower().replace(" ", "")
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if not results:
                params["instructionsRequired"] = False
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    return response.json().get('results', [])
            return results
    except Exception as e:
        st.warning(f"API call issue: {str(e)}. Showing all available recipes.")
    return []

# Function to extract nutrition values
def extract_nutrition_values(nutrition):
    if not nutrition or not isinstance(nutrition, dict):
        return {
            'calories': 0,
            'protein': 0,
            'fat': 0,
            'carbs': 0
        }
    
    # Extract nutrients from the nutrients list
    nutrients = nutrition.get('nutrients', [])
    nutrition_values = {
        'calories': 0,
        'protein': 0,
        'fat': 0,
        'carbs': 0
    }
    
    for nutrient in nutrients:
        if nutrient.get('name', '').lower() == 'calories':
            nutrition_values['calories'] = nutrient.get('amount', 0)
        elif nutrient.get('name', '').lower() == 'protein':
            nutrition_values['protein'] = nutrient.get('amount', 0)
        elif nutrient.get('name', '').lower() == 'fat':
            nutrition_values['fat'] = nutrient.get('amount', 0)
        elif nutrient.get('name', '').lower() == 'carbohydrates':
            nutrition_values['carbs'] = nutrient.get('amount', 0)
    
    return nutrition_values

# Function to preprocess recipe data
def preprocess_recipe_data(recipes, target_calories):
    nutrition_data = []
    valid_recipes = []
    
    # First pass: collect all recipes with valid nutrition data
    for recipe in recipes:
        if 'nutrition' in recipe:
            nutrition_values = extract_nutrition_values(recipe['nutrition'])
            if nutrition_values['calories'] > 0:  # Only include recipes with valid calorie information
                nutrition_data.append(nutrition_values)
                valid_recipes.append(recipe)
    
    if not valid_recipes:
        return pd.DataFrame(), []
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(nutrition_data)
    
    # If we have more than 5 recipes, try to filter by calorie range
    if len(valid_recipes) > 5:
        # Calculate calorie ranges
        min_cals = max(0, target_calories - 500)  # Wider range
        max_cals = target_calories + 500
        
        # Create mask for calories within range
        cal_mask = (df['calories'] >= min_cals) & (df['calories'] <= max_cals)
        
        # If we have at least 5 recipes within calorie range, filter
        if cal_mask.sum() >= 5:
            df = df[cal_mask]
            valid_recipes = [r for i, r in enumerate(valid_recipes) if cal_mask.iloc[i]]
    
    return df, valid_recipes

# Function to calculate weighted nutritional difference
def calculate_nutritional_difference(target_values, recipe_values):
    # Weights for different nutritional components
    weights = np.array([0.4, 0.4, 0.1, 0.1])  # Higher weights for calories and protein
    differences = np.abs(target_values - recipe_values)
    return np.sum(weights * differences)

# Function to clean HTML from text
def clean_html(text):
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters and extra whitespace
    text = re.sub(r'&[^;]+;', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to format nutritional information
def format_nutrition(nutrition):
    nutrition_values = extract_nutrition_values(nutrition)
    return {
        'calories': round(nutrition_values['calories']),
        'protein': round(nutrition_values['protein'], 1),
        'fat': round(nutrition_values['fat'], 1),
        'carbs': round(nutrition_values['carbs'], 1)
    }

def estimate_difficulty(recipe):
    if 'readyInMinutes' not in recipe:
        return "N/A"
    
    time = recipe['readyInMinutes']
    ingredients_count = len(recipe.get('extendedIngredients', []))
    
    if time <= 30 and ingredients_count <= 7:
        return "Easy"
    elif time >= 60 or ingredients_count >= 15:
        return "Hard"
    else:
        return "Medium"

def main():
    st.title("🍽️ Nutrition-Based Recipe Recommender")
    
    # Sidebar for filters
    with st.sidebar:
        st.header("Filters")
        cuisine = st.selectbox("Cuisine Type", CUISINES)
        meal_type = st.selectbox("Meal Type", MEAL_TYPES)
        diet = st.selectbox("Dietary Restrictions", DIETS)
        servings = st.number_input("Number of Servings", min_value=1, max_value=10, value=4)
        
        st.header("Saved Recipes")
        if st.button("View Saved Recipes"):
            saved_recipes = load_saved_recipes()
            if saved_recipes:
                for recipe in saved_recipes:
                    st.write(f"• {recipe['title']} (Saved on {recipe['saved_date']})")
            else:
                st.write("No saved recipes yet!")

    # Main content
    st.write("Enter your target nutritional values to find matching recipes!")

    # Load models
    scaler, knn_model, encoder_model = load_models()

    # Create input fields in a more compact layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        target_calories = st.number_input("Target Calories", min_value=0, value=2000)
    with col2:
        target_protein = st.number_input("Target Protein (g)", min_value=0, value=50)
    with col3:
        target_fat = st.number_input("Target Fat (g)", min_value=0, value=70)
    with col4:
        target_carbs = st.number_input("Target Carbs (g)", min_value=0, value=250)

    if st.button("Find Recipes", type="primary"):
        with st.spinner("Searching for recipes..."):
            recipes = fetch_recipes(
                os.getenv('SPOONACULAR_API_KEY'),
                target_calories,
                target_protein,
                target_fat,
                target_carbs,
                cuisine,
                meal_type,
                diet
            )

            if not recipes:
                st.error("API error. Please try again.")
                return

            nutrition_df, valid_recipes = preprocess_recipe_data(recipes, target_calories)
            
            if nutrition_df.empty:
                st.warning("No recipes found with valid nutritional information. Retrying with different criteria...")
                recipes = fetch_recipes(
                    os.getenv('SPOONACULAR_API_KEY'),
                    target_calories,
                    target_protein,
                    target_fat,
                    target_carbs,
                    cuisine,
                    meal_type,
                    diet
                )
                nutrition_df, valid_recipes = preprocess_recipe_data(recipes, target_calories)
                if nutrition_df.empty:
                    st.error("Still no recipes found. Please try again.")
                    return
            
            target_values = np.array([[target_calories, target_protein, target_fat, target_carbs]])
            scaled_target = scaler.transform(target_values)
            scaled_recipes = scaler.transform(nutrition_df)
            
            differences = []
            for i, recipe_values in enumerate(scaled_recipes):
                diff = calculate_nutritional_difference(scaled_target[0], recipe_values)
                differences.append((diff, i))
            
            differences.sort(key=lambda x: x[0])
            top_indices = [idx for _, idx in differences[:min(5, len(differences))]]
            
            st.subheader(f"Top {len(top_indices)} Matching Recipes")
            
            for idx in top_indices:
                recipe = valid_recipes[idx]
                with st.container():
                    st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                    
                    # Recipe header with title and meta information
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if recipe.get('image'):
                            try:
                                response = requests.get(recipe['image'])
                                img = Image.open(BytesIO(response.content))
                                st.image(img, use_column_width=True)
                            except Exception as e:
                                st.warning("Could not load recipe image")
                    
                    with col2:
                        st.markdown(f"### {recipe['title']}")
                        
                        # Meta information
                        difficulty = estimate_difficulty(recipe)
                        difficulty_class = f"difficulty-{difficulty.lower()}"
                        meta_html = f"""
                        <div class="recipe-meta">
                            <span>⏱️ {recipe.get('readyInMinutes', 'N/A')} minutes</span> | 
                            <span>👥 {recipe.get('servings', 'N/A')} servings</span> | 
                            <span class="{difficulty_class}">📊 {difficulty} difficulty</span>
                        </div>
                        """
                        st.markdown(meta_html, unsafe_allow_html=True)
                        
                        # Save recipe button
                        if st.button(f"Save Recipe: {recipe['title']}", key=f"save_{recipe['id']}"):
                            save_recipe(recipe)
                            st.success("Recipe saved!")
                        
                        # Nutritional information
                        if 'nutrition' in recipe:
                            nutrition = format_nutrition(recipe['nutrition'])
                            st.markdown(f"""
                            <div class="nutrition-box">
                            <h4>Nutritional Information (per serving):</h4>
                            <p>• Calories: {nutrition['calories']} kcal</p>
                            <p>• Protein: {nutrition['protein']}g</p>
                            <p>• Fat: {nutrition['fat']}g</p>
                            <p>• Carbs: {nutrition['carbs']}g</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Recipe summary
                        if 'summary' in recipe:
                            clean_summary = clean_html(recipe['summary'])
                            st.markdown(f"""
                            <div class="summary-box">
                            <h4>Recipe Summary:</h4>
                            <p>{clean_summary}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Instructions (in an expander)
                        if 'analyzedInstructions' in recipe and recipe['analyzedInstructions']:
                            with st.expander("Cooking Instructions"):
                                instructions = recipe['analyzedInstructions'][0].get('steps', [])
                                for i, step in enumerate(instructions, 1):
                                    st.write(f"{i}. {step['step']}")
                        
                        # Ingredients (in an expander)
                        if 'extendedIngredients' in recipe:
                            with st.expander("Ingredients"):
                                ingredients = recipe['extendedIngredients']
                                for ingredient in ingredients:
                                    st.write(f"• {ingredient.get('original', '')}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 