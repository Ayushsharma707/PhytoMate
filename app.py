# Import necessary libraries
import streamlit as st  # Streamlit is used for creating the web interface of the application.
import pandas as pd  # Pandas is used for handling and processing data, such as CSV files or tabular data.
from PIL import Image  # PIL (Pillow) is used for image processing, such as loading and displaying images.
import tensorflow as tf  # TensorFlow is used for creating and working with the machine learning models.
import numpy as np  # NumPy is used for numerical computations, such as handling arrays and matrix operations.
import joblib  # Joblib is used for saving and loading machine learning models and data pipelines efficiently.
import google.generativeai as genai  # Google Generative AI (Gemini) is used for AI-based recommendations and generative functionalities.
from sklearn.exceptions import NotFittedError  # Exception handling for sklearn models that have not been trained/fitted.
import re  # Regular expressions (regex) are used for pattern matching and string manipulation.
import os
from dotenv import load_dotenv
# Load environment variables from .env file
st.set_page_config(page_title="PhytoMate", layout="wide")


load_dotenv()


if "API_KEY" not in st.secrets:
    st.error("🚫 API key not found in Streamlit Secrets. Please add it in your app settings.")
    st.stop()

api_key = st.secrets["API_KEY"]

genai.configure(api_key=api_key)

model_path = "Plant_Disease_Dataset/trained_plant_disease_model.keras"

# Check if the file exists before loading
if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
    try:
        model = tf.keras.models.load_model(model_path)
        
    except Exception as e:
        st.error(f"Error loading plant disease model: {e}")
else:
    st.error("Model file not found or is empty! Check the file path and retrain if necessary.")

def get_crop_description_from_gemini(crop_name):
    try:
        # Construct the prompt for the crop description
        prompt = f"Provide a detailed description of the crop '{crop_name}' grown in India, including its uses, benefits, and ideal growing conditions."
        
        # Use the generative model to generate the crop description
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        return response.text  # Return the generated description

    except Exception as e:
        return f"Error fetching crop description: {e}"

def model_prediction(test_image):
    # Preprocess the input image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    
    # Make prediction
    predictions = model.predict(input_arr)
    
    # Return index of the class with the highest probability
    return np.argmax(predictions)

# Function to get solution from Gemini AI
def get_solution_from_gemini(disease_name):
    try:
        # Construct the prompt for treatment recommendation
        prompt = f"Provide a treatment for the plant disease called '{disease_name}'"
        
        # Use the generative model to generate the solution
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        return response.text  # Return the generated solution

    except Exception as e:
        return f"Error fetching solution: {e}"

# Set up page configuration
st.markdown(""" 
    <style>
    .title { font-size: 3rem; color: #4CAF50; text-align: center; font-family: 'Segoe UI'; }
    .sub-header { font-size: 1.2rem; color: #333; font-family: 'Segoe UI'; }
    .footer { text-align: center; font-size: 1rem; margin-top: 40px; color: #888; }
    .btn { background-color: #4CAF50; color: white; font-size: 1.1rem; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
    .btn:hover { background-color: #45a049; }
    </style>
""", unsafe_allow_html=True)

# Sidebar with three dropdowns
with st.sidebar:
    option = st.selectbox(
        'Select an option',
        ['🏠 Home', 'ℹ️ About', '🔍 Features']
    )

if option == "🏠 Home":
    # Main project title with increased size
    st.markdown('<h1 style="font-size: 4rem; color: #4CAF50;  font-family: \'Segoe UI\';">PhytoMate 🌿</h1>', unsafe_allow_html=True)
    
    # Subheading
    st.subheader("Plant Disease Recognition & Crop Recommendation System ")

    st.image("Climate-Smart-Agriculture-for-a-Sustainable-Future-1024x631.png", width=700)  # Adjust width as needed
    
    st.markdown(""" 
    <style>
        .big-font {
            font-size:18px !important;
        }
        .highlight {
            color: green;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">Welcome to <span class="highlight">PhytoMate</span>! Upload a plant image to analyze for potential diseases and get tailored crop recommendations based on soil and environmental conditions.</p>', unsafe_allow_html=True)

    st.markdown("""
    ### How It Works
    1. **Plant Disease Recognition**: Upload an image of a plant to detect if it has any diseases.
    2. **Crop Recommendation**: Input your soil and environmental conditions to get recommendations on the best crops to plant in your area.
    3. **Analysis**: The system uses machine learning to provide accurate predictions and recommendations for plant care and crop selection.
    4. **Results**: Get disease detection results and crop recommendations with actionable solutions.

    ### Why Choose Us?
    - **Accurate**: Powered by advanced machine learning models for both disease recognition and crop recommendations.
    - **Fast**: Receive instant results to make timely and informed decisions for plant care and farming.
    - **User-Friendly**: Simple, intuitive interface suitable for anyone, from farmers to hobbyists.
    - **Actionable**: Get practical, easy-to-understand solutions to address plant diseases and optimize crop selection for your environment.
    """)

elif option == 'ℹ️ About':
    st.header("About the Project")
    st.markdown("""
    This project contains two main features:

    1. **Plant Disease Detection**: This feature helps in detecting plant diseases from uploaded images using a pre-trained deep learning model. It identifies diseases and suggests solutions to help in plant care.

    2. **Crop Recommendation**: Based on the soil and environmental conditions, this feature recommends the most suitable crops to plant. It uses a machine learning model trained on agricultural data to provide tailored crop recommendations.

    ### About the Features
    - **Plant Disease Detection**:
      - Upload an image of a plant, and the system will predict if the plant is infected with a disease.
      - The app also provides a treatment solution based on the predicted disease.

    - **Crop Recommendation**:
      - Input soil properties like nitrogen, phosphorus, potassium, temperature, humidity, and pH levels.
      - Get recommendations on the best crops to grow in your region.

    ### Team Members
    - <span style="color: white;">**Aditya Gupta**</span>
    - <span style="color: white;">**Ayush Sharma**</span>
    - <span style="color: white;">**Aryan Gupta**</span>
    - <span style="color: white;">**Aryan Singh**</span>
    """, unsafe_allow_html=True)

elif option == '🔍 Features':
    st.header("Select Feature")
    feature_option = st.selectbox(
        'Choose a feature:',
        ['Plant Disease Detection', 'Crop Recommendation']
    )
    
    if feature_option == 'Plant Disease Detection':
        st.markdown("""<div class="header" style="color: green; font-weight: bold; font-size: 36px;">Plant Disease Recognition</div>""", unsafe_allow_html=True)


        # File uploader
        st.markdown('<div class="sub-header" style="color: white;">Upload a plant image to identify its disease:</div>', unsafe_allow_html=True)
        test_image = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "webp"])

        # Store the uploaded image in session_state
        if test_image:
            st.session_state.image = test_image

        # Initialize 'predicted_disease' and 'solution' in session_state
        if 'predicted_disease' not in st.session_state:
            st.session_state.predicted_disease = None
        if 'solution' not in st.session_state:
            st.session_state.solution = None

        # Toggle image visibility
        if 'show_image' not in st.session_state:
            st.session_state.show_image = True

        toggle_image_button = st.button("Image", key="toggle_image")
        if toggle_image_button:
            st.session_state.show_image = not st.session_state.show_image

        # Display the uploaded image
        if 'image' in st.session_state and st.session_state.show_image:
            st.image(st.session_state.image, caption="Uploaded Image", width=300)

        # Predict button logic
        st.markdown('<div class="sub-header" style="color: white;">Ready to predict? Click the button below:</div>', unsafe_allow_html=True)
        if st.button("Predict", key="predict", help="Click to predict disease"):
            if 'image' in st.session_state:
                st.snow()  # Visual effect
                result_index = model_prediction(st.session_state.image)

                # Class names for the model
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]

                st.session_state.predicted_disease = class_name[result_index]
                st.success(f"🌱 Prediction: This looks like **{st.session_state.predicted_disease}**")
            else:
                st.error("Please upload an image to make a prediction.")

        # Fetch solution button logic
        if st.session_state.predicted_disease:
            if st.button("Get Solution", key="get_solution", help="Fetch treatment solution"):
                with st.spinner('Fetching recommended solution...'):
                    st.session_state.solution = get_solution_from_gemini(st.session_state.predicted_disease)

        # Display the fetched solution
        if st.session_state.solution:
            st.write(f"💡 **Recommended Solution:** {st.session_state.solution}")




    elif feature_option == 'Crop Recommendation':
        # Clear any existing TensorFlow sessions
        tf.keras.backend.clear_session()

        # Load the plant disease model
        model_path = "crop-recommendation-system-main/Model/knn.pkl"

        if not os.path.exists(model_path):
            print("Error: Model file not found!")
        else:
            try:
                model = tf.keras.models.load_model(model_path)
                print("Model loaded successfully!")
            except Exception as e:
                print("Error loading model:", str(e))

        # Cache data and model loading for efficiency
        @st.cache_data
        def load_data():
            try:
                crop_desc = pd.read_csv('crop-recommendation-system-main/Dataset/Crop_Desc.csv', sep=';', encoding='utf-8')
                crop_recommendation = pd.read_csv('crop-recommendation-system-main/Dataset/Crop_recommendation.csv')
                return crop_desc, crop_recommendation
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return None, None

        @st.cache_resource
        def load_model():
            try:
                return joblib.load('crop-recommendation-system-main/Model/knn.pkl')
            except Exception as e:
                st.error(f"Error loading recommendation model: {e}")
                return None

        # Extract image URL from HTML in CSV
        def extract_image_url(html_string):
            match = re.search(r'src="([^"]+)"', html_string)
            return match.group(1) if match else None

        # Load the data and model
        df_desc, df = load_data()
        rdf_clf = load_model()

        # Ensure data and model are loaded successfully
        if df is None or rdf_clf is None:
            st.stop()

        # Add image URLs to description dataframe
        df_desc['image_url'] = df_desc['image'].apply(extract_image_url)

        # Feature input instructions
        st.markdown("<h3 style='text-align: center;'>Enter the following details to recommend the best crop.</h3><br>", unsafe_allow_html=True)

        # User Inputs for Crop Recommendation
        n_input = st.number_input('Nitrogen (kg/ha):', min_value=0, max_value=140, value=50)
        p_input = st.number_input('Phosphorus (kg/ha):', min_value=5, max_value=145, value=60)
        k_input = st.number_input('Potassium (kg/ha):', min_value=5, max_value=205, value=50)
        temp_input = st.number_input('Temperature (ºC):', min_value=9.0, max_value=43.0, value=25.0, step=1.0)
        hum_input = st.number_input('Humidity (%):', min_value=15.0, max_value=99.0, value=70.0, step=1.0)
        ph_input = st.number_input('Soil pH:', min_value=3.6, max_value=9.9, value=6.5, step=0.1)
        rain_input = st.number_input('Rainfall (mm):', min_value=21.0, max_value=298.0, value=100.0, step=0.1)
        location = st.selectbox(
            'Select location:',
            ['Central India', 'Eastern India', 'North Eastern India', 'Northern India', 'Western India', 'Other']
        )

        # Convert location to encoded format
        location_dict = {
            'Central India': [1, 0, 0, 0, 0, 0],
            'Eastern India': [0, 1, 0, 0, 0, 0],
            'North Eastern India': [0, 0, 1, 0, 0, 0],
            'Northern India': [0, 0, 0, 1, 0, 0],
            'Western India': [0, 0, 0, 0, 1, 0],
            'Other': [0, 0, 0, 0, 0, 1]
        }
        predict_inputs = [[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input] + location_dict[location]]

        # Predict Crop
        if st.button('Recommend Crop'):
            try:
                # Perform prediction
                rdf_predicted_value = rdf_clf.predict(predict_inputs)
                recommended_crop = rdf_predicted_value[0]
                st.success(f"Best crop to plant: {recommended_crop}")

                # Fetch crop image URL
                crop_row = df_desc[df_desc['label'].str.strip().str.lower() == recommended_crop.lower()]
                if not crop_row.empty:
                    crop_name = crop_row.iloc[0]['label']
                    image_url = crop_row.iloc[0]['image_url']
                    if image_url:
                        st.image(image_url, caption=f"Recommended Crop: {crop_name}", use_column_width=False, width=300)
                    else:
                        st.warning(f"No image URL available for {crop_name}.")
                else:
                    st.warning(f"No description or image available for {recommended_crop}.")

                 # Fetch crop description using Gemini API
                st.markdown("**Fetching crop description...**")
                crop_description = get_crop_description_from_gemini(recommended_crop)
                st.write(f"📋 **Crop Description:** {crop_description}")       

            except NotFittedError:
                st.error("The recommendation model is not properly fitted. Please ensure the model is trained and reloaded.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                

