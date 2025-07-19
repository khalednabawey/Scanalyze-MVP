import os
import kagglehub
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras import backend as K


def load_model_from_kaggle(user: str, model_slug: str, variation_slug: str, filename: str, framework: str = "keras"):
    """Load model from Kaggle Hub with caching and error handling."""
    key = f"{user}/{model_slug}/{variation_slug}/{filename}"

    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN backend

        K.clear_session()
        model_handle = f"{user}/{model_slug}/{framework}/{variation_slug}"
        MODEL_PATH = kagglehub.model_download(model_handle)

        model_file = os.path.join(MODEL_PATH, filename)
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at {model_file}")

        model = load_model(model_file, compile=False)
        print(f"Model loaded successfully from {model_file}")
        return model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def preprocess_image(img):
    """Preprocesses the uploaded image for ResNet50."""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # Normalize using ResNet50's preprocessing
    img_array = preprocess_input(img_array)
    return img_array
