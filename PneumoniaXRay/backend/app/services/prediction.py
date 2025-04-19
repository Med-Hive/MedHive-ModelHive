import pickle
import numpy as np
import logging
import cv2
import asyncio
from typing import Dict, Tuple, Optional, Any
from app.core.config import settings
from app.core.exceptions import ModelLoadError, PredictionError, FeatureError

# Setup logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelService:
    _model = None
    _metadata = None

    @classmethod
    def load_model(cls):
        if cls._model is None:
            try:
                logger.info(f"Loading model from {settings.MODEL_PATH}")
                with open(settings.MODEL_PATH, "rb") as f:
                    cls._model = pickle.load(f)
                logger.info("Model loaded successfully")
            except FileNotFoundError as e:
                error_msg = f"Model file not found at {settings.MODEL_PATH}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg)
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg)

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls.load_model()
        return cls._model


    @classmethod
    def predict(cls, input_data: Dict[str, float]) -> Tuple[int, float]:
        try:
            model = cls.get_model()
            features = cls.prepare_features(input_data)
            
            try:
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0][1]
                logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
                return int(prediction), float(probability)
            except Exception as e:
                error_msg = f"Error during model prediction: {str(e)}"
                logger.error(error_msg)
                raise PredictionError(error_msg)
        
        except (ModelLoadError, FeatureError):
            raise
        except Exception as e:
            error_msg = f"Unexpected error during prediction: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)

    @classmethod
    def preprocess_image(cls, image_path: str) -> np.ndarray:
        """
        Preprocess an image for model prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array ready for model input
        """
        try:
            img_size = settings.IMG_SIZE
            # Load image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                error_msg = f"Failed to load image from {image_path}"
                logger.error(error_msg)
                raise PredictionError(error_msg)
                
            # Resize to expected input size
            img_resized = cv2.resize(img, (img_size, img_size))
            
            # Normalize pixel values
            img_normalized = img_resized / 255.0
            
            # Reshape to match model input shape (batch_size, height, width, channels)
            img_ready = img_normalized.reshape(1, img_size, img_size, 1)
            
            return img_ready
            
        except Exception as e:
            error_msg = f"Error preprocessing image: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)

    @classmethod
    async def predict_from_image(cls, image_path: str) -> Tuple[int, float]:
        """
        Predict pneumonia from chest X-ray image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (prediction, probability)
        """
        try:
            # Get the model
            model = cls.get_model()
            
            # Use a thread pool for CPU-intensive image preprocessing
            loop = asyncio.get_running_loop()
            processed_img = await loop.run_in_executor(
                None, cls.preprocess_image, image_path
            )
            
            # Make prediction
            raw_prediction = model.predict(processed_img)[0][0]
            logger.info(f"Raw prediction: {raw_prediction}")
            
            # Interpret results (based on your model's output format)
            prediction = 1 if raw_prediction > 0.5 else 0
            probability = float(raw_prediction) if prediction == 1 else float(1 - raw_prediction)
            
            logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
            return prediction, probability
            
        except ModelLoadError:
            raise
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)

    @classmethod
    def is_ready(cls) -> bool:
        return cls._model is not None and cls._metadata is not None