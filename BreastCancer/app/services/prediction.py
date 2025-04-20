import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
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
    def load_metadata(cls):
        if cls._metadata is None:
            try:
                logger.info(f"Loading metadata from {settings.METADATA_PATH}")
                with open(settings.METADATA_PATH, "rb") as f:
                    cls._metadata = pickle.load(f)
                logger.info("Metadata loaded successfully")
            except FileNotFoundError:
                logger.warning(f"Metadata file not found at {settings.METADATA_PATH}")
                cls._metadata = {"features": []}
            except Exception as e:
                logger.warning(f"Error loading metadata: {str(e)}")
                cls._metadata = {"features": []}

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls.load_model()
        return cls._model

    @classmethod
    def get_metadata(cls):
        if cls._metadata is None:
            cls.load_metadata()
        return cls._metadata

    @classmethod
    def prepare_features(cls, input_data: Dict[str, float]) -> pd.DataFrame:
        try:
            metadata = cls.get_metadata()
            all_features = metadata.get("features", [])

            if not all_features:
                logger.warning("No feature metadata available. Using only provided features.")
                df = pd.DataFrame([input_data])
            else:
                # Create feature mapping for variations in feature names
                feature_mapping = {
                    "concave_points_mean": "concave points_mean",
                    "concave_points_se": "concave points_se",
                    "concave_points_worst": "concave points_worst"
                }
                
                # Initialize feature dictionary with all features set to 0.0
                feature_dict = {feature: 0.0 for feature in all_features}
                
                # Map input features to model features
                for input_feature, value in input_data.items():
                    # Check if the feature needs to be mapped
                    mapped_feature = feature_mapping.get(input_feature, input_feature)
                    if mapped_feature in all_features:
                        feature_dict[mapped_feature] = value
                    else:
                        error_msg = f"Feature {input_feature} not found in model features"
                        logger.error(error_msg)
                        raise FeatureError(error_msg)

                df = pd.DataFrame([feature_dict])
                logger.debug(f"Prepared features with shape: {df.shape}")
                logger.debug(f"Feature names: {df.columns.tolist()}")

            return df
        
        except FeatureError:
            raise
        except Exception as e:
            error_msg = f"Error preparing features: {str(e)}"
            logger.error(error_msg)
            raise FeatureError(error_msg)

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
    def is_ready(cls) -> bool:
        return cls._model is not None and cls._metadata is not None