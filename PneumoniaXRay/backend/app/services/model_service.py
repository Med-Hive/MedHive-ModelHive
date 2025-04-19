import logging
import joblib
import numpy as np
from typing import Dict, Any
from app.core.exceptions import ModelLoadError, PredictionError, FeatureError
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self) -> None:
        try:
            logger.info(f"Loading model from {settings.MODEL_PATH}")
            self.model = joblib.load(settings.MODEL_PATH)
            logger.info("Model loaded successfully")
            
            logger.info(f"Loading metadata from {settings.METADATA_PATH}")
            self.metadata = joblib.load(settings.METADATA_PATH)
            self.feature_names = self.metadata.get('feature_names', [])
            logger.info("Metadata loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(detail=error_msg)
    
    def _validate_features(self, features: Dict[str, float]) -> None:
        if not self.feature_names:
            error_msg = "Feature names not available in metadata"
            logger.error(error_msg)
            raise FeatureError(detail=error_msg)
            
        missing_features = set(self.feature_names) - set(features.keys())
        if missing_features:
            error_msg = f"Missing required features: {missing_features}"
            logger.error(error_msg)
            raise FeatureError(detail=error_msg)
            
        extra_features = set(features.keys()) - set(self.feature_names)
        if extra_features:
            error_msg = f"Unexpected features provided: {extra_features}"
            logger.error(error_msg)
            raise FeatureError(detail=error_msg)
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        try:
            logger.info("Validating input features")
            self._validate_features(features)
            
            # Convert features to numpy array in correct order
            feature_values = np.array([features[feature] for feature in self.feature_names])
            feature_values = feature_values.reshape(1, -1)
            
            logger.info("Making prediction")
            prediction = self.model.predict(feature_values)[0]
            probability = self.model.predict_proba(feature_values)[0][1]
            
            result = {
                "prediction": int(prediction),
                "probability": float(probability),
                "feature_importance": self._get_feature_importance(features)
            }
            
            logger.info(f"Prediction successful: {result}")
            return result
            
        except FeatureError:
            raise
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(detail=error_msg)
    
    def _get_feature_importance(self, features: Dict[str, float]) -> Dict[str, float]:
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = dict(zip(self.feature_names, self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                importance = dict(zip(self.feature_names, self.model.coef_[0]))
            else:
                importance = {feature: 0.0 for feature in self.feature_names}
            return importance
        except Exception as e:
            logger.warning(f"Failed to calculate feature importance: {str(e)}")
            return {feature: 0.0 for feature in self.feature_names} 