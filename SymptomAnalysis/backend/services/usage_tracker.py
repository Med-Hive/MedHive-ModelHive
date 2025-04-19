from typing import Dict, Any
import httpx
from app.core.config import get_settings
import logging
from datetime import datetime

settings = get_settings()
logger = logging.getLogger(__name__)

class UsageTracker:
    def __init__(self):
        self.supabase_url = settings.SUPABASE_URL
        self.supabase_key = settings.SUPABASE_KEY
        
    async def track_analysis(self, user_id: str, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> None:
        """
        Track a symptom analysis request for billing and analytics
        """
        try:
            usage_data = {
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "request_type": "symptom_analysis",
                "tokens_used": len(str(request_data)) + len(str(response_data)),  # Simple approximation
                "model_used": settings.GROQ_MODEL_NAME,
                "metadata": {
                    "symptoms_count": len(request_data.get("symptoms", "").split(",")),
                    "has_additional_context": bool(request_data.get("additional_context")),
                    "diseases_analyzed": len(response_data.get("possible_diseases", [])),
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.supabase_url}/rest/v1/model_usage",
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Content-Type": "application/json"
                    },
                    json=usage_data
                )
                response.raise_for_status()
                
        except Exception as e:
            logger.error(f"Failed to track usage: {str(e)}")
            # Don't raise the exception to avoid affecting the main application flow
            pass

    async def get_user_usage(self, user_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get usage statistics for a user within a date range
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.supabase_url}/rest/v1/model_usage",
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                    },
                    params={
                        "user_id": user_id,
                        "timestamp.gte": start_date,
                        "timestamp.lte": end_date
                    }
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Failed to get user usage: {str(e)}")
            return {}