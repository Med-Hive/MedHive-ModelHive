import groq
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings
from app.models.responses import Disease, SymptomAnalysisResponse
from astrapy.db import AstraDB
from .usage_tracker import UsageTracker
from .mlops_tracker import MLOpsTracker
from app.core.metrics import (
    track_latency, 
    SERVICE_LATENCY, 
    MODEL_TOKENS, 
    MODEL_ERRORS,
    SERVICE_STATUS
)
from app.core.model_monitor import ModelMonitor
import time
import uuid

settings = get_settings()

class ChatService:
    def __init__(self):
        self.groq_client = groq.Groq(api_key=settings.GROQ_API_KEY)
        self.model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
        self.astra_db = AstraDB(
            token=settings.ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=settings.ASTRA_DB_API_ENDPOINT
        )
        self.collection = self.astra_db.collection("disease_vectors")
        self.usage_tracker = UsageTracker()
        self.model_monitor = ModelMonitor()
        self.mlops_tracker = MLOpsTracker()
        self.current_model_version = str(uuid.uuid4())  # In production, get from model registry

    @track_latency(SERVICE_LATENCY, {'service': 'chat', 'operation': 'analyze_symptoms'})
    async def analyze_symptoms(self, symptoms: str, additional_context: str = None, user_id: Optional[str] = None) -> SymptomAnalysisResponse:
        start_time = time.time()
        try:
            # Generate embedding for the input symptoms
            symptoms_embedding = self.model.encode(symptoms).tolist()

            # Search for similar symptom patterns in AstraDB
            similar_cases = await self._search_similar_cases(symptoms_embedding)

            # Prepare context for LLM
            context = self._prepare_llm_context(similar_cases, symptoms, additional_context)

            # Get analysis from Groq
            completion = await self._get_groq_analysis(context)
            
            # Calculate latency for tracking
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Parse and validate the response
            try:
                analysis_dict = json.loads(completion)
                response = SymptomAnalysisResponse(**analysis_dict)
                
                # Track model tokens and performance
                tokens_used = len(str(context)) + len(completion)
                MODEL_TOKENS.labels(
                    model=settings.GROQ_MODEL_NAME,
                    operation='analyze_symptoms'
                ).inc(tokens_used)
                
                # Track prediction for monitoring
                self.model_monitor.track_prediction(
                    input_features=symptoms_embedding,
                    prediction=analysis_dict,
                    latency=latency,
                    ground_truth=None  # We don't have immediate feedback
                )
                
                # Track prediction in MLOps system
                await self.mlops_tracker.track_prediction(
                    input_text=symptoms,
                    input_embedding=symptoms_embedding,
                    prediction=analysis_dict,
                    confidence=max(d.confidence for d in response.possible_diseases) if response.possible_diseases else 0.0,
                    latency_ms=latency,
                    model_version_id=self.current_model_version
                )
                
                # Update model metrics
                await self.mlops_tracker.update_model_metrics(
                    model_version_id=self.current_model_version,
                    metrics={
                        "average_latency_ms": latency,
                        "request_count": 1,
                        "error_count": 0,
                        "token_count": tokens_used,
                        "drift_score": self.model_monitor.get_performance_metrics().get("drift_score", 0)
                    }
                )
                
                # Track usage if user_id is provided
                if user_id:
                    request_data = {
                        "symptoms": symptoms,
                        "additional_context": additional_context
                    }
                    await self.usage_tracker.track_analysis(
                        user_id=user_id,
                        request_data=request_data,
                        response_data=analysis_dict
                    )
                
                # Update service status
                SERVICE_STATUS.labels(service='groq').set(1)
                SERVICE_STATUS.labels(service='astra_db').set(1)
                
                return response
            except json.JSONDecodeError:
                MODEL_ERRORS.labels(
                    model=settings.GROQ_MODEL_NAME,
                    error_type='json_decode'
                ).inc()
                return self._create_fallback_response(completion, similar_cases)
                
        except Exception as e:
            MODEL_ERRORS.labels(
                model=settings.GROQ_MODEL_NAME,
                error_type='general'
            ).inc()
            # Track error in MLOps system
            await self.mlops_tracker.update_model_metrics(
                model_version_id=self.current_model_version,
                metrics={
                    "average_latency_ms": 0,
                    "request_count": 1,
                    "error_count": 1,
                    "token_count": 0,
                    "drift_score": 0
                }
            )
            raise e

    async def record_feedback(
        self,
        prediction_id: str,
        user_id: str,
        correct_diagnosis: Optional[str] = None,
        feedback_text: Optional[str] = None,
        severity_reported: Optional[str] = None
    ) -> None:
        """Record user feedback for a prediction"""
        await self.mlops_tracker.record_feedback(
            prediction_id=prediction_id,
            user_id=user_id,
            correct_diagnosis=correct_diagnosis,
            feedback_text=feedback_text,
            severity_reported=severity_reported
        )

    @track_latency(SERVICE_LATENCY, {'service': 'astra_db', 'operation': 'search'})
    async def _search_similar_cases(self, embedding: List[float]) -> List[Dict]:
        try:
            similar_cases = self.collection.find_many(
                filter={},
                sort={"$vector": embedding},
                limit=5
            )
            SERVICE_STATUS.labels(service='astra_db').set(1)
            return similar_cases
        except Exception as e:
            SERVICE_STATUS.labels(service='astra_db').set(0)
            raise e

    @track_latency(SERVICE_LATENCY, {'service': 'groq', 'operation': 'generate'})
    async def _get_groq_analysis(self, context: str) -> str:
        try:
            chat_completion = await self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical diagnostic AI assistant. Provide analyses in the requested JSON format. Be conservative with severity assessments and always recommend professional medical consultation."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                model=settings.GROQ_MODEL_NAME,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            SERVICE_STATUS.labels(service='groq').set(1)
            return chat_completion.choices[0].message.content
        except Exception as e:
            SERVICE_STATUS.labels(service='groq').set(0)
            raise e

    def _prepare_llm_context(self, similar_cases: List[Dict], symptoms: str, additional_context: str = None) -> str:
        context = f"""As a medical diagnostic AI, analyze the following symptoms:
Symptoms: {symptoms}
"""
        if additional_context:
            context += f"Additional Context: {additional_context}\n"

        context += "\nSimilar cases from the database:\n"
        for case in similar_cases:
            context += f"- Disease: {case['disease']}\n  Symptoms: {case['symptoms_text']}\n"

        context += """\nProvide a detailed analysis in the following JSON format:
{
    "possible_diseases": [
        {
            "name": "disease name",
            "confidence": 0.0 to 1.0,
            "symptoms_matched": ["symptom1", "symptom2"],
            "recommendation": "specific recommendation"
        }
    ],
    "analysis_summary": "detailed analysis",
    "severity_level": "LOW/MEDIUM/HIGH",
    "seek_immediate_care": boolean
}"""

        return context

    def _create_fallback_response(self, completion: str, similar_cases: List[Dict]) -> SymptomAnalysisResponse:
        # Create a basic response when JSON parsing fails
        possible_diseases = [
            Disease(
                name=case["disease"],
                confidence=0.5,
                symptoms_matched=case["symptoms_text"].split(", "),
                recommendation="Please consult a healthcare professional for accurate diagnosis."
            )
            for case in similar_cases[:3]
        ]

        return SymptomAnalysisResponse(
            possible_diseases=possible_diseases,
            analysis_summary="Analysis could not be fully processed. Please consult a healthcare professional.",
            severity_level="MEDIUM",
            seek_immediate_care=True
        )

    async def get_model_health(self) -> Dict[str, Any]:
        """Get current model health metrics"""
        return {
            "monitoring": self.model_monitor.get_performance_metrics(),
            "tokens_processed": MODEL_TOKENS.labels(
                model=settings.GROQ_MODEL_NAME,
                operation='analyze_symptoms'
            )._value.get(),
            "error_count": MODEL_ERRORS.labels(
                model=settings.GROQ_MODEL_NAME,
                error_type='general'
            )._value.get(),
        }