"""
OpenAI integration functionality for archaeological literature analysis.

This module provides classes and functions for integrating with OpenAI o3/o4 mini
and GPT-4.1 models to analyze archaeological literature, extract site information,
and generate descriptions for discovered sites.
"""

import asyncio
import json
import os
import re
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import tiktoken
import openai
from openai import OpenAI, AsyncOpenAI
import logging


@dataclass
class CompletionResponse:
    """Response from OpenAI completion."""
    content: str
    token_usage: int
    model: str
    finish_reason: str = "stop"


@dataclass
class ModelCapabilities:
    """Model capabilities assessment."""
    reasoning: float
    text_analysis: float
    code_generation: float
    multilingual: float
    context_length: int


class OpenAIClient:
    """Client for OpenAI API interactions."""
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, timeout: int = 60):
        """Initialize OpenAI client."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.max_retries = max_retries
        self.timeout = timeout
        self.default_model = "gpt-4.1"
        
        # Initialize clients
        self.sync_client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.async_client = AsyncOpenAI(api_key=self.api_key, timeout=timeout)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def async_completion(self, messages: List[Dict[str, str]], model: str = None, **kwargs) -> CompletionResponse:
        """Create async completion with retry logic."""
        model = model or self.default_model
        
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                
                return CompletionResponse(
                    content=response.choices[0].message.content,
                    token_usage=response.usage.total_tokens,
                    model=model,
                    finish_reason=response.choices[0].finish_reason
                )
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def sync_completion(self, messages: List[Dict[str, str]], model: str = None, **kwargs) -> CompletionResponse:
        """Create synchronous completion with retry logic."""
        model = model or self.default_model
        
        for attempt in range(self.max_retries):
            try:
                response = self.sync_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                
                return CompletionResponse(
                    content=response.choices[0].message.content,
                    token_usage=response.usage.total_tokens,
                    model=model,
                    finish_reason=response.choices[0].finish_reason
                )
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff


class LiteratureAnalyzer:
    """Analyzes archaeological literature using OpenAI models."""
    
    def __init__(self, openai_client: OpenAIClient):
        """Initialize literature analyzer."""
        self.openai_client = openai_client
        self.extracted_sites = []
        self.logger = logging.getLogger(__name__)
    
    def extract_site_coordinates(self, literature_text: str) -> List[Dict[str, Any]]:
        """Extract site coordinates from literature text."""
        prompt = f"""
        Analyze the following archaeological literature text and extract any site coordinates mentioned.
        Return the results as a JSON array of objects with 'latitude', 'longitude', and 'confidence' fields.
        Confidence should be between 0 and 1, where 1 means very certain about the coordinates.
        
        Literature text:
        {literature_text}
        
        Response format:
        [
            {{"latitude": -3.2, "longitude": -60.25, "confidence": 0.9}},
            {{"latitude": -3.5, "longitude": -61.2, "confidence": 0.8}}
        ]
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in archaeological literature analysis. Extract geographical coordinates mentioned in texts with high precision."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.openai_client.sync_completion(messages, model="gpt-4.1")
            coordinates = json.loads(response.content)
            
            # Validate coordinate format
            validated_coords = []
            for coord in coordinates:
                if all(key in coord for key in ['latitude', 'longitude', 'confidence']):
                    # Check if coordinates are in Amazon region (rough bounds)
                    if -10 <= coord['latitude'] <= 5 and -75 <= coord['longitude'] <= -45:
                        validated_coords.append(coord)
            
            return validated_coords
            
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Failed to extract coordinates: {e}")
            return []
    
    def analyze_site_descriptions(self, site_description: str) -> Dict[str, Any]:
        """Analyze archaeological site descriptions."""
        prompt = f"""
        Analyze this archaeological site description and extract key information.
        Return the analysis as JSON with the following structure:
        
        {{
            "site_type": "type of site (e.g., settlement, ceremonial, burial)",
            "features": ["list of architectural/cultural features"],
            "period": "time period or dates",
            "cultural_indicators": ["pottery", "tools", "etc"],
            "significance": "low/medium/high"
        }}
        
        Site description:
        {site_description}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert archaeologist specializing in pre-Columbian Amazon cultures. Analyze site descriptions with scholarly precision."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.openai_client.sync_completion(messages, model="gpt-4.1")
            analysis = json.loads(response.content)
            return analysis
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Failed to analyze site description: {e}")
            return {}
    
    def extract_temporal_information(self, text: str) -> Dict[str, Any]:
        """Extract temporal/dating information from text."""
        prompt = f"""
        Extract all temporal information from this archaeological text.
        Look for dates, time periods, radiocarbon dates, cultural periods, etc.
        Return as JSON:
        
        {{
            "radiocarbon_dates": ["list of C14 dates"],
            "cultural_periods": ["list of cultural periods"],
            "date_ranges": ["list of date ranges"],
            "dating_methods": ["methods used for dating"]
        }}
        
        Text:
        {text}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in archaeological dating methods and chronology. Extract all temporal information precisely."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.openai_client.sync_completion(messages, model="o3-mini")
            temporal_info = json.loads(response.content)
            return temporal_info
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Failed to extract temporal information: {e}")
            return {}
    
    def validate_extracted_information(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted information against known databases."""
        # Simple validation logic
        validation_result = {
            "coordinates_valid": True,
            "period_valid": True,
            "consistency_score": 0.8
        }
        
        # Check coordinates if present
        if "coordinates" in extracted_data:
            coords = extracted_data["coordinates"]
            if isinstance(coords, list) and coords:
                for coord in coords:
                    lat, lon = coord.get("latitude", 0), coord.get("longitude", 0)
                    # Amazon region check
                    if not (-10 <= lat <= 5 and -75 <= lon <= -45):
                        validation_result["coordinates_valid"] = False
                        validation_result["consistency_score"] -= 0.3
        
        # Check temporal period
        if "period" in extracted_data:
            period = extracted_data["period"]
            # Simple check for reasonable archaeological periods
            if not any(keyword in period.lower() for keyword in ["ce", "bp", "pre-columbian", "colonial"]):
                validation_result["period_valid"] = False
                validation_result["consistency_score"] -= 0.2
        
        validation_result["consistency_score"] = max(0.0, validation_result["consistency_score"])
        return validation_result


class SiteDescriptionGenerator:
    """Generates descriptions for discovered archaeological sites."""
    
    def __init__(self, openai_client: OpenAIClient):
        """Initialize site description generator."""
        self.openai_client = openai_client
        self.description_templates = {
            "discovery": "Standard site discovery description",
            "validation": "Validation report template",
            "technical": "Technical analysis description"
        }
        self.logger = logging.getLogger(__name__)
    
    def generate_site_description(self, site_data: Dict[str, Any]) -> str:
        """Generate archaeological site description."""
        coordinates = site_data.get("coordinates", (0, 0))
        features = site_data.get("features", {})
        confidence = site_data.get("confidence", 0.0)
        
        prompt = f"""
        Generate a professional archaeological site description based on the following data:
        
        Coordinates: {coordinates[0]:.4f}°, {coordinates[1]:.4f}°
        Confidence Score: {confidence:.2f}
        Detected Features: {features}
        
        Write a 2-3 paragraph description suitable for an archaeological report.
        Include details about:
        - Location and geographical context
        - Detected features and their archaeological significance
        - Potential cultural affiliation
        - Recommendations for further investigation
        
        Use professional archaeological terminology and maintain scientific objectivity.
        """
        
        messages = [
            {"role": "system", "content": "You are a professional archaeologist writing site descriptions for scientific publication. Use precise, objective language."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.openai_client.sync_completion(messages, model="gpt-4.1")
            return response.content
        except Exception as e:
            self.logger.error(f"Failed to generate site description: {e}")
            return f"Site at coordinates {coordinates[0]:.4f}°, {coordinates[1]:.4f}° requires further investigation."
    
    def generate_validation_report(self, site_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation report for discovered sites."""
        coordinates = site_data.get("coordinates", (0, 0))
        confidence = site_data.get("confidence", 0.0)
        evidence = site_data.get("supporting_evidence", {})
        
        prompt = f"""
        Generate a validation report for this potential archaeological site:
        
        Site Coordinates: {coordinates}
        Detection Confidence: {confidence}
        Supporting Evidence: {evidence}
        
        Create a JSON report with:
        {{
            "validation_summary": "brief summary of validation",
            "evidence_assessment": {{
                "lidar_evidence": "Strong/Moderate/Weak",
                "literature_support": "Strong/Moderate/Weak/None",
                "visual_confirmation": "Confirmed/Unconfirmed"
            }},
            "recommended_actions": ["list of recommended next steps"],
            "priority_level": "High/Medium/Low"
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an archaeological validation specialist. Assess evidence objectively and recommend appropriate follow-up actions."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.openai_client.sync_completion(messages, model="o4-mini")
            report = json.loads(response.content)
            return report
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Failed to generate validation report: {e}")
            return {"validation_summary": "Validation failed", "priority_level": "Low"}


class ArchaeologicalKnowledgeExtractor:
    """Extracts archaeological knowledge from literature corpus."""
    
    def __init__(self, openai_client: OpenAIClient):
        """Initialize knowledge extractor."""
        self.openai_client = openai_client
        self.knowledge_base = {}
        self.logger = logging.getLogger(__name__)
    
    def extract_site_characteristics(self, literature_corpus: List[str]) -> Dict[str, Any]:
        """Extract characteristic features from literature corpus."""
        # Combine literature texts
        combined_text = "\n\n".join(literature_corpus[:10])  # Limit for token management
        
        prompt = f"""
        Analyze this corpus of archaeological literature about Amazon sites and extract:
        
        1. Typical site features and characteristics
        2. Common site types and their indicators
        3. Environmental and landscape preferences
        
        Literature corpus:
        {combined_text}
        
        Return as JSON:
        {{
            "typical_features": ["list of common features"],
            "site_types": {{
                "settlement": ["indicators for settlements"],
                "ceremonial": ["indicators for ceremonial sites"],
                "agricultural": ["indicators for agricultural sites"]
            }},
            "environmental_indicators": ["landscape preferences"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in Amazon archaeology. Extract patterns and characteristics from scholarly literature."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.openai_client.sync_completion(messages, model="gpt-4.1")
            characteristics = json.loads(response.content)
            self.knowledge_base.update(characteristics)
            return characteristics
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Failed to extract site characteristics: {e}")
            return {}
    
    def build_pattern_database(self, known_sites: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build pattern database from known archaeological sites."""
        pattern_db = {
            "feature_patterns": {},
            "spatial_patterns": {},
            "temporal_patterns": {}
        }
        
        # Extract feature patterns
        all_features = []
        for site in known_sites:
            features = site.get("features", [])
            all_features.extend(features)
        
        # Count feature frequencies
        from collections import Counter
        feature_counts = Counter(all_features)
        pattern_db["feature_patterns"] = dict(feature_counts.most_common(10))
        
        # Extract spatial patterns (simplified)
        coordinates = [site.get("coordinates", (0, 0)) for site in known_sites]
        if coordinates:
            lats, lons = zip(*coordinates)
            pattern_db["spatial_patterns"] = {
                "lat_range": [min(lats), max(lats)],
                "lon_range": [min(lons), max(lons)],
                "centroid": [sum(lats)/len(lats), sum(lons)/len(lons)]
            }
        
        # Extract temporal patterns
        periods = [site.get("period", "") for site in known_sites if site.get("period")]
        pattern_db["temporal_patterns"] = {
            "common_periods": list(set(periods)),
            "period_count": len(periods)
        }
        
        return pattern_db


class ModelSelector:
    """Selects optimal OpenAI model for specific tasks."""
    
    def __init__(self):
        """Initialize model selector."""
        self.available_models = {
            "o3-mini": ModelCapabilities(0.8, 0.9, 0.7, 0.8, 128000),
            "o4-mini": ModelCapabilities(0.9, 0.95, 0.8, 0.85, 128000), 
            "gpt-4.1": ModelCapabilities(0.95, 0.98, 0.9, 0.9, 128000)
        }
        
        self.task_preferences = {
            "coordinate_extraction": "gpt-4.1",
            "text_analysis": "gpt-4.1",
            "report_generation": "gpt-4.1",
            "simple_extraction": "o3-mini",
            "complex_analysis": "gpt-4.1"
        }
    
    def select_optimal_model(self, task_type: str) -> str:
        """Select optimal model for specific task."""
        return self.task_preferences.get(task_type, "gpt-4.1")
    
    def assess_model_capabilities(self, model_name: str) -> Dict[str, float]:
        """Assess model capabilities for different tasks."""
        if model_name not in self.available_models:
            return {}
        
        capabilities = self.available_models[model_name]
        return {
            "reasoning": capabilities.reasoning,
            "text_analysis": capabilities.text_analysis,
            "code_generation": capabilities.code_generation,
            "multilingual": capabilities.multilingual
        }
    
    def select_cost_optimized_model(self, task_complexity: str) -> str:
        """Select cost-optimized model based on task complexity."""
        if task_complexity in ["simple", "simple_extraction"]:
            return "o3-mini"
        elif task_complexity in ["medium", "text_analysis"]:
            return "o4-mini"
        else:
            return "gpt-4.1"


class TokenManager:
    """Manages token usage and optimization."""
    
    def __init__(self, model: str = "gpt-4.1"):
        """Initialize token manager."""
        self.model = model
        self.max_tokens = 128000  # Default context length
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def optimize_prompt_length(self, text: str, max_tokens: int) -> str:
        """Optimize prompt length to fit within token limit."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def optimize_batch_processing(self, texts: List[str], max_tokens_per_batch: int = 100000) -> List[List[str]]:
        """Optimize texts into batches for processing."""
        batches = []
        current_batch = []
        current_token_count = 0
        
        for text in texts:
            text_tokens = self.count_tokens(text)
            
            if current_token_count + text_tokens > max_tokens_per_batch:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [text]
                current_token_count = text_tokens
            else:
                current_batch.append(text)
                current_token_count += text_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches


class PromptTemplate:
    """Manages prompt templates for different tasks."""
    
    def __init__(self):
        """Initialize prompt templates."""
        self.templates = {
            "coordinate_extraction": self._coordinate_template,
            "site_analysis": self._site_analysis_template,
            "validation": self._validation_template
        }
    
    def _coordinate_template(self, text: str) -> str:
        """Template for coordinate extraction."""
        return f"""
        Extract all geographical coordinates from the following archaeological literature.
        Look for latitude/longitude pairs, UTM coordinates, or descriptive locations.
        Return results as JSON array with latitude, longitude, and confidence (0-1).
        
        Text: {text}
        
        Format: [{{"latitude": -3.2, "longitude": -60.25, "confidence": 0.9}}]
        """
    
    def _site_analysis_template(self, description: str) -> str:
        """Template for site analysis."""
        return f"""
        Analyze this archaeological site description and extract key information:
        
        Description: {description}
        
        Identify:
        - Site type (settlement, ceremonial, agricultural, etc.)
        - Cultural features and artifacts
        - Temporal period if mentioned
        - Architectural elements
        - Significance level
        
        Provide analysis as structured JSON.
        """
    
    def _validation_template(self, site_data: Dict[str, Any]) -> str:
        """Template for validation prompts."""
        return f"""
        Validate this potential archaeological site discovery:
        
        Site Data: {site_data}
        
        Assess:
        - Coordinate validity and regional appropriateness
        - Feature consistency with known archaeological patterns
        - Confidence level justification
        - Recommended validation steps
        
        Provide validation assessment as JSON.
        """
    
    def create_coordinate_extraction_prompt(self, literature_text: str) -> str:
        """Create coordinate extraction prompt."""
        return self._coordinate_template(literature_text)
    
    def create_site_analysis_prompt(self, site_description: str) -> str:
        """Create site analysis prompt."""
        return self._site_analysis_template(site_description)
    
    def create_validation_prompt(self, site_data: Dict[str, Any]) -> str:
        """Create validation prompt."""
        return self._validation_template(site_data)
    
    def create_custom_template(self, task: str, variables: List[str], instructions: str) -> str:
        """Create custom prompt template."""
        variable_placeholders = "{" + "}, {".join(variables) + "}"
        
        template = f"""
        Task: {task}
        
        Instructions: {instructions}
        
        Variables: {variable_placeholders}
        
        Please provide a comprehensive response following the instructions above.
        """
        
        return template