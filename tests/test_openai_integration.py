"""
Tests for OpenAI integration functionality.

Following TDD approach, these tests define the expected interface and behavior
for OpenAI model integration before implementation. This includes o3/o4 mini
and GPT-4.1 models for archaeological literature analysis.
"""

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Import modules to be tested (these will fail initially)
try:
    from src.openai_integration import (
        ArchaeologicalKnowledgeExtractor,
        LiteratureAnalyzer,
        ModelSelector,
        OpenAIClient,
        PromptTemplate,
        SiteDescriptionGenerator,
        TokenManager,
    )
except ImportError:
    # Expected to fail initially in TDD
    pass


class TestOpenAIClient:
    """Test cases for OpenAI client functionality."""

    def test_client_initialization(self):
        """Test OpenAI client initialization with API key."""
        client = OpenAIClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.default_model is not None
        assert client.max_retries > 0

    def test_client_initialization_from_env(self):
        """Test OpenAI client initialization from environment variable."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env_test_key"}):
            client = OpenAIClient()
            assert client.api_key == "env_test_key"

    def test_client_missing_api_key_raises_error(self):
        """Test that missing API key raises appropriate error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                OpenAIClient()

    @pytest.mark.asyncio
    async def test_async_completion(self):
        """Test async completion functionality."""
        with patch("src.openai_integration.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.usage.total_tokens = 100

            mock_openai.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            client = OpenAIClient(api_key="test_key")
            response = await client.async_completion(
                messages=[{"role": "user", "content": "Test prompt"}],
                model="gpt-4-turbo",
            )

            assert response.content == "Test response"
            assert response.token_usage == 100

    def test_sync_completion(self):
        """Test synchronous completion functionality."""
        with patch("src.openai_integration.OpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.usage.total_tokens = 100

            mock_openai.return_value.chat.completions.create.return_value = (
                mock_response
            )

            client = OpenAIClient(api_key="test_key")
            response = client.sync_completion(
                messages=[{"role": "user", "content": "Test prompt"}], model="o3-mini"
            )

            assert response.content == "Test response"
            assert response.token_usage == 100

    def test_error_handling_with_retries(self):
        """Test error handling and retry mechanism."""
        with patch("src.openai_integration.OpenAI") as mock_openai:
            # First two calls fail, third succeeds
            mock_openai.return_value.chat.completions.create.side_effect = [
                Exception("API Error"),
                Exception("Rate Limit"),
                MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Success"))],
                    usage=MagicMock(total_tokens=50),
                ),
            ]

            client = OpenAIClient(api_key="test_key", max_retries=3)
            response = client.sync_completion(
                messages=[{"role": "user", "content": "Test"}], model="gpt-4-turbo"
            )

            assert response.content == "Success"
            assert mock_openai.return_value.chat.completions.create.call_count == 3


class TestLiteratureAnalyzer:
    """Test cases for archaeological literature analysis."""

    def test_analyzer_initialization(self):
        """Test literature analyzer initialization."""
        client = Mock()
        analyzer = LiteratureAnalyzer(client)
        assert analyzer.openai_client is client
        assert analyzer.extracted_sites == []

    def test_extract_site_coordinates(self):
        """Test extraction of site coordinates from literature text."""
        client = Mock()
        analyzer = LiteratureAnalyzer(client)

        # Mock literature text with coordinates
        literature_text = """
        The ancient settlement was discovered at coordinates 3°12'S, 60°15'W.
        Excavations revealed pottery fragments and stone tools.
        Another site at -3.5°, -61.2° showed similar artifacts.
        """

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.content = json.dumps(
            [
                {"latitude": -3.2, "longitude": -60.25, "confidence": 0.9},
                {"latitude": -3.5, "longitude": -61.2, "confidence": 0.8},
            ]
        )
        client.sync_completion.return_value = mock_response

        coordinates = analyzer.extract_site_coordinates(literature_text)

        assert isinstance(coordinates, list)
        assert len(coordinates) == 2
        assert coordinates[0]["latitude"] == -3.2
        assert coordinates[0]["longitude"] == -60.25
        assert all(0 <= coord["confidence"] <= 1 for coord in coordinates)

    def test_analyze_site_descriptions(self):
        """Test analysis of archaeological site descriptions."""
        client = Mock()
        analyzer = LiteratureAnalyzer(client)

        site_description = """
        The site features a large rectangular platform measuring 50x30 meters,
        surrounded by smaller circular structures. Pottery analysis suggests
        occupation from 800-1200 CE. The location shows evidence of
        agricultural terracing and water management systems.
        """

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.content = json.dumps(
            {
                "site_type": "ceremonial complex",
                "features": ["platform", "circular structures", "terracing"],
                "period": "800-1200 CE",
                "cultural_indicators": ["pottery", "water management"],
                "significance": "high",
            }
        )
        client.sync_completion.return_value = mock_response

        analysis = analyzer.analyze_site_descriptions(site_description)

        assert isinstance(analysis, dict)
        assert "site_type" in analysis
        assert "features" in analysis
        assert analysis["site_type"] == "ceremonial complex"
        assert "platform" in analysis["features"]

    def test_extract_temporal_information(self):
        """Test extraction of temporal/dating information."""
        client = Mock()
        analyzer = LiteratureAnalyzer(client)

        text_with_dates = """
        Radiocarbon dating of charcoal samples yielded dates of 1150±50 BP.
        Ceramic analysis suggests the site was occupied during the late
        pre-Columbian period, approximately 1000-1500 CE.
        """

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.content = json.dumps(
            {
                "radiocarbon_dates": ["1150±50 BP"],
                "cultural_periods": ["late pre-Columbian"],
                "date_ranges": ["1000-1500 CE"],
                "dating_methods": ["radiocarbon", "ceramic analysis"],
            }
        )
        client.sync_completion.return_value = mock_response

        temporal_info = analyzer.extract_temporal_information(text_with_dates)

        assert isinstance(temporal_info, dict)
        assert "radiocarbon_dates" in temporal_info
        assert "date_ranges" in temporal_info
        assert "1150±50 BP" in temporal_info["radiocarbon_dates"]

    def test_validate_extracted_information(self):
        """Test validation of extracted information against known databases."""
        client = Mock()
        analyzer = LiteratureAnalyzer(client)

        extracted_data = {
            "coordinates": [{"latitude": -3.2, "longitude": -60.25}],
            "site_type": "settlement",
            "period": "1000-1500 CE",
        }

        validation_result = analyzer.validate_extracted_information(extracted_data)

        assert isinstance(validation_result, dict)
        assert "coordinates_valid" in validation_result
        assert "period_valid" in validation_result
        assert "consistency_score" in validation_result
        assert isinstance(validation_result["consistency_score"], float)
        assert 0 <= validation_result["consistency_score"] <= 1


class TestSiteDescriptionGenerator:
    """Test cases for generating site descriptions."""

    def test_generator_initialization(self):
        """Test site description generator initialization."""
        client = Mock()
        generator = SiteDescriptionGenerator(client)
        assert generator.openai_client is client
        assert generator.description_templates is not None

    def test_generate_site_description(self):
        """Test generation of archaeological site descriptions."""
        client = Mock()
        generator = SiteDescriptionGenerator(client)

        site_data = {
            "coordinates": (-3.2, -60.25),
            "features": {
                "elevation": 150.0,
                "slope": 2.5,
                "vegetation_anomaly": True,
                "terrain_features": ["platform", "linear_feature"],
            },
            "confidence": 0.85,
        }

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.content = """
        Potential archaeological site identified at coordinates 3°12'S, 60°15'W
        in the Amazon rainforest. The site exhibits a raised platform structure
        approximately 50 meters in diameter, visible through LiDAR analysis
        despite dense forest cover. Vegetation patterns suggest historical
        human modification of the landscape.
        """
        client.sync_completion.return_value = mock_response

        description = generator.generate_site_description(site_data)

        assert isinstance(description, str)
        assert len(description) > 50  # Substantial description
        assert "archaeological site" in description.lower()
        assert str(site_data["coordinates"][0]) in description or "3°12" in description

    def test_generate_validation_report(self):
        """Test generation of validation reports for discovered sites."""
        client = Mock()
        generator = SiteDescriptionGenerator(client)

        site_data = {
            "coordinates": (-3.2, -60.25),
            "confidence": 0.85,
            "supporting_evidence": {
                "lidar_tile_id": "TILE_123",
                "literature_references": ["10.1234/example.doi"],
                "visual_confirmation": True,
            },
        }

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.content = json.dumps(
            {
                "validation_summary": "High confidence site with strong evidence",
                "evidence_assessment": {
                    "lidar_evidence": "Strong",
                    "literature_support": "Moderate",
                    "visual_confirmation": "Confirmed",
                },
                "recommended_actions": ["Field survey", "Additional remote sensing"],
                "priority_level": "High",
            }
        )
        client.sync_completion.return_value = mock_response

        report = generator.generate_validation_report(site_data)

        assert isinstance(report, dict)
        assert "validation_summary" in report
        assert "evidence_assessment" in report
        assert "priority_level" in report


class TestArchaeologicalKnowledgeExtractor:
    """Test cases for extracting archaeological knowledge."""

    def test_extractor_initialization(self):
        """Test knowledge extractor initialization."""
        client = Mock()
        extractor = ArchaeologicalKnowledgeExtractor(client)
        assert extractor.openai_client is client
        assert extractor.knowledge_base == {}

    def test_extract_site_characteristics(self):
        """Test extraction of characteristic features from literature."""
        client = Mock()
        extractor = ArchaeologicalKnowledgeExtractor(client)

        literature_corpus = [
            "Amazon settlements typically feature raised platforms for flood protection",
            "Circular plazas are common in ceremonial sites",
            "Agricultural terraces indicate intensive farming practices",
        ]

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.content = json.dumps(
            {
                "typical_features": [
                    "raised platforms",
                    "circular plazas",
                    "agricultural terraces",
                ],
                "site_types": {
                    "settlement": ["raised platforms", "flood protection"],
                    "ceremonial": ["circular plazas", "formal architecture"],
                    "agricultural": ["terraces", "water management"],
                },
                "environmental_indicators": ["flood protection", "water management"],
            }
        )
        client.sync_completion.return_value = mock_response

        characteristics = extractor.extract_site_characteristics(literature_corpus)

        assert isinstance(characteristics, dict)
        assert "typical_features" in characteristics
        assert "site_types" in characteristics
        assert "raised platforms" in characteristics["typical_features"]

    def test_build_pattern_database(self):
        """Test building of archaeological pattern database."""
        client = Mock()
        extractor = ArchaeologicalKnowledgeExtractor(client)

        known_sites = [
            {
                "coordinates": (-3.1, -60.2),
                "features": ["platform", "plaza"],
                "period": "1000-1500 CE",
                "site_type": "ceremonial",
            },
            {
                "coordinates": (-3.5, -61.0),
                "features": ["terraces", "canals"],
                "period": "800-1200 CE",
                "site_type": "agricultural",
            },
        ]

        pattern_db = extractor.build_pattern_database(known_sites)

        assert isinstance(pattern_db, dict)
        assert "feature_patterns" in pattern_db
        assert "spatial_patterns" in pattern_db
        assert "temporal_patterns" in pattern_db
        assert len(pattern_db["feature_patterns"]) > 0


class TestModelSelector:
    """Test cases for OpenAI model selection functionality."""

    def test_model_selector_initialization(self):
        """Test model selector initialization."""
        selector = ModelSelector()
        assert selector.available_models is not None
        assert "o3-mini" in selector.available_models
        assert "o4-mini" in selector.available_models
        assert "gpt-4-turbo" in selector.available_models

    def test_select_optimal_model(self):
        """Test selection of optimal model for specific tasks."""
        selector = ModelSelector()

        # Test different task types
        coord_extraction_model = selector.select_optimal_model("coordinate_extraction")
        text_analysis_model = selector.select_optimal_model("text_analysis")
        report_generation_model = selector.select_optimal_model("report_generation")

        assert coord_extraction_model in selector.available_models
        assert text_analysis_model in selector.available_models
        assert report_generation_model in selector.available_models

    def test_model_capability_assessment(self):
        """Test assessment of model capabilities for different tasks."""
        selector = ModelSelector()

        capabilities = selector.assess_model_capabilities("gpt-4-turbo")

        assert isinstance(capabilities, dict)
        assert "reasoning" in capabilities
        assert "text_analysis" in capabilities
        assert "code_generation" in capabilities
        assert all(0 <= score <= 1 for score in capabilities.values())

    def test_cost_optimization(self):
        """Test cost-optimized model selection."""
        selector = ModelSelector()

        # Should prefer cheaper models for simple tasks
        simple_task_model = selector.select_cost_optimized_model("simple_extraction")
        complex_task_model = selector.select_cost_optimized_model("complex_analysis")

        assert simple_task_model in ["o3-mini", "o4-mini"]  # Cheaper models
        assert complex_task_model in selector.available_models


class TestTokenManager:
    """Test cases for token management and optimization."""

    def test_token_manager_initialization(self):
        """Test token manager initialization."""
        manager = TokenManager()
        assert manager.encoding is not None
        assert manager.max_tokens > 0

    def test_count_tokens(self):
        """Test token counting functionality."""
        manager = TokenManager()

        test_text = "This is a test text for token counting."
        token_count = manager.count_tokens(test_text)

        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < 100  # Should be reasonable for short text

    def test_optimize_prompt_length(self):
        """Test prompt length optimization."""
        manager = TokenManager()

        long_text = "This is a very long text. " * 1000  # Very long text
        optimized_text = manager.optimize_prompt_length(long_text, max_tokens=100)

        assert len(optimized_text) < len(long_text)
        assert manager.count_tokens(optimized_text) <= 100

    def test_batch_processing_optimization(self):
        """Test optimization for batch processing."""
        manager = TokenManager()

        texts = ["Short text one", "Short text two", "Short text three"]

        batches = manager.optimize_batch_processing(texts, max_tokens_per_batch=200)

        assert isinstance(batches, list)
        assert len(batches) >= 1
        assert all(isinstance(batch, list) for batch in batches)


class TestPromptTemplate:
    """Test cases for prompt template functionality."""

    def test_template_initialization(self):
        """Test prompt template initialization."""
        template = PromptTemplate()
        assert template.templates is not None
        assert len(template.templates) > 0

    def test_coordinate_extraction_template(self):
        """Test coordinate extraction prompt template."""
        template = PromptTemplate()

        literature_text = "The site is located at 3°S, 60°W in the Amazon."
        prompt = template.create_coordinate_extraction_prompt(literature_text)

        assert isinstance(prompt, str)
        assert "coordinate" in prompt.lower()
        assert literature_text in prompt
        assert "json" in prompt.lower()  # Should request JSON format

    def test_site_analysis_template(self):
        """Test site analysis prompt template."""
        template = PromptTemplate()

        site_description = "Large platform with surrounding structures"
        prompt = template.create_site_analysis_prompt(site_description)

        assert isinstance(prompt, str)
        assert "archaeolog" in prompt.lower()
        assert site_description in prompt
        assert "analyz" in prompt.lower()

    def test_validation_template(self):
        """Test validation prompt template."""
        template = PromptTemplate()

        site_data = {
            "coordinates": (-3.2, -60.25),
            "features": ["platform", "plaza"],
            "confidence": 0.85,
        }

        prompt = template.create_validation_prompt(site_data)

        assert isinstance(prompt, str)
        assert "validat" in prompt.lower()
        assert str(site_data["coordinates"][0]) in prompt
        assert "platform" in prompt

    def test_custom_template_creation(self):
        """Test creation of custom prompt templates."""
        template = PromptTemplate()

        custom_template = template.create_custom_template(
            task="feature_extraction",
            variables=["input_data", "feature_types"],
            instructions="Extract the specified features from the input data",
        )

        assert isinstance(custom_template, str)
        assert "{input_data}" in custom_template
        assert "{feature_types}" in custom_template
        assert "Extract" in custom_template


if __name__ == "__main__":
    pytest.main([__file__])
