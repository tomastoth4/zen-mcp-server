"""Tests for Gemini CLI provider implementation."""

import os
import shutil
import subprocess
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from providers.base import ProviderType
from providers.gemini_cli import GeminiCLIProvider


class TestGeminiCLIProvider:
    """Test suite for Gemini CLI provider."""

    def test_init_without_cli(self):
        """Test initialization when Gemini CLI is not installed."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Gemini CLI not found"):
                GeminiCLIProvider()

    def test_init_with_cli(self):
        """Test successful initialization when Gemini CLI is installed."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            assert provider.gemini_path == "/usr/local/bin/gemini"
            assert provider._temp_files == []

    def test_get_capabilities(self):
        """Test getting model capabilities."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            # Test known model
            caps = provider.get_capabilities("gemini-cli")
            assert caps.model_name == "gemini-cli"
            assert caps.provider == ProviderType.GEMINI_CLI
            assert caps.context_window == 1_000_000
            assert not caps.supports_temperature
            assert caps.supports_images
            
            # Test alias
            caps = provider.get_capabilities("gcli")
            assert caps.model_name == "gemini-cli"
            
            # Test specific model mapping
            caps = provider.get_capabilities("gemini-2.5-flash")
            assert caps.model_name == "gemini-2.5-flash"

    def test_validate_model_name(self):
        """Test model name validation."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            # Known models should be valid
            assert provider.validate_model_name("gemini-cli")
            assert provider.validate_model_name("gcli")
            assert provider.validate_model_name("gemini-2.5-flash")
            
            # Unknown models should still be valid (CLI will handle)
            assert provider.validate_model_name("unknown-model")

    def test_generate_content_basic(self):
        """Test basic content generation."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            # Mock subprocess.run
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Generated response from Gemini CLI"
            mock_result.stderr = ""
            
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                response = provider.generate_content(
                    prompt="Hello, world!",
                    model_name="gemini-cli"
                )
                
                # Check subprocess was called correctly
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert args[0] == "/usr/local/bin/gemini"
                assert args[1] == "-p"
                assert args[2] == "Hello, world!"
                
                # Check response
                assert response.content == "Generated response from Gemini CLI"
                assert response.model_name == "gemini-cli"
                assert response.provider == ProviderType.GEMINI_CLI

    def test_generate_content_with_system_prompt(self):
        """Test content generation with system prompt."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Response"
            mock_result.stderr = ""
            
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                response = provider.generate_content(
                    prompt="User prompt",
                    model_name="gemini-cli",
                    system_prompt="System instructions"
                )
                
                # Check the combined prompt
                args = mock_run.call_args[0][0]
                assert args[2] == "System instructions\n\nUser prompt"

    def test_generate_content_with_files(self):
        """Test content generation with file references."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Response"
            mock_result.stderr = ""
            
            # Mock tempfile creation
            with patch("tempfile.mkstemp", return_value=(999, "/tmp/zen_gemini_123.txt")):
                with patch("os.fdopen", MagicMock()):
                    with patch("subprocess.run", return_value=mock_result) as mock_run:
                        response = provider.generate_content(
                            prompt="Analyze this",
                            model_name="gemini-cli",
                            files=[{"content": "File content", "path": "test.txt"}]
                        )
                        
                        # Check the prompt includes file reference
                        args = mock_run.call_args[0][0]
                        assert "@/tmp/zen_gemini_123.txt" in args[2]
                        assert "Analyze this" in args[2]

    def test_generate_content_with_images(self):
        """Test content generation with image references."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Response"
            mock_result.stderr = ""
            
            # Test with file path
            with patch("os.path.exists", return_value=True):
                with patch("subprocess.run", return_value=mock_result) as mock_run:
                    response = provider.generate_content(
                        prompt="Describe this image",
                        model_name="gemini-cli",
                        images=["/path/to/image.png"]
                    )
                    
                    args = mock_run.call_args[0][0]
                    assert "@/path/to/image.png" in args[2]

    def test_generate_content_with_data_url_image(self):
        """Test content generation with data URL images."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Response"
            mock_result.stderr = ""
            
            # Mock tempfile for image
            with patch("tempfile.mkstemp", return_value=(999, "/tmp/zen_gemini_img_123.png")):
                with patch("os.fdopen", MagicMock()):
                    with patch("subprocess.run", return_value=mock_result) as mock_run:
                        # Simple base64 PNG data
                        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                        response = provider.generate_content(
                            prompt="Describe this",
                            model_name="gemini-cli",
                            images=[data_url]
                        )
                        
                        args = mock_run.call_args[0][0]
                        assert "@/tmp/zen_gemini_img_123.png" in args[2]

    def test_generate_content_with_sandbox(self):
        """Test content generation with sandbox mode."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Response"
            mock_result.stderr = ""
            
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                response = provider.generate_content(
                    prompt="Run this code",
                    model_name="gemini-cli",
                    sandbox=True
                )
                
                args = mock_run.call_args[0][0]
                assert "-s" in args

    def test_generate_content_with_model_flag(self):
        """Test content generation with specific model flag."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Response"
            mock_result.stderr = ""
            
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                response = provider.generate_content(
                    prompt="Hello",
                    model_name="gemini-2.5-flash"
                )
                
                args = mock_run.call_args[0][0]
                assert "-m" in args
                assert "gemini-2.5-flash" in args

    def test_generate_content_error_handling(self):
        """Test error handling in content generation."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            # Test non-zero return code
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "Error occurred"
            
            with patch("subprocess.run", return_value=mock_result):
                with pytest.raises(RuntimeError, match="Gemini CLI error.*exit code 1.*Error occurred"):
                    provider.generate_content("Hello", "gemini-cli")
            
            # Test quota exhaustion
            mock_result.stderr = "RESOURCE_EXHAUSTED: Quota exceeded"
            with patch("subprocess.run", return_value=mock_result):
                with pytest.raises(RuntimeError, match="Gemini quota exhausted"):
                    provider.generate_content("Hello", "gemini-cli")
            
            # Test timeout
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("gemini", 300)):
                with pytest.raises(RuntimeError, match="Gemini CLI timed out"):
                    provider.generate_content("Hello", "gemini-cli")

    def test_count_tokens(self):
        """Test token counting estimation."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            # Should estimate ~4 chars per token
            assert provider.count_tokens("Hello world!", "gemini-cli") == 3
            assert provider.count_tokens("A" * 100, "gemini-cli") == 25

    def test_cleanup_temp_files(self):
        """Test temporary file cleanup."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            # Add some fake temp files
            provider._temp_files = ["/tmp/file1", "/tmp/file2"]
            
            with patch("os.path.exists", return_value=True):
                with patch("os.unlink") as mock_unlink:
                    provider._cleanup_temp_files()
                    
                    assert mock_unlink.call_count == 2
                    assert provider._temp_files == []

    def test_provider_type(self):
        """Test provider type identification."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            assert provider.get_provider_type() == ProviderType.GEMINI_CLI

    def test_supports_thinking_mode(self):
        """Test thinking mode support (should be False for CLI)."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            assert not provider.supports_thinking_mode("gemini-cli")

    def test_response_content_extraction(self):
        """Test extraction of response content from CLI output."""
        with patch("shutil.which", return_value="/usr/local/bin/gemini"):
            provider = GeminiCLIProvider()
            
            # Test output with extra info before response
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Loading model...\n\nActual response content"
            mock_result.stderr = ""
            
            with patch("subprocess.run", return_value=mock_result):
                response = provider.generate_content("Hello", "gemini-cli")
                assert response.content == "Actual response content"