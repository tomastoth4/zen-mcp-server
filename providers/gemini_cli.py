"""Gemini CLI model provider implementation.

This provider enables access to Gemini models through the locally installed
Gemini CLI without requiring API keys. It spawns the CLI as a subprocess
for each request and handles file references using Gemini's @ syntax.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from .base import (
    FixedTemperatureConstraint,
    ModelCapabilities,
    ModelProvider,
    ModelResponse,
    ProviderType,
)

logger = logging.getLogger(__name__)


class GeminiCLIProvider(ModelProvider):
    """Gemini CLI provider using subprocess execution - no API key required."""

    # Model configurations for CLI-based Gemini
    SUPPORTED_MODELS = {
        "gemini-cli": ModelCapabilities(
            provider=ProviderType.GEMINI_CLI,
            model_name="gemini-cli",
            friendly_name="Gemini CLI",
            context_window=1_000_000,  # Gemini's large context window
            max_output_tokens=50_000,  # Conservative estimate
            supports_extended_thinking=False,  # CLI doesn't expose thinking mode
            supports_system_prompts=False,  # CLI doesn't support system prompts
            supports_streaming=False,  # No streaming via CLI
            supports_function_calling=False,
            supports_json_mode=False,  # No JSON mode in CLI
            supports_images=True,  # Via file paths
            max_image_size_mb=20.0,  # Conservative limit
            supports_temperature=False,  # No temperature control in CLI
            temperature_constraint=FixedTemperatureConstraint(1.0),
            description="Gemini via local CLI - no API key required, supports @ file references",
            aliases=["gcli", "gemini-local", "gemini-no-api"],
        ),
        # Map common Gemini model names to CLI version
        "gemini-2.5-flash": ModelCapabilities(
            provider=ProviderType.GEMINI_CLI,
            model_name="gemini-2.5-flash",
            friendly_name="Gemini CLI (Flash)",
            context_window=1_000_000,
            max_output_tokens=50_000,
            supports_extended_thinking=False,
            supports_system_prompts=False,
            supports_streaming=False,
            supports_function_calling=False,
            supports_json_mode=False,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=False,
            temperature_constraint=FixedTemperatureConstraint(1.0),
            description="Gemini Flash via CLI - no API key required",
            aliases=["flash-cli", "gemini-flash-cli"],
        ),
        "gemini-2.5-pro": ModelCapabilities(
            provider=ProviderType.GEMINI_CLI,
            model_name="gemini-2.5-pro",
            friendly_name="Gemini CLI (Pro)",
            context_window=1_000_000,
            max_output_tokens=50_000,
            supports_extended_thinking=False,
            supports_system_prompts=False,
            supports_streaming=False,
            supports_function_calling=False,
            supports_json_mode=False,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=False,
            temperature_constraint=FixedTemperatureConstraint(1.0),
            description="Gemini Pro via CLI - no API key required",
            aliases=["pro-cli", "gemini-pro-cli"],
        ),
    }

    def __init__(self, api_key: str = "", **kwargs):
        """Initialize Gemini CLI provider.
        
        Args:
            api_key: Not used for CLI provider, included for interface compatibility
            **kwargs: Additional configuration options
        """
        # API key not needed for CLI, but keep interface consistent
        super().__init__(api_key or "not-required", **kwargs)
        
        # Check if Gemini CLI is available
        self.gemini_path = shutil.which("gemini")
        if not self.gemini_path:
            raise RuntimeError("Gemini CLI not found. Please install: npm install -g @google/generative-ai-cli")
        
        # Track temporary files for cleanup
        self._temp_files = []
        
        logger.info(f"Gemini CLI provider initialized using: {self.gemini_path}")

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific Gemini CLI model."""
        resolved_name = self._resolve_model_name(model_name)
        
        # For CLI, we support common model names but they all use the same CLI
        if resolved_name not in self.SUPPORTED_MODELS:
            # Default to gemini-cli for any unrecognized model
            resolved_name = "gemini-cli"
        
        # Check if model is allowed by restrictions
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(ProviderType.GEMINI_CLI, resolved_name, model_name):
            raise ValueError(f"Gemini CLI model '{resolved_name}' is not allowed by restriction policy.")
        
        return self.SUPPORTED_MODELS[resolved_name]

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,  # Ignored for CLI
        max_output_tokens: Optional[int] = None,  # Ignored for CLI
        images: Optional[list[str]] = None,
        files: Optional[list[dict]] = None,
        sandbox: bool = False,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using Gemini CLI.
        
        Args:
            prompt: User prompt to send to Gemini
            model_name: Model name (used for logging, actual model depends on CLI config)
            system_prompt: System prompt (will be prepended to user prompt)
            temperature: Ignored - CLI doesn't support temperature
            max_output_tokens: Ignored - CLI doesn't support output limits
            images: List of image paths or data URLs to include
            files: List of file dictionaries with 'path' and 'content' to include
            sandbox: Whether to run in sandbox mode
            **kwargs: Additional parameters (ignored)
            
        Returns:
            ModelResponse with generated content
        """
        resolved_name = self._resolve_model_name(model_name)
        
        # Build the full prompt with file references
        full_prompt = self._build_prompt_with_references(prompt, system_prompt, images, files)
        
        # Build command
        cmd = [self.gemini_path, "-p", full_prompt]
        
        # Add model flag if it's a specific model (not generic gemini-cli)
        if resolved_name in ["gemini-2.5-flash", "gemini-2.5-pro"]:
            cmd.extend(["-m", resolved_name])
        
        # Add sandbox flag if requested
        if sandbox:
            cmd.append("-s")
        
        # Add debug flag if in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            cmd.append("-d")
        
        # Execute command
        try:
            logger.debug(f"Executing Gemini CLI: {' '.join(cmd[:3])}...")  # Don't log full prompt
            
            # Set up environment to suppress Node.js warnings
            env = os.environ.copy()
            env["NODE_NO_WARNINGS"] = "1"
            
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env,
            )
            
            # Clean up any temporary files
            self._cleanup_temp_files()
            
            if result.returncode != 0:
                error_msg = result.stderr.strip()
                
                # Check for quota exhaustion
                if "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    raise RuntimeError(f"Gemini quota exhausted: {error_msg}")
                
                # Check for model not found
                if "model not found" in error_msg.lower():
                    raise ValueError(f"Model '{resolved_name}' not available in Gemini CLI")
                
                raise RuntimeError(f"Gemini CLI error (exit code {result.returncode}): {error_msg}")
            
            # Extract the response content
            content = result.stdout.strip()
            
            # Sometimes Gemini CLI outputs additional info before the actual response
            # Look for common patterns and extract just the response
            if "\n\n" in content:
                # Take everything after the last double newline as the response
                parts = content.split("\n\n")
                content = parts[-1]
            
            return ModelResponse(
                content=content,
                usage={},  # CLI doesn't provide token usage
                model_name=resolved_name,
                friendly_name=f"Gemini CLI ({resolved_name})",
                provider=ProviderType.GEMINI_CLI,
                metadata={
                    "sandbox": sandbox,
                    "cli_path": self.gemini_path,
                },
            )
            
        except subprocess.TimeoutExpired:
            self._cleanup_temp_files()
            raise RuntimeError(f"Gemini CLI timed out after 5 minutes")
        except Exception as e:
            self._cleanup_temp_files()
            raise RuntimeError(f"Gemini CLI execution failed: {str(e)}")

    def count_tokens(self, text: str, model_name: str) -> int:
        """Estimate token count for text.
        
        Note: CLI doesn't provide token counting, so we use a rough estimate.
        """
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.GEMINI_CLI

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is supported."""
        resolved_name = self._resolve_model_name(model_name)
        
        # Check if it's a known model or can default to gemini-cli
        if resolved_name in self.SUPPORTED_MODELS:
            return True
        
        # For CLI, we can try any model name and let the CLI handle it
        # This allows using models that might be available in CLI but not listed here
        logger.debug(f"Model '{model_name}' not in SUPPORTED_MODELS, will attempt with CLI")
        return True

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        # CLI doesn't expose thinking mode
        return False

    def _build_prompt_with_references(
        self,
        prompt: str,
        system_prompt: Optional[str],
        images: Optional[list[str]],
        files: Optional[list[dict]],
    ) -> str:
        """Build prompt with @ file references for Gemini CLI.
        
        This method handles converting embedded file content and data URLs
        to temporary files that can be referenced with @ syntax.
        """
        references = []
        
        # Handle file content that needs to be saved to temp files
        if files:
            for file_info in files:
                if isinstance(file_info, dict):
                    if "content" in file_info and file_info["content"]:
                        # Save content to temp file
                        temp_path = self._save_content_to_temp(
                            file_info["content"],
                            file_info.get("path", "content.txt"),
                        )
                        references.append(f"@{temp_path}")
                    elif "path" in file_info and os.path.exists(file_info["path"]):
                        # Use existing file path
                        references.append(f"@{file_info['path']}")
        
        # Handle images
        if images:
            for image_path in images:
                if image_path.startswith("data:"):
                    # Save data URL to temp file
                    temp_path = self._save_data_url_to_temp(image_path)
                    if temp_path:
                        references.append(f"@{temp_path}")
                elif os.path.exists(image_path):
                    references.append(f"@{image_path}")
                else:
                    logger.warning(f"Image path not found: {image_path}")
        
        # Combine system prompt, references, and user prompt
        parts = []
        
        if system_prompt:
            parts.append(system_prompt)
        
        if references:
            parts.append(" ".join(references))
        
        parts.append(prompt)
        
        return "\n\n".join(parts)

    def _save_content_to_temp(self, content: str, original_path: str) -> str:
        """Save content to a temporary file and return its path."""
        try:
            # Get file extension from original path
            ext = Path(original_path).suffix or ".txt"
            
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="zen_gemini_")
            
            # Write content
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Track for cleanup
            self._temp_files.append(temp_path)
            
            logger.debug(f"Created temp file: {temp_path} for {original_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create temp file for {original_path}: {e}")
            return ""

    def _save_data_url_to_temp(self, data_url: str) -> Optional[str]:
        """Save data URL to temporary file and return its path."""
        try:
            # Parse data URL
            header, data = data_url.split(",", 1)
            mime_type = header.split(";")[0].split(":")[1]
            
            # Determine file extension
            ext_map = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/jpg": ".jpg",
                "image/gif": ".gif",
                "image/webp": ".webp",
            }
            ext = ext_map.get(mime_type, ".png")
            
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="zen_gemini_img_")
            
            # Decode and write image data
            import base64
            image_bytes = base64.b64decode(data)
            
            with os.fdopen(fd, "wb") as f:
                f.write(image_bytes)
            
            # Track for cleanup
            self._temp_files.append(temp_path)
            
            logger.debug(f"Created temp image file: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to save data URL to temp file: {e}")
            return None

    def _cleanup_temp_files(self):
        """Clean up temporary files created during request."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
        
        self._temp_files.clear()

    def close(self):
        """Clean up any remaining temporary files."""
        self._cleanup_temp_files()
        super().close()

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        """Get preferred model for a category.
        
        For CLI provider, we always prefer the generic gemini-cli model
        since the actual model used depends on the user's CLI configuration.
        """
        # Check if any CLI model is in allowed list
        cli_models = ["gemini-cli", "gcli", "gemini-local"]
        for model in cli_models:
            if model in allowed_models:
                return model
        
        # Check for specific model names
        for model in allowed_models:
            if model in self.SUPPORTED_MODELS:
                return model
        
        return None