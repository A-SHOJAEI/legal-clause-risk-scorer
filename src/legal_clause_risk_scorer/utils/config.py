"""Configuration management utilities for legal clause risk scorer.

This module provides utilities for loading and managing configuration files,
with support for YAML configuration files and environment variable overrides.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


logger = logging.getLogger(__name__)


class Config:
    """Configuration manager with YAML file support and environment overrides.

    This class loads configuration from YAML files and provides access to
    configuration values with support for nested keys and type conversion.

    Attributes:
        _config: The loaded configuration dictionary
        config_path: Path to the loaded configuration file
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file. If None, uses default config.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the configuration file is malformed
        """
        if config_path is None:
            # Default to configs/default.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "configs" / "default.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._apply_env_overrides()

        logger.info(f"Configuration loaded from {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if config is None:
                raise ValueError("Configuration file is empty or invalid")

            return config

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration.

        Environment variables should follow the pattern:
        LEGAL_RISK_<SECTION>_<KEY> = value

        For example: LEGAL_RISK_TRAINING_BATCH_SIZE=32
        """
        env_prefix = "LEGAL_RISK_"

        for key, value in os.environ.items():
            if not key.startswith(env_prefix):
                continue

            # Remove prefix and convert to lowercase
            config_key = key[len(env_prefix):].lower()

            # Split nested keys (section_key format)
            if '_' in config_key:
                parts = config_key.split('_', 1)
                section, nested_key = parts[0], parts[1]

                if section in self._config and isinstance(self._config[section], dict):
                    # Convert value to appropriate type
                    converted_value = self._convert_env_value(value)
                    self._config[section][nested_key] = converted_value

                    logger.info(f"Environment override: {section}.{nested_key} = {converted_value}")

    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type.

        Args:
            value: String value from environment variable

        Returns:
            Converted value with appropriate type
        """
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string if no conversion applies
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key with support for nested keys.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config.get('training.batch_size')
            16
            >>> config.get('model.dropout', 0.1)
            0.1
        """
        try:
            if '.' in key:
                # Handle nested keys
                current = self._config
                for part in key.split('.'):
                    current = current[part]
                return current
            else:
                return self._config.get(key, default)

        except (KeyError, TypeError):
            logger.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key with support for nested keys.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set

        Examples:
            >>> config.set('training.batch_size', 32)
            >>> config.set('model.learning_rate', 1e-4)
        """
        if '.' in key:
            # Handle nested keys
            parts = key.split('.')
            current = self._config

            # Navigate to parent of target key
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the final key
            current[parts[-1]] = value
        else:
            self._config[key] = value

        logger.info(f"Configuration updated: {key} = {value}")

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Section name

        Returns:
            Section configuration dictionary

        Raises:
            KeyError: If section doesn't exist
        """
        if section not in self._config:
            raise KeyError(f"Configuration section '{section}' not found")

        return self._config[section].copy()

    def to_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()

    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to YAML file.

        Args:
            output_path: Path to save configuration. If None, overwrites current file.
        """
        if output_path is None:
            output_path = self.config_path
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    self._config,
                    f,
                    default_flow_style=False,
                    indent=2,
                    sort_keys=False
                )
            logger.info(f"Configuration saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    def validate_config(self) -> bool:
        """Validate configuration for required fields and reasonable values.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = ['data', 'model', 'training', 'evaluation']

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Required configuration section missing: {section}")

        # Validate specific fields
        training_config = self._config['training']

        if training_config.get('batch_size', 0) <= 0:
            raise ValueError("Training batch_size must be positive")

        if not (0 < training_config.get('learning_rate', 0) < 1):
            raise ValueError("Training learning_rate must be between 0 and 1")

        if training_config.get('num_epochs', 0) <= 0:
            raise ValueError("Training num_epochs must be positive")

        # Validate data splits sum to 1
        data_config = self._config['data']
        splits_sum = (
            data_config.get('train_split', 0) +
            data_config.get('val_split', 0) +
            data_config.get('test_split', 0)
        )

        if not (0.99 <= splits_sum <= 1.01):  # Allow for floating point precision
            raise ValueError(f"Data splits must sum to 1.0, got {splits_sum}")

        logger.info("Configuration validation passed")
        return True

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(path={self.config_path}, sections={list(self._config.keys())})"

    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return f"Config(config_path='{self.config_path}', config={self._config})"


def setup_logging(config: Config) -> None:
    """Setup logging based on configuration.

    Args:
        config: Configuration object with logging settings
    """
    log_config = config.get_section('logging')

    # Create logs directory if it doesn't exist
    if 'file' in log_config:
        log_file = Path(log_config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_config.get('file', 'logs/app.log'))
        ]
    )


def set_random_seeds(config: Config) -> None:
    """Set random seeds for reproducibility based on configuration.

    Args:
        config: Configuration object with random seed settings
    """
    import random
    import numpy as np
    import torch
    from transformers import set_seed

    # Get seeds from config
    random_seed = config.get('random_seed', 42)
    torch_seed = config.get('torch_seed', 42)
    numpy_seed = config.get('numpy_seed', 42)
    transformers_seed = config.get('transformers_seed', 42)

    # Set seeds
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    set_seed(transformers_seed)

    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seeds set: random={random_seed}, numpy={numpy_seed}, "
                f"torch={torch_seed}, transformers={transformers_seed}")


def create_output_directories(config: Config) -> None:
    """Create output directories based on configuration.

    Args:
        config: Configuration object with path settings
    """
    paths = config.get_section('paths')

    for path_name, path_value in paths.items():
        path = Path(path_value)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Global configuration instance

    Raises:
        RuntimeError: If configuration hasn't been initialized
    """
    global _global_config
    if _global_config is None:
        raise RuntimeError("Configuration not initialized. Call load_config() first.")
    return _global_config


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load and set global configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded configuration instance
    """
    global _global_config
    _global_config = Config(config_path)
    return _global_config