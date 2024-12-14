"""YAML configuration file reader with validation and caching."""

from typing import Any, Dict, Optional
import os
from pathlib import Path
import yaml
from mb_utils.src.logging import logger

__all__ = ['YAMLReader']

class YAMLReader:
    """Read and validate YAML configuration files."""
    
    def __init__(self, yaml_path: str):
        """
        Initialize YAML reader.
        
        Args:
            yaml_path: Path to YAML file
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If file extension is not .yaml or .yml
        """
        self.yaml_path = Path(yaml_path)
        
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
            
        if not self.yaml_path.suffix in ['.yaml', '.yml']:
            raise ValueError(f"Invalid file extension: {self.yaml_path.suffix}")
            
        self._data: Optional[Dict[str, Any]] = None
        
    def read(self, logger: Optional[Any] = None) -> Dict[str, Any]:
        """
        Read and parse YAML file.
        
        Args:
            logger: Optional logger instance
            
        Returns:
            Parsed YAML data
            
        Raises:
            yaml.YAMLError: If YAML parsing fails
        """
        if self._data is None:
            try:
                with open(self.yaml_path, 'r', encoding='utf-8') as f:
                    self._data = yaml.safe_load(f)
                    
                if logger:
                    logger.info(f"Successfully read YAML file: {self.yaml_path}")
                    
            except yaml.YAMLError as e:
                if logger:
                    logger.error(f"Failed to parse YAML file: {e}")
                raise
                
        return self._data
    
    def get_value(
        self,
        key: str,
        default: Any = None,
        required: bool = False,
        logger: Optional[Any] = None
    ) -> Any:
        """
        Get value from YAML data by key.
        
        Args:
            key: Key to lookup
            default: Default value if key not found
            required: Whether key is required
            logger: Optional logger instance
            
        Returns:
            Value from YAML data
            
        Raises:
            KeyError: If required key is missing
        """
        if self._data is None:
            self.read(logger)
            
        try:
            value = self._data
            for k in key.split('.'):
                value = value[k]
            return value
            
        except KeyError:
            if required:
                if logger:
                    logger.error(f"Required key not found: {key}")
                raise KeyError(f"Required key not found: {key}")
                
            if logger:
                logger.warning(f"Key not found, using default: {key} = {default}")
            return default
    
    def validate_keys(
        self,
        required_keys: list,
        logger: Optional[Any] = None
    ) -> bool:
        """
        Validate presence of required keys.
        
        Args:
            required_keys: List of required keys
            logger: Optional logger instance
            
        Returns:
            True if all required keys present
            
        Raises:
            KeyError: If any required key is missing
        """
        if self._data is None:
            self.read(logger)
            
        missing_keys = []
        for key in required_keys:
            try:
                self.get_value(key, required=True)
            except KeyError:
                missing_keys.append(key)
                
        if missing_keys:
            if logger:
                logger.error(f"Missing required keys: {missing_keys}")
            raise KeyError(f"Missing required keys: {missing_keys}")
            
        return True
