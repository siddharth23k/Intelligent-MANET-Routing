"""Configuration loader for centralized settings management."""

import yaml
from pathlib import Path
from typing import Dict, Any, List


class Config:
    """Centralized configuration manager."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "paper_scenarios.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    @property
    def area_size(self) -> List[int]:
        """Get area size [width, height]."""
        return self.config['paper_reference']['area_m']
    
    @property
    def area_center(self) -> float:
        """Get center point of the area (assuming square)."""
        area = self.area_size
        return area[0] / 2
    
    @property
    def communication_radius_default(self) -> float:
        """Get default communication radius."""
        return float(self.config['evaluation_defaults']['radius_m'])
    
    @property
    def communication_radii(self) -> List[float]:
        """Get all available communication radii."""
        return [float(r) for r in self.config['paper_reference']['communication_radius_m']]
    
    @property
    def random_seed(self) -> int:
        """Get default random seed."""
        return int(self.config['evaluation_defaults']['random_seed'])
    
    @property
    def test_run_count(self) -> int:
        """Get default test run count."""
        return int(self.config['evaluation_defaults']['test_run_count'])
    
    @property
    def pairs_per_step(self) -> int:
        """Get default pairs per step for evaluation."""
        return int(self.config['evaluation_defaults']['pairs_per_step'])
    
    @property
    def hello_interval(self) -> float:
        """Get hello interval (default 1.0)."""
        return 1.0
    
    def get_paper_reference_config(self) -> Dict[str, Any]:
        """Get the full paper reference configuration."""
        return self.config['paper_reference']
    
    def get_evaluation_defaults(self) -> Dict[str, Any]:
        """Get the evaluation defaults configuration."""
        return self.config['evaluation_defaults']


# Global config instance
_config = None

def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
