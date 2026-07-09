from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class Config:
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parents[1]
    
    # Subdirectories
    MASTER_DIR: Path = BASE_DIR / "data" / "master"
    MODEL_DIR: Path = BASE_DIR / "model"
    INPUT_DIR: Path = BASE_DIR / "data" / "input"
    OUTPUT_DIR: Path = BASE_DIR / "data" / "output"
    
    # Files
    MASTER_PATH: Path = MASTER_DIR / "검색량.csv"
    MODEL_PATH: Path = MODEL_DIR / "rf_classifier.joblib"
    VECTOR_CACHE_PATH: Path = MODEL_DIR / "master_vectors.npy"
    SITEMAP_PATH: Path = BASE_DIR / "samsung_model_master.txt"
    SITEMAP_XML_PATH: Path = BASE_DIR / "data" / "sitemaps" / "sitemap_main.xml"
    SITEMAP_PROD_XML_PATH: Path = BASE_DIR / "data" / "sitemaps" / "sitemap_products.xml"

    # Business Logic Constants
    ALLOWED_BRANDS: List[str] = None
    SAMSUNG_IDENTIFIERS: List[str] = None

    def __post_init__(self):
        # Initialize constants
        self.ALLOWED_BRANDS = [
            'Samsung', 'Apple', 'Bissell', 'Bosch', 'Bose', 'Dyson', 'Ecovacs', 'Electrolux', 
            'Eufy', 'Fitbit', 'Generic', 'iHome', 'iRobot Roomba', 'LG', 'Miele', 'Oppo', 
            'Others', 'Roborock', 'Shark', 'Vivo', 'Whirlpool', 'Xiaomi'
        ]
        self.SAMSUNG_IDENTIFIERS = [
            'samsung', 'galaxy', 'bespoke', 'jet bot', 'odyssey', 'neo qled', 'qled', 'uhd', 'frame tv', 
            's95', 's90', 's85', 'the frame', 'the serif', 'smart monitor', 'galaxy book'
        ]

    def validate_paths(self):
        """Checks if critical paths exist."""
        required = {
            "Master CSV": self.MASTER_PATH,
            # "Model File": self.MODEL_PATH, # Optional?
        }
        missing = []
        for name, p in required.items():
            if not p.exists():
                missing.append(f"{name} ({p})")
        
        if missing:
            raise FileNotFoundError(f"Missing critical files: {', '.join(missing)}")

    def ensure_directories(self):
        """Creates output directory if not exists."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

