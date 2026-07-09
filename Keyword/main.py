import sys
import logging
import pandas as pd
from pathlib import Path
from logging.handlers import RotatingFileHandler
from src.config import Config
from src.sitemap_mgr import SitemapManager
from src.inference import SBERTVectorizer, MLModel
from src.engine import UltimateHybridEngine
from src.pipeline import Pipeline

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # console
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(c_handler)
    
    # file (rotating) - Use absolute path in output dir
    try:
        log_path = Config.OUTPUT_DIR / 'app.log'
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        f_handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
        f_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(f_handler)
    except Exception as e:
        print(f"!!! Warning: Could not setup file logging: {e}")
    
    return logger

def main():
    logger = setup_logging()
    logger.info(">>> [System] Starting Keyword Engine...")
    
    # 0. Ensure scripts folder is in path for imports
    config = Config()
    scripts_path = str((config.BASE_DIR / "scripts").resolve())
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)
    
    try:
        config.validate_paths()
        
        # 1. Load Master Table (Now from data/master)
        logger.info(f">>> Loading Master DB from {config.MASTER_PATH}")
        master_df = pd.read_csv(config.MASTER_PATH)
        
        # Preprocess Master Data
        logger.info(">>> Preprocessing Master Data...")
        master_df['kw_low'] = master_df['Keyword'].astype(str).str.lower().str.strip()
        master_df = master_df.drop_duplicates(subset=['kw_low'])
        
        # 2. Check/Train ML Model
        logger.info(f">>> Checking ML Model at {config.MODEL_PATH}")
        if not config.MODEL_PATH.exists():
            logger.warning("ML Model not found. Starting automatic training from master data...")
            try:
                import train_rf
                train_rf.train_model()
                logger.info("Training complete.")
            except Exception as e:
                logger.error(f"Automatic training failed: {e}")
                raise
        
        # 3. Init Components
        logger.info(">>> Initializing Components...")
        sitemap = SitemapManager(config.SITEMAP_PATH)
        vectorizer = SBERTVectorizer()
        ml_model = MLModel(config.MODEL_PATH)
        
        # 4. Run Engine & Pipeline
        engine = UltimateHybridEngine(
            master_df=master_df,
            sitemap_mgr=sitemap,
            vectorizer=vectorizer,
            ml_model=ml_model
        )
        
        pipeline = Pipeline(engine, config)
        pipeline.run_all()
        
        logger.info(">>> [System] All tasks completed successfully.")
        
    except Exception as e:
        logger.error(f"!!! Critical Failure: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()


