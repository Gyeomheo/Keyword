import os
import re
from pathlib import Path
from typing import Dict, Optional

class SitemapManager:
    def __init__(self, sitemap_path: Path):
        self.sitemap_path = sitemap_path
        # Use model_codes_kb.txt if it exists in the same directory as sitemap_path
        self.kb_path = sitemap_path.parent / "model_codes_kb.txt"
        self.kb: Dict[str, Dict[str, str]] = {}
        self.pattern = None
        self.load_kb()

    def load_kb(self):
        """Loads the key-value knowledge base from file."""
        target_path = self.kb_path if self.kb_path.exists() else self.sitemap_path
        
        if not target_path.exists():
            print(f">>> [Warning] Knowledge Base file not found: {target_path}")
            return
            
        print(f">>> [System] Loading Sitemap KB from {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                # Format: Model \t Division \t Product
                if len(parts) >= 2:
                    model = parts[0].lower()
                    div = parts[1]
                    prod = parts[2] if len(parts) > 2 else "Model" # Default to Model if Product missing
                    
                    if div != "Unknown" and not model.isdigit() and len(model) > 3:
                        self.kb[model] = {'Division': div, 'Product': prod}
        
        print(f">>> [System] Sitemap KB Loaded: {len(self.kb)} entries.")
        self._build_pattern()

    def _build_pattern(self):
        """Builds a regex pattern for all keys in the KB, sorted by length."""
        if not self.kb:
            return
            
        # Sort by length descending to match longest sequences first
        sorted_models = sorted(self.kb.keys(), key=len, reverse=True)
        # Escape for regex
        pattern_str = '|'.join(map(re.escape, sorted_models))
        self.pattern = re.compile(f"({pattern_str})", re.IGNORECASE)
        print(f">>> [System] Sitemap Algo Armed: {len(sorted_models)} models indexed.")

    def search(self, keyword: str) -> Optional[Dict[str, str]]:
        """Searches for a known model in the keyword. Returns dict {Division, Product}."""
        if not self.pattern:
            return None
            
        match = self.pattern.search(keyword)
        if match:
            found_model = match.group().lower()
            return self.kb.get(found_model)
        return None
