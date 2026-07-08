import xml.etree.ElementTree as ET
from pathlib import Path
import re

def build_kb():
    # Setup Paths
    BASE_DIR = Path("C:/Keyword")
    SITEMAP_PATH = BASE_DIR / "sitemap_products.xml"
    OUTPUT_MODEL_KB = BASE_DIR / "model_codes_kb.txt"
    OUTPUT_CAT_KB = BASE_DIR / "category_keywords_kb.txt"
    
    if not SITEMAP_PATH.exists():
        print(f"Error: {SITEMAP_PATH} not found.")
        return

    print("Parsing sitemap...")
    try:
        tree = ET.parse(SITEMAP_PATH)
        root = tree.getroot()
        # Handle namespaces if any (sitemaps usually have them)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [elem.text for elem in root.findall('.//ns:loc', ns)]
        if not urls: # Try without namespace
             urls = [elem.text for elem in root.findall('.//loc')]
    except Exception as e:
        print(f"XML Parse Error: {e}")
        return

    print(f"Found {len(urls)} URLs. Extracting patterns...")
    
    model_kb = {} # model -> (Division, Product)
    
    # regex for model codes
    # QN55S95D -> matches [A-Z0-9]+
    # Typical Samsung Model: start with letter, end with letter/number. 
    # Extract from last segment of URL: '.../qn55s95dafxza'
    
    for url in urls:
        if not url: continue
        parts = url.strip('/').split('/')
        if len(parts) < 2: continue
        
        last_seg = parts[-1].lower()
        
        # Division/Product logic based on URL segments
        div = "Unknown"
        prod = "Model"
        
        # Determine based on path
        if 'mobile' in parts or 'smartphones' in parts:
            div = 'MX'
            prod = 'SMP'
            if 'galaxy-s' in url: prod = 'SMP'
            if 'galaxy-z' in url: prod = 'SMP'
            if 'tablets' in url: prod = 'Tab'
            if 'watches' in url: prod = 'Watch'
            if 'audio' in url or 'buds' in url: prod = 'TWS'
            
        elif 'tvs' in parts or 'tv' in parts:
            div = 'VD'
            prod = 'TV'
            if 'oled' in url or 'oled-tv' in url: prod = 'OLED TV'
            elif 'qled' in url or 'neo-qled' in url: prod = 'QLED TV'
            elif 'lifestyle' in url: prod = 'Lifestyle TV'
            
        elif 'home-appliances' in parts:
            div = 'DA'
            if 'refrigerators' in parts: prod = 'REF'
            elif 'washers' in parts or 'dryers' in parts or 'laundry' in parts: prod = 'Laundry'
            elif 'vacuums' in parts: prod = 'Vacuum'
            
        elif 'monitors' in parts:
            div = 'VD'
            prod = 'Monitor'
            if 'gaming' in url or 'odyssey' in url: prod = 'Gaming Monitor'
            
        # Extract Model Code
        # Attempt to split regex matches or use the whole segment if it looks like a model code
        # Example: qn55s95dafxza
        # Logic: If it contains digits and letters, length > 5
        if re.search(r'\d', last_seg) and re.search(r'[a-z]', last_seg) and len(last_seg) > 5:
            # Clean model code (remove options sometimes?)
            # Usually strict mapping is better.
            model_kb[last_seg] = (div, prod)
            
            # Also try to extract base model (e.g. qn55s95d from qn55s95dafxza)
            # Samsung pattern: 
            # TV: simple truncate might work? No, too risky.
            # Just keep the full code and also try to capture extracting common patterns if possible.
            # Let's stick to the exact segment for now, sitemap usually has the full SKU.
            
            # Special handling for queries like "qn55s95d" (missing suffix)
            # We can try to generate a prefix key if it's long enough
            if div == 'VD' and len(last_seg) > 8:
                 short_code = last_seg[:8] # simplistic shortening
                 if short_code not in model_kb:
                     model_kb[short_code] = (div, prod)
            
    # Write to file
    with open(OUTPUT_MODEL_KB, 'w', encoding='utf-8') as f:
        for model, (div, prod) in model_kb.items():
            f.write(f"{model}\t{div}\t{prod}\n")
            
    print(f"Generated {len(model_kb)} entries in {OUTPUT_MODEL_KB}")

if __name__ == "__main__":
    build_kb()
