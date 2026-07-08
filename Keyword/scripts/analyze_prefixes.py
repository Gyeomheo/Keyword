import xml.etree.ElementTree as ET
import re
from collections import defaultdict
import os

def analyze_sitemap(path):
    print(f"Analyzing {path}...", flush=True)
    if not os.path.exists(path):
        print(f"File not found: {path}", flush=True)
        return

    urls = []
    # Count lines for progress check
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if '<loc>' in line:
                    url = line.strip().replace('<loc>', '').replace('</loc>', '')
                    urls.append(url)
    except Exception as e:
        print(f"Error reading file: {e}", flush=True)
        return

    print(f"Found {len(urls)} URLs. Processing...", flush=True)

    word_freq = defaultdict(lambda: defaultdict(int))
    prefix_map = defaultdict(lambda: defaultdict(int))
    
    # Common stop words in URLs to ignore
    STOP_WORDS = {'samsung', 'us', 'com', 'www', 'https', 'http', 'and', 'with', 'in', 'the', 'for', 'of', 'to', 'at', 'on', 'a', 'black', 'white', 'silver', 'gray', 'grey', 'stainless', 'steel'}

    for url in urls:
        # 1. Extract Category (Same logic as before)
        category = "Unknown"
        if '/tvs/' in url or 'televisions' in url: category = 'TV'
        elif '/monitors/' in url: category = 'Monitor'
        elif '/refrigerators/' in url: category = 'Refrigerator'
        elif '/washers/' in url: category = 'Washer'
        elif '/dryers/' in url: category = 'Dryer'
        elif '/ranges/' in url: category = 'Range'
        elif '/microwaves/' in url: category = 'Microwave'
        elif '/dishwashers/' in url: category = 'Dishwasher'
        elif '/tablets/' in url: category = 'Tablet'
        elif '/phones/' in url: category = 'Smartphone'
        elif '/laptops/' in url or '/galaxy-books/' in url: category = 'Laptop'
        elif '/sound-bars/' in url: category = 'Soundbar'
        elif '/vacuums/' in url: category = 'Vacuum'
        
        if category == 'Unknown': continue

        # 2. Extract Keywords from URL path
        # Split by '/' and '-'
        clean_url = url.split('?')[0] # Remove query params
        parts = re.split(r'[/-]', clean_url)
        
        for part in parts:
            word = part.lower().strip()
            if not word or word.isdigit() or len(word) < 3: continue
            if word in STOP_WORDS: continue
            
            # Check if it looks like a model code (has numbers)
            if re.search(r'\d', word):
                 # Model Code Prefix Analysis
                match = re.search(r'([a-z]{2,3}\d{2,}[a-z0-9]*)', word, re.IGNORECASE)
                if match:
                    code = match.group(1).upper()
                    if len(code) >= 4:
                        prefix_map[category][code[:2]] += 1
            else:
                # Pure text keyword
                word_freq[category][word] += 1

    print("\n=== 1. Top Keywords per Category (Potential Rule Triggers) ===")
    for cat, freqs in sorted(word_freq.items()):
        print(f"\n[{cat}]")
        sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        # Show top 10 unique words
        top_words = [f"{w}({c})" for w, c in sorted_words[:15]]
        print(f"  Keywords: {', '.join(top_words)}")

    print("\n=== 2. Dictionary of Model Prefixes per Category ===")
    for cat, prefixes in sorted(prefix_map.items()):
        print(f"\n[{cat}]")
        sorted_prefixes = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)
        top_prefixes = [f"{p}({c})" for p, c in sorted_prefixes if c > 5]
        print(f"  Prefixes: {', '.join(top_prefixes)}")

if __name__ == "__main__":
    analyze_sitemap(r"C:\Keyword\sitemap_products.xml")
