"""
Enhanced Hard Mapping Rules - Contains-Based Matching + Context Aware
Based on actual master data keyword patterns and Sitemap Analysis.

키워드가 특정 패턴을 **포함**하면 바로 분류 (부분 일치)
"""
import re
from typing import Optional, Dict, List, Tuple

# ============================================================
# 정규화 함수 - 띄어쓰기/하이픈 변형을 통일
# ============================================================
def normalize_keyword(kw: str) -> Tuple[str, str]:
    """
    키워드 정규화: 띄어쓰기, 하이픈 등 변형을 통일
    'i phone', 'i-phone', 'iPhone' -> 'iphone'
    """
    kw = kw.lower().strip()
    kw = re.sub(r'[-_]', ' ', kw)  # 하이픈/언더스코어 -> 공백
    kw = re.sub(r'\s+', ' ', kw)   # 다중 공백 -> 단일 공백
    kw_nospace = kw.replace(' ', '')
    return kw, kw_nospace

# ============================================================
# Brand Detection (키워드에 포함되면 해당 브랜드로 분류)
# ============================================================
BRAND_CONTAINS = {
    'Samsung': ['samsung', 'galaxy', 'bespoke'],
    'Apple': ['apple', 'iphone', 'i phone', 'ipad', 'airpods', 'macbook'],
    'LG': ['lg '],
    'Sony': ['sony', 'playstation'],
    'Dyson': ['dyson'],
    'Shark': ['shark'],
    'Bose': ['bose'],
    'Xiaomi' :['xiaomi']
    # ... (Add more as needed)
}

# ============================================================
# Division + Product Detection (contains 기반)
# 순서 중요: 더 구체적인 패턴이 먼저 와야 함
# ============================================================
PRODUCT_CONTAINS = [
    # ===== MX Division (Smartphones/Tablets/Wearables) =====
    # Smartphone (Samsung)
    (['galaxy s25', 's25 ultra', 's25+', 's25 plus', 's25'], 'MX', 'SMP', 'Samsung'), # Added 's25'
    (['galaxy s24', 's24 ultra', 's24+', 's24 plus', 's24 fe', 's24'], 'MX', 'SMP', 'Samsung'),
    (['galaxy s23', 's23 ultra', 's23+', 's23 plus', 's23 fe', 's23'], 'MX', 'SMP', 'Samsung'),
    (['galaxy s22', 's22 ultra', 's22+', 's22 plus', 's22'], 'MX', 'SMP', 'Samsung'),
    (['galaxy s21', 's21 ultra', 's21+', 's21 plus', 's21 fe', 's21'], 'MX', 'SMP', 'Samsung'),
    (['galaxy z fold', 'z fold7', 'z fold6', 'z fold5', 'z fold4', 'z fold3', 'z fold2', 'z fold', 'fold7', 'fold6', 'fold5', 'fold4', 'fold3', 'fold2', 'fold 7', 'fold 6', 'fold 5', 'fold 4', 'fold 3', 'fold fe'], 'MX', 'SMP', 'Samsung'),
    (['galaxy z flip', 'z flip7', 'z flip6', 'z flip5', 'z flip4', 'z flip3', 'z flip', 'flip7', 'flip6', 'flip5', 'flip4', 'flip3', 'flip 7', 'flip 6', 'flip 5', 'flip 4', 'flip 3', 'flip fe', 'flip7 fe', 'flip6 fe'], 'MX', 'SMP', 'Samsung'),
    (['galaxy note', 'note 20', 'note 10'], 'MX', 'SMP', 'Samsung'),
    (['galaxy a', 'a55', 'a54', 'a35', 'a25', 'a15'], 'MX', 'SMP', 'Samsung'),
    (['galaxy xcover'], 'MX', 'SMP', 'Samsung'),
    
    # Smartphone (Competitors)
    (['iphone 16', 'iphone 15', 'iphone 14', 'iphone 13', 'iphone se', 'iphone'], 'MX', 'SMP', 'Apple'),
    (['pixel 9', 'pixel 8', 'pixel 7', 'pixel fold', 'google pixel'], 'MX', 'SMP', 'Google'),
    (['oneplus 12', 'oneplus 11', 'oneplus open'], 'MX', 'SMP', 'OnePlus'),
    (['motorola', 'moto g', 'razr'], 'MX', 'SMP', 'Motorola'),
    
    # Smartphone (Generic)
    (['smartphone', 'cell phone', 'android phone', 'mobile phone', 'phone', 'phones'], 'MX', 'SMP', None),

    # Tablets
    (['galaxy tab', 'tab s10', 'tab s9', 'tab s8', 'tab a9', 'tab a8'], 'MX', 'Tab', 'Samsung'),
    (['ipad pro', 'ipad air', 'ipad mini', 'ipad'], 'MX', 'Tab', 'Apple'),
    (['tablet'], 'MX', 'Tab', None),

    # Wearables (Watch/Buds)
    (['galaxy watch', 'watch 7', 'watch ultra', 'watch 6', 'watch 5'], 'MX', 'Watch', 'Samsung'),
    (['apple watch', 'watch series', 'watch ultra'], 'MX', 'Watch', 'Apple'),
    (['pixel watch'], 'MX', 'Watch', 'Google'),
    (['smartwatch'], 'MX', 'Watch', None),
    
    (['galaxy ring'], 'MX', 'Ring', 'Samsung'), # New Category
    
    (['galaxy buds', 'buds 3', 'buds 2', 'buds pro', 'buds live', 'buds fe'], 'MX', 'TWS', 'Samsung'),
    (['airpods pro', 'airpods', 'airpods max'], 'MX', 'TWS', 'Apple'),
    (['pixel buds'], 'MX', 'TWS', 'Google'),
    (['earbuds', 'headphones', 'wireless earphones'], 'MX', 'TWS', None),
    
    # Laptops
    (['galaxy book', 'book4', 'book3', 'book2'], 'MX', 'Laptop', 'Samsung'),
    (['macbook pro', 'macbook air', 'macbook'], 'MX', 'Laptop', 'Apple'),
    (['laptop', 'notebook'], 'MX', 'Laptop', None),
    
    # ===== VD Division (TV/Monitor) =====
    # Data-Driven Signatures from Sitemap
    (['odyssey', 'gaming monitor', 'curved monitor', 'wqhd', 'qhd'], 'VD', 'Gaming Monitor', 'Samsung'),
    (['samsung monitor', 'smart monitor'], 'VD', 'Monitor', 'Samsung'),
    (['monitor', 'display'], 'VD', 'Monitor', None),
    
    (['neo qled', 'qn90', 'qn85'], 'VD', 'QLED TV', 'Samsung'),
    (['oled tv', 's95', 's90'], 'VD', 'OLED TV', 'Samsung'),
    (['television', 'theater', 'soundbar'], 'VD', 'TV', None), # Generic TV terms
    
    # ===== DA Division (Appliances) =====
    # Signatures
    (['bespoke fridge', 'flex refrigerator', 'french door'], 'DA', 'REF', 'Samsung'),
    (['refrigerator', 'fridge', 'freezer'], 'DA', 'REF', None),
    
    (['washing machine', 'washer', 'laundry', 'activewash'], 'DA', 'Laundry', None),
    (['dryer', 'tumble dryer'], 'DA', 'Laundry', None),
    
    (['jet bot', 'bespoke jet', 'clean station', 'stick vacuum'], 'DA', 'Vacuum', 'Samsung'),
    (['vacuum', 'hoover', 'robot vacuum'], 'DA', 'Vacuum', None),
    
    # Samsung Specific Dishwasher Tech (Verified from Sitemap) -> Force Samsung
    (['stormwash', 'waterwall'], 'DA', 'Dishwasher', 'Samsung'), 
    (['dishwasher', 'rotary'], 'DA', 'Dishwasher', None),
    
    (['microwave', 'over-the-range'], 'DA', 'Microwave', None),
    (['induction', 'freestanding range', 'gas range', 'cooktop'], 'DA', 'Cooking', None),
]

# ============================================================
# 모델코드 정규표현식 패턴 (Prefix Analysis Result)
# ============================================================
MODEL_CODE_PATTERNS = [
    # ===== MX Division (Smartphones/Tablets/Wearables) =====
    # Galaxy S Series: SM-S9XX (S2x), SM-S7XX (older)
    (r'sm-s9\d{2}', 'MX', 'SMP', 'Samsung'),   # SM-S928 = S25, SM-S918 = S24
    (r'sm-s7\d{2}', 'MX', 'SMP', 'Samsung'),   # Older S series
    
    # Galaxy Z Fold/Flip: SM-F9XX
    (r'sm-f9\d{2}', 'MX', 'SMP', 'Samsung'),   # SM-F956 = Fold6, SM-F741 = Flip6
    (r'sm-f7\d{2}', 'MX', 'SMP', 'Samsung'),   # Older Flip
    
    # Galaxy A Series: SM-AXXX
    (r'sm-a\d{3}', 'MX', 'SMP', 'Samsung'),    # SM-A556 = A55, SM-A356 = A35
    
    # Galaxy Note: SM-N9XX
    (r'sm-n9\d{2}', 'MX', 'SMP', 'Samsung'),   # SM-N986 = Note20
    
    # Galaxy Tab: SM-X, SM-T
    (r'sm-x\d{3}', 'MX', 'Tab', 'Samsung'),    # SM-X910 = Tab S9 Ultra
    (r'sm-t\d{3}', 'MX', 'Tab', 'Samsung'),    # Older tablets
    
    # Galaxy Watch: SM-R
    (r'sm-r\d{3}', 'MX', 'Watch', 'Samsung'),  # SM-R960 = Watch7
    (r'sm-l\d{3}', 'MX', 'Watch', 'Samsung'),  # LTE variants
    
    # Galaxy Ring: SM-Q
    (r'sm-q\d{3}', 'MX', 'Ring', 'Samsung'),   # SM-Q505 = Galaxy Ring
    
    # Galaxy Buds: SM-R (overlaps watch, but buds have specific codes)
    (r'sm-r5\d{2}', 'MX', 'TWS', 'Samsung'),   # Buds series (R510, R530...)
    
    # Galaxy Book: NP (Notebook PC)
    (r'np\d{3}[a-z]', 'MX', 'Laptop', 'Samsung'), # NP960 = Book4
    
    # ===== VD Division (TV/Monitor) =====
    # Neo QLED / QLED TV
    (r'qn\d{2}[a-z]', 'VD', 'QLED TV', 'Samsung'),   # QN90D, QN85D
    (r'qe\d{2}[a-z]', 'VD', 'QLED TV', 'Samsung'),   # QLED entry
    (r'qa\d{2}[a-z]', 'VD', 'QLED TV', 'Samsung'),   # Asia models
    
    # Crystal UHD / UHD TV
    (r'un\d{2}[a-z]', 'VD', 'UHD TV', 'Samsung'),    # UN65DU8000
    (r'ua\d{2}[a-z]', 'VD', 'UHD TV', 'Samsung'),    # Asia UHD
    (r'cu\d{2}[a-z]', 'VD', 'UHD TV', 'Samsung'),    # Crystal UHD
    (r'du\d{2}[a-z]', 'VD', 'UHD TV', 'Samsung'),    # DU series
    (r'tu\d{2}[a-z]', 'VD', 'UHD TV', 'Samsung'),    # TU series (older)
    
    # OLED TV
    (r's9\d[a-z]', 'VD', 'OLED TV', 'Samsung'),      # S95D, S90D
    (r's8\d[a-z]', 'VD', 'OLED TV', 'Samsung'),      # S85D
    
    # The Frame / Lifestyle
    (r'ls\d{2}[a-z]', 'VD', 'Lifestyle TV', 'Samsung'), # LS03D = Frame
    
    # Monitor
    (r'lc\d{2}[a-z]', 'VD', 'Monitor', 'Samsung'),   # LC = Curved Monitor
    (r'lf\d{2}[a-z]', 'VD', 'Monitor', 'Samsung'),   # LF = Flat Monitor
    (r'lu\d{2}[a-z]', 'VD', 'Monitor', 'Samsung'),   # LU = UHD Monitor
    (r'cf\d{2}[a-z]', 'VD', 'Monitor', 'Samsung'),   # CF = Curved (older)
    
    # Soundbar
    (r'hw-[a-z]\d{2}', 'VD', 'Soundbar', 'Samsung'), # HW-Q990D, HW-S800D
    
    # ===== DA Division (Appliances) =====
    # Refrigerator
    (r'rf\d{2}[a-z]', 'DA', 'REF', 'Samsung'),       # RF29 = French Door
    (r'rs\d{2}[a-z]', 'DA', 'REF', 'Samsung'),       # RS28 = Side-by-Side
    (r'rb\d{2}[a-z]', 'DA', 'REF', 'Samsung'),       # RB = Bespoke
    (r'rz\d{2}[a-z]', 'DA', 'REF', 'Samsung'),       # RZ = Freezer
    
    # Laundry (Washer/Dryer)
    (r'wf\d{2}[a-z]', 'DA', 'Laundry', 'Samsung'),   # WF53 = Front Load
    (r'wa\d{2}[a-z]', 'DA', 'Laundry', 'Samsung'),   # WA54 = Top Load
    (r'wd\d{2}[a-z]', 'DA', 'Laundry', 'Samsung'),   # WD = Combo
    (r'dv\d{2}[a-z]?', 'DA', 'Laundry', 'Samsung'),  # DV = Dryer
    (r'dve\d{2}', 'DA', 'Laundry', 'Samsung'),       # DVE = Electric Dryer
    (r'dvg\d{2}', 'DA', 'Laundry', 'Samsung'),       # DVG = Gas Dryer
    
    # Dishwasher
    (r'dw\d{2}[a-z]?', 'DA', 'Dishwasher', 'Samsung'), # DW80 = Dishwasher
    
    # Microwave
    (r'me\d{2}[a-z]?', 'DA', 'Microwave', 'Samsung'), # ME21 = Over-the-Range
    (r'ms\d{2}[a-z]?', 'DA', 'Microwave', 'Samsung'), # MS = Countertop
    
    # Range/Cooktop
    (r'nx\d{2}[a-z]?', 'DA', 'Cooking', 'Samsung'),   # NX = Gas Range
    (r'ne\d{2}[a-z]?', 'DA', 'Cooking', 'Samsung'),   # NE = Electric Range
    (r'nz\d{2}[a-z]?', 'DA', 'Cooking', 'Samsung'),   # NZ = Induction Cooktop
    (r'nv\d{2}[a-z]?', 'DA', 'Cooking', 'Samsung'),   # NV = Wall Oven
    (r'nq\d{2}[a-z]?', 'DA', 'Cooking', 'Samsung'),   # NQ = Microwave Combo Oven
    
    # Vacuum
    (r'vr\d{2}[a-z]?', 'DA', 'Vacuum', 'Samsung'),    # VR = Jet Bot Robot
    (r'vs\d{2}[a-z]?', 'DA', 'Vacuum', 'Samsung'),    # VS = Jet Stick
]

TYPE_PATTERNS = {
    'Comparison': ['vs', 'versus', 'compare'],
    'Research': ['review', 'specs', 'best', 'newest', 'latest'],
    'Sales': ['price', 'buy', 'sale', 'deal', 'cost'],
    'Model': [],
}

def detect_brand(keyword: str) -> Tuple[str, str]:
    kw_spaced, kw_nospace = normalize_keyword(keyword)
    for brand, patterns in BRAND_CONTAINS.items():
        for ptn in patterns:
            ptn_nospace = ptn.replace(' ', '')
            if ptn in kw_spaced or ptn_nospace in kw_nospace:
                return brand, 'Samsung' if brand == 'Samsung' else 'Other Brands'
    return 'Generic', 'Generic'

def detect_product(keyword: str) -> Optional[Tuple[str, str, Optional[str]]]:
    kw_spaced, kw_nospace = normalize_keyword(keyword)
    # 1. Pattern Match
    for patterns, division, product, brand in PRODUCT_CONTAINS:
        for ptn in patterns:
            ptn_nospace = ptn.replace(' ', '')
            if ptn in kw_spaced or ptn_nospace in kw_nospace:
                return (division, product, brand)
    # 2. Regex Match
    for regex, division, product, brand in MODEL_CODE_PATTERNS:
        if re.search(regex, kw_spaced, re.IGNORECASE):
            return (division, product, brand)
    return None

def detect_type(keyword: str, has_model_code: bool) -> str:
    kw = keyword.lower()
    if 'vs' in kw: return 'Comparison'
    
    for type_name, patterns in TYPE_PATTERNS.items():
        for ptn in patterns:
            if ptn in kw: return type_name
            
    return 'Model' if has_model_code else 'Category'

def apply_enhanced_rules(keyword: str) -> Optional[Dict]:
    kw = keyword.lower().strip()
    
    # 1. Product/Division
    product_result = detect_product(kw)
    if product_result is None: return None
    
    div, prod, detected_brand = product_result
    
    # 2. Brand (Overlap Check)
    brand, kw_type = detect_brand(kw)
    if detected_brand: 
        brand = detected_brand
        kw_type = 'Samsung' if brand == 'Samsung' else 'Other Brands'
        
    # 3. Type
    has_model_code = any(re.search(ptn, kw) for ptn, _, _, _ in MODEL_CODE_PATTERNS)
    final_type = detect_type(kw, has_model_code)
    
    return {
        'Division': div, 'Product': prod, 'Keyword Type': kw_type, 
        'Brand': brand, 'Type': final_type, 'Method': 'Enhanced Hard Rule'
    }
