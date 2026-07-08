import re
from typing import Optional, Dict, Tuple

def apply_keyword_rules(keyword_text: str, allowed_brands: list, samsung_identifiers: list) -> Optional[Dict]:
    kw = keyword_text.lower().strip()
    is_comparison = bool(re.search(r'\bvs\b|\bvs\.\b', kw))
    final_brand, final_kw_type = 'Generic', 'Generic'
    
    # Brand Identification
    if any(x in kw for x in samsung_identifiers) or re.search(r'sm-[a-z]\d{3}', kw):
        final_brand, final_kw_type = 'Samsung', 'Samsung'
    else:
        for b in allowed_brands:
            if b == 'Samsung': continue
            if b.lower() in kw:
                final_brand, final_kw_type = b, 'Other Brands'
                break

    # MX Division Logic
    if re.search(r'sm-[a-z]\d{3,4}', kw) or re.search(r'\bnp\d{3}[a-z]', kw) or 'galaxy book' in kw:
        div, prod = 'MX', 'SMP'
        if any(x in kw for x in ['sm-x', 'sm-p', 'tab']): prod = 'Tab'
        elif any(x in kw for x in ['sm-r', 'watch']): prod = 'Watch'
        elif 'buds' in kw: prod = 'TWS'
        elif 'book' in kw or 'np' in kw: prod = 'Laptop'
        return create_res(div, prod, final_kw_type, final_brand, kw, is_comparison)

    # VD Division Logic
    vd_pattern = re.search(r'(qn|qa|ua|kq|un|sc|ls|s|lu|lc)\d{2}[a-z]{1,2}', kw)
    if vd_pattern:
        prefix = vd_pattern.group(1)
        div = 'VD'
        if prefix in ['qn', 'kq']: prod = 'QLED TV'
        elif prefix in ['sc', 's'] and re.search(r's[8-9]\d[a-z]', kw): prod = 'OLED TV'
        elif prefix in ['lu', 'lc', 'ls'] and 'monitor' in kw: prod = 'Monitor'
        else: prod = 'TV'
        return create_res(div, prod, final_kw_type, final_brand, kw, is_comparison)

    # DA Division Logic
    if re.search(r'\brf\d{2}|\brs\d{2}', kw) or 'refrigerator' in kw:
        return create_res('DA', 'REF', final_kw_type, final_brand, kw, is_comparison)
    if re.search(r'\bwf\d{2}|\bdv\d{2}|\bwa\d{2}', kw) or 'washer' in kw or 'dryer' in kw:
        return create_res('DA', 'Laundry', final_kw_type, final_brand, kw, is_comparison)
        
    return None

def create_res(div: str, prod: str, kw_type: str, brand: str, kw: str, is_comparison: bool) -> Dict:
    if is_comparison: 
        final_type = 'Comparison'
    else:
        if any(x in kw for x in ['review', 'test', 'specs']): final_type = 'Research'
        elif any(x in kw for x in ['buy', 'price', 'sale']): final_type = 'Sales'
        elif any(x in kw for x in ['s24', 's23', 'iphone', 'sm-', 'qn', 'rf']): final_type = 'Model'
        else: final_type = 'Category'
        
    return {
        'Division': div, 
        'Product': prod, 
        'Keyword Type': kw_type, 
        'Brand': brand, 
        'Type': final_type, 
        'Method': 'Hard Rule (Regex)'
    }
