import re
import os

def build_context_aware_master_v3():
    BASE_DIR = r"C:\Keyword"
    input_file = os.path.join(BASE_DIR, "sitemap_products.xml")
    output_file = os.path.join(BASE_DIR, "samsung_model_master.txt")

    if not os.path.exists(input_file):
        print(f"!!! Error: {input_file} 파일 존재하지 않음")
        return

    with open(input_file, "r", encoding="utf-8", errors='ignore') as f:
        content = f.read()

    urls = re.findall(r'<loc>(.*?)</loc>', content)
    print(f">>> Processing {len(urls)} URLs...")

    master_data = {}

    for url in urls:
        path_list = url.lower().split('/')
        segments = [s for s in path_list if s]
        if len(segments) < 2: continue

        division = "Unknown"
        
        # [1단계: URL 맥락 정밀 분석]
        mx_path = ['mobile', 'tablets', 'watches', 'computing', 'memory-storage', 'smartphones', 'certified-re-newed-store', 'galaxy-book', 'galaxy-buds']
        vd_path = ['televisions', 'home-theater', 'monitors', 'displays', 'pro-tv', 'smart-signage', 'lifestyle-tvs']
        da_path = ['home-appliances', 'refrigerators', 'washers', 'dryers', 'microwaves', 'ranges', 'laundry', 'cooking-appliances', 'vacuums', 'air-dresser']

        if any(x in path_list for x in mx_path): division = "MX"
        elif any(x in path_list for x in vd_path): division = "VD"
        elif any(x in path_list for x in da_path): division = "DA"

        # [모델명 추출 로직]
        last_seg = segments[-1].upper()
        if '-' in last_seg:
            parts = last_seg.split('-')
            if 'SKU' in parts:
                model = parts[parts.index('SKU') + 1]
            else:
                for p in reversed(parts):
                    if len(p) >= 5 and re.search(r'\d', p):
                        model = p
                        break
                else: model = parts[-1]
        else:
            model = last_seg

        # [2단계: 모델명 접두사 기반 강제 할당 (안전장치)]
        if division == "Unknown":
            # MX 계열: 스마트폰, 태블릿, 워치, 저장장치(MB/MZ/MU), 노트북(NP)
            if model.startswith(('SM', 'NP', 'GT', 'SC', 'MB', 'MZ', 'MU', 'SY', 'EJ', 'EF', 'ET', 'EP')):
                division = "MX"
            # VD 계열: TV, 모니터(LS/LC), 사이니지(LH)
            elif model.startswith(('QN', 'UN', 'LS', 'LU', 'LC', 'KQ', 'UA', 'QA', 'LH', 'MN', 'SP')):
                division = "VD"
            # DA 계열: 냉장고(RF/RS), 세탁기(WF/WA), 건조기(DV/DG), 오븐(NV/NQ), 레인지(NX/NS/cc)
            elif model.startswith(('RF', 'RS', 'WF', 'WA', 'DV', 'WA', 'NV', 'NQ', 'NX', 'NS', 'NA', 'ME', 'MS', 'MC', 'RT', 'VR', 'VS', 'DW')):
                division = "DA"

        # [3단계: 최종 데이터 정제]
        if len(model) >= 5 and re.search(r'\d', model) and not model.isdigit():
            # 모델명 뒤에 붙은 불필요한 .HTML이나 쿼리 제거
            model = model.split('?')[0].split('#')[0].replace('.HTML', '').strip('/')
            
            if model not in master_data or master_data[model] == "Unknown":
                master_data[model] = division

    # 결과 저장
    with open(output_file, "w", encoding="utf-8") as f:
        for model, div in sorted(master_data.items()):
            f.write(f"{model}\t{div}\tModel\n")

    print(f"\n>>> [Updated V3] Master List Complete! Total: {len(master_data)}")
    
    # Unknown 통계 확인
    unknown_count = list(master_data.values()).count("Unknown")
    print(f">>> Remaining Unknowns: {unknown_count}")

if __name__ == "__main__":
    build_context_aware_master_v3()