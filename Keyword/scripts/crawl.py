import re
import os

def extract_next_sitemap_targets():
    # 1. 파일 경로 설정
    BASE_DIR = r"C:\Keyword"
    input_file = os.path.join(BASE_DIR, "sitemap_main.xml") 
    
    if not os.path.exists(input_file):
        print(f"!!! Error: {input_file} 파일이 없네. 메모장에 붙여넣고 저장부터 하게.")
        return

    # 2. 파일 읽기
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 3. 정규표현식으로 <loc> 태그 안의 URL만 추출
    # 네임스페이스(sitemap:)가 있든 없든 상관없이 잡아내네.
    urls = re.findall(r'<loc>(.*?)</loc>', content)
    print(f">>> Found {len(urls)} URLs in the index.")

    # 4. 우리가 찾는 'Product Detail (pd)' 타겟 필터링
    targets = [url for url in urls if 'pd-b2c' in url]

    if targets:
        print("\n>>> [Target Found!] 이 주소들이 진짜 보물지도라네:")
        for t in targets:
            print(f" - {t}")
        
        # 다음 단계를 위해 저장
        with open(os.path.join(BASE_DIR, "next_targets.txt"), "w") as f:
            for t in targets:
                f.write(t + "\n")
    else:
        print("!!! 'pd-b2c' 패턴을 못 찾았네. 파일 내용을 다시 확인해보게.")

if __name__ == "__main__":
    extract_next_sitemap_targets()