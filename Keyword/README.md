# Keyword Engine README

## 개요
`C:\Keyword`는 검색어(`Keyword`)를 제품/브랜드/의도 기준으로 자동 분류하는 하이브리드 엔진입니다.

- 출력 라벨 컬럼: `Division`, `Product`, `Keyword Type`, `Brand`, `Type`
- 처리 방식: 룰 기반 + 임베딩 + ML 추론

## 폴더 구조
- 입력: `C:\Keyword\data\input`
- 출력: `C:\Keyword\data\output`
- 마스터 DB: `C:\Keyword\data\master\검색량.csv`
- 모델/벡터: `C:\Keyword\model`
- 코드: `C:\Keyword\src`, `C:\Keyword\main.py`

## 요구사항
- Python 3.10+ 권장
- 의존성 설치:

```bash
pip install -r C:\Keyword\requirements.txt
```

## 실행 방법
```bash
python C:\Keyword\main.py
```

실행 시 `data/input` 내 `*.csv`를 순차 처리하여 `data/output/RESULT_*.csv`로 저장합니다.

## 입력 형식
- 파일 형식: `.csv`
- 필수 컬럼: `Keyword`
- 인코딩 권장: `utf-8-sig`

## 출력 형식
원본 컬럼 + 아래 분류 결과 컬럼이 추가됩니다.

- `Division`
- `Product`
- `Keyword Type`
- `Brand`
- `Type`
- `Method`
- `Confidence`

## 분류 파이프라인 (Tier)
1. Cache 조회
2. Sitemap 부분 일치 (`samsung_model_master.txt`)
3. Enhanced Rules
4. Regex Rules
5. Master Exact Match
6. SBERT + TF-IDF + RandomForest 추론

`vs` 키워드가 포함되면 `Type=Comparison`으로 후처리됩니다.

## 분류 3단 구조 (마케터 설명용)
1. Hard Rule
- 확실한 검색어를 즉시 분류합니다.
- 사용 기준: Sitemap(삼성 SKU 크롤링), 강화 룰, Regex 룰, 마스터 Exact Match
- 특징: 빠르고 설명이 쉬운 1차 분류

2. SBERT + TF-IDF
- Hard Rule에서 못 잡힌 검색어를 ML 입력 형태로 변환합니다.
- SBERT: 검색어의 의미 유사도를 수치화
- TF-IDF: 눈에 띄는 단어(모델명, 브랜드명, 구매 의도 단어)에 점수를 줌
- 특징: "무슨 뜻인지"(SBERT) + "어떤 단어가 있는지"(TF-IDF)를 함께 반영
- 핵심 포인트: 이 결합 특징은 `Type`(검색의도: Comparison/Research/Sales/Model/Category) 분류에 특히 유리함
- 예시:
  - `galaxy s24 case` -> `case` 단어 점수(TF-IDF)가 높아 액세서리 의도를 강하게 반영
  - `s24 ultra price` / `galaxy s24 ultra cost` -> 표현이 달라도 의미가 유사해 SBERT가 가깝게 인식
  - `samsung refrigerator discount` -> 제품 단어 + 구매 의도 단어를 함께 보고 카테고리/의도를 라벨링

3. Random Forest
- SBERT+TF-IDF 결합 특징을 받아 최종 라벨을 결정합니다.
- 여러 트리가 투표해 `Division/Product/Keyword Type/Brand/Type`를 예측합니다.
- 특징: 규칙만 쓸 때보다 일반화 성능이 좋고 안정적

## 로직 상세 (언제 무엇을 쓰는가)
이 엔진은 "빠르고 확실한 규칙부터 먼저" 처리하고, 남은 건 ML로 보완합니다.

1. Cache
- 같은 키워드 재등장 시 이전 결과를 재사용해 속도를 높입니다.

2. Sitemap 매칭
- 삼성 모델명/제품코드와 부분 일치하면 바로 분류합니다.
- 이 단계에서 잡히면 ML 단계로 가지 않습니다(바이패스).

3. Enhanced/Regex Rules
- 브랜드/모델 패턴, 제품군 패턴 등 명시 규칙으로 분류합니다.
- 규칙에 맞으면 높은 신뢰도로 즉시 확정합니다.

4. Exact Match
- 마스터 DB(`검색량.csv`)의 정규화 키워드와 동일하면 그대로 라벨을 복사합니다.

5. ML 추론 (마지막 단계)
- 위 단계에서 못 잡힌 키워드만 SBERT/TF-IDF/RF를 사용합니다.
- 즉, ML은 전체가 아니라 "애매한 잔여 건" 처리용입니다.

## SBERT / TF-IDF / RandomForest 설명
### 1) SBERT란?
- SBERT(Sentence-BERT)는 문장을 벡터(숫자 표현)로 바꾸는 임베딩 모델입니다.
- 이 프로젝트는 `all-MiniLM-L6-v2`를 사용합니다.
- 장점:
  - 의미가 비슷한 표현을 가까운 벡터로 만듭니다.
  - 오탈자/표현 차이가 있어도 의미 기반 일반화가 가능합니다.

### 2) TF-IDF란?
- TF-IDF는 단어의 중요도를 수치화하는 전통적인 텍스트 특징입니다.
- 희소하지만 식별력 높은 토큰(모델명, 약어, 특정 단어)에 강합니다.
- 장점:
  - 키워드 기반 분류에서 선명한 단어 신호를 잘 잡습니다.

### 3) RandomForest란?
- RandomForest는 여러 결정트리(Decision Tree)를 앙상블한 분류기입니다.
- 입력 특징(SBERT 벡터 + TF-IDF 특징)을 받아 최종 라벨을 예측합니다.
- 장점:
  - 비선형 경계를 잘 다루고, 실무 데이터에서 안정적입니다.
  - 다중 출력(`Division/Product/Keyword Type/Brand/Type`) 예측에 적합합니다.

## 왜 SBERT와 TF-IDF를 같이 쓰나?
- SBERT만 사용 시:
  - 의미 일반화는 좋지만, 일부 정확한 키워드 토큰 신호를 놓칠 수 있습니다.
- TF-IDF만 사용 시:
  - 토큰 매칭은 강하지만, 표현 변형/문맥 일반화가 약합니다.
- 결합 시:
  - 의미 정보(SBERT) + 토큰 정보(TF-IDF)를 동시에 반영해 정확도를 높입니다.

## 추론 흐름 예시
1. 키워드가 모델명 패턴과 맞으면 규칙 단계에서 종료
2. 규칙 불일치 시 SBERT 임베딩 생성
3. TF-IDF 벡터 생성(벡터라이저가 있을 때)
4. 두 벡터를 결합해 RF에 입력
5. `Division/Product/Keyword Type/Brand/Type` + `Confidence` 반환

참고: ML은 `Type`만 단독 예측하지 않고, 위 5개 타깃을 동시에 예측하는 멀티아웃풋 구조입니다.

## Confidence 해석
- 규칙/정확매칭 단계: 보통 `1.0`으로 부여
- ML 단계: 모델 확률(출력별 최대 확률 평균)을 사용
- 운영 권장:
  - 낮은 `Confidence` 구간을 샘플링해 룰/학습데이터를 보강하면 성능이 개선됩니다.

## 핵심 파일
- 실행 진입점: `C:\Keyword\main.py`
- 파이프라인: `C:\Keyword\src\pipeline.py`
- 엔진: `C:\Keyword\src\engine.py`
- 임베딩/ML 래퍼: `C:\Keyword\src\inference.py`
- 설정/경로: `C:\Keyword\src\config.py`

## 모델 관련
- RF 모델 파일: `C:\Keyword\model\rf_classifier.joblib`
- TF-IDF 벡터라이저: `C:\Keyword\model\tfidf_vectorizer.joblib`
- SBERT 모델: `all-MiniLM-L6-v2` (실행 시 로드)
- 모델이 없으면 `main.py`에서 자동 학습 시도(`scripts/train_rf.py`)
- 현재 RF 학습 고정값:
  - `n_estimators=150` (트리 개수)
  - `max_depth=None` (깊이 제한 없음)
  - `class_weight='balanced'` (클래스 불균형 자동 보정)

## 왜 이 설정이 좋은가
- `n_estimators=150`:
  - 트리 수가 너무 적을 때 생기는 분산(예측 흔들림)을 줄이면서,
  - 200~300개 대비 학습/추론 시간 증가를 크게 만들지 않는 균형점입니다.
- `max_depth=None`:
  - 키워드 분류처럼 규칙 경계가 복잡한 문제에서 세밀한 분기 학습이 가능합니다.
  - SBERT+TF-IDF 결합 특징의 비선형 패턴을 충분히 흡수할 수 있습니다.
- `class_weight='balanced'`:
  - 데이터가 많은 클래스에만 치우치지 않도록 자동 보정합니다.
  - 소수 클래스 재현율 개선에 유리합니다.

요약하면, 현재 설정은 정확도 우선 + 실무 처리속도의 균형을 노린 운영값입니다.

## 로그
- 콘솔 로그 + 파일 로그 동시 기록
- 로그 파일: `C:\Keyword\data\output\app.log`

## 트러블슈팅
- `Keyword` 컬럼 누락: 입력 CSV에 `Keyword` 헤더 추가 필요
- 모델 로드 실패: `model` 폴더 내 `rf_classifier.joblib`/`tfidf_vectorizer.joblib` 확인
- 마스터 누락: `data/master/검색량.csv` 경로 확인
- 대용량 처리 지연: `src/pipeline.py`의 `batch_size` 조정 권장

## 참고
- `Keyword_revolution.py`는 단일 스크립트형 레거시 실행 버전입니다.
- 현재 권장 진입점은 모듈형 파이프라인 `main.py`입니다.
