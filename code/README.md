# 코드 안내

`01`–`06`은 자료 통합과 유형 분류, `07`–`10`은 대표 사례 분석과 그림을 담은 노트북이다. `plotting.py`는 사례별 시각화, `render_maps.js`는 지도 이미지 렌더링에 사용한다.

Python 환경에서 `pip install -r requirements.txt`로 패키지를 설치하고 Jupyter에서 노트북을 연다. 패키지 버전은 고정하지 않았으며 설치 조합별 실행 여부는 검증하지 않았다. 지도 이미지 생성에는 Node.js와 Playwright 패키지 및 Playwright의 Chromium 설치가 별도로 필요하다 (`npm install playwright`, `npx playwright install chromium`). 그림에는 AppleGothic 글꼴을 사용하므로 다른 환경에서는 사용 가능한 한글 글꼴 설정이 필요하다.

SKT 원자료와 결합된 비공개 시장 특성표 등 입력 자료가 포함되지 않아 전체 분석과 수치를 재계산할 수 없다. 노트북의 자료 경로와 외부 API를 사용하는 단계는 해당 자료와 접근 권한이 필요하다. 기존 결과 그림은 상위 `figs/`에서 확인할 수 있다.
