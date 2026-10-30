"""전통시장 유형과 대표 사례의 공개 근거 그림을 그리는 공통 함수.

입력 그림에는 공개 요약값이 사용된다. 원시 OD/STAY와 결합 특성표는
이 모듈에 포함되지 않으며, 비공개 입력이 필요한 지도 함수는 경로를 받는다.
"""

import html
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import folium
    from branca.element import Element
except ImportError:
    folium = None
    Element = None

ROOT = Path(__file__).resolve().parent.parent
FIGS = Path(os.environ.get("BC_FIG_DIR", ROOT / "figs"))
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
INK, FOCUS, REF = "#243447", "#176b87", "#aebcc7"
BLUE, PALE, GOLD = "#176b87", "#afc7d0", "#c68c40"
ORANGE, GRAY = "#c58b38", "#9ca9b4"
DAYS = ["월", "화", "수", "목", "금", "토", "일"]

def _case_comparison(path, title, panels, note):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.25 * len(panels), 3.8))
    fig.subplots_adjust(left=.075, right=.99, bottom=.19, top=.75, wspace=.33)
    for ax, (label, names, values, unit) in zip(axes, panels):
        bars = ax.barh(range(len(names)), values, color=[FOCUS] + [REF] * (len(names) - 1), height=.55)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.30 if max(values) else 1)
        ax.set_title(label, color=INK, fontsize=12, loc="left", pad=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        for bar, value in zip(bars, values):
            shown = f"{value:.2f}" if unit == "배" else f"{value:.1f}"
            ax.text(bar.get_width() + max(values) * .025, bar.get_y() + bar.get_height() / 2,
                    f"{shown}{unit}", va="center", fontsize=10, color=INK)
    fig.suptitle(title, x=.02, y=.97, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(.02, .045, note, fontsize=8.5, color="#5c6974")
    fig.savefig(FIGS / path, dpi=175, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)

CASE_COMPARISONS = {'구룡포시장': ['fig_v4_1_guryongpo.png',
           '외곽형 대표 · 구룡포시장',
           [('여행 목적 도착 OD', ['구룡포', '외곽형 평균', '전국 평균'], [35.6, 22.4, 8.2], '%'),
            ('여행 목적 체류 STAY', ['구룡포', '외곽형 평균', '전국 평균'], [15.3, 10.8, 5.5], '%'),
            ('쇼핑 OD 출발지 10km 이상', ['구룡포', '외곽형 평균', '전국 평균'], [80.5, 71.2, 29.3], '%'),
            ('주말/평일 일평균 도착', ['구룡포', '외곽형 평균', '전국 평균'], [2.33, 1.54, 1.03], '배')],
           '일반일 38일 · OD·STAY 별도 집계 · 출발지 거리는 행정동 중심점 간 직선거리'],
 '중마시장': ['fig_v4_2_jungma.png',
          '독립 근린형 대표 · 중마시장',
          [('쇼핑 OD 출발지 10km 안', ['중마', '전국 평균'], [89.6, 70.7], '%'),
           ('귀가 목적 도착 OD', ['중마', '전국 평균'], [62.4, 55.7], '%'),
           ('반경 500m 버스정류장', ['중마', '전국 평균'], [41, 19.9], '곳'),
           ('반경 500m 공영주차장', ['중마', '전국 평균'], [11, 5.3], '곳')],
          '일반일 38일 · 행정동 OD · 정류장·공영주차장 수는 공개 위치 자료의 반경 집계'],
 '신림 시장권': ['fig_v4_3_sillim.png',
            '주거 귀가형 대표 · 신림 시장권',
            [('귀가 목적 도착 OD', ['신림', '주거 귀가형 평균', '전국 평균'], [84.1, 74.4, 55.7], '%'),
             ('평일 20·30대 19–24시 OD', ['신림', '전국 평균'], [51.0, 28.9], '%'),
             ('쇼핑 OD 출발지 5km 안', ['신림', '전국 평균'], [86.8, 52.1], '%')],
            '신림동 소재 시장 4곳이 같은 행정동을 공유 · 일반일 38일'],
 '국제시장권': ['fig_v4_4_international.png',
           '쇼핑 밀집형 대표 · 국제시장권',
           [('쇼핑여가 목적 도착 OD', ['국제시장권', '전국 평균'], [40.7, 5.9], '%'),
            ('주말/평일 일평균 도착', ['국제시장권', '전국 평균'], [1.66, 1.03], '배'),
            ('해당 시장 행정동 도착 비중', ['국제시장권', '전국 평균'], [22.7, 16.3], '%')],
           '광복동 시장권 · 5km 안 출발동의 관측 쇼핑 OD 중 시장 행정동 도착 비중 · 일반일 38일']}

def plot_case_comparison(case):
    """저장된 공개 요약값으로 사례와 전국·유형 평균을 비교한다."""
    if case not in CASE_COMPARISONS:
        raise KeyError(case)
    return _case_comparison(*CASE_COMPARISONS[case])


def plot_type_counts(counts=None):
    """유형별 시장 수 그림을 저장한다."""
    counts = counts or {"외곽형": 204, "독립 근린형": 255,
                        "주거 귀가형": 266, "쇼핑 밀집형": 105}
    names = list(counts)
    values = list(counts.values())
    fig, ax = plt.subplots(figsize=(10.5, 3.7), constrained_layout=True)
    bars = ax.barh(names, values, color=["#8eacc0", "#50a3a2", "#176b87", "#c99d5b"], height=.6)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.24)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.set_xlabel(f"시장 수 (총 {sum(values)}곳)")
    for bar, value in zip(bars, values):
        ax.text(value + 7, bar.get_y() + bar.get_height() / 2, f"{value}곳", va="center", fontsize=11)
    ax.set_title("시장 유형 분포", loc="left", fontsize=15, fontweight="bold", color=INK)
    fig.savefig(FIGS / "fig_v4_0_type_counts.png", dpi=175, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)

def _stay_comparison(filename, title, panels, note):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    fig.subplots_adjust(left=.10, right=.98, bottom=.23, top=.75, wspace=.38)
    for ax, (label, case, case_value, national_value) in zip(axes, panels):
        values = [case_value, national_value]
        bars = ax.barh([0, 1], values, color=[FOCUS, REF], height=.55)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([case, "전국 평균"])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.32)
        ax.set_title(label, loc="left", fontsize=12, color=INK, pad=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + max(values) * .025,
                    bar.get_y() + bar.get_height() / 2, f"{value:.1f}%",
                    va="center", fontsize=10, color=INK)
    fig.suptitle(title, x=.02, y=.97, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(.02, .055, note, fontsize=8.5, color="#5c6974")
    fig.savefig(FIGS / filename, dpi=175, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)

STAY_CASES = {
    "중마시장": ("fig_v4_2b_jungma_stay.png", "중마시장 · 목적별 체류 구성",
               "일반일 38일 · 행정동 체류 인원 시간별 합의 목적 비율 · 전국은 830개 시장 행 평균"),
    "신림 시장권": ("fig_v4_3c_sillim_stay.png", "신림 시장권 · 연령과 목적별 체류 구성",
                   "신림동의 시장 4곳이 같은 행정동 값을 공유 · 일반일 38일 · 전국은 830개 시장 행 평균"),
    "국제시장권": ("fig_v4_4c_international_stay.png", "국제시장권 · 쇼핑 관련 체류 구성",
                 "광복동의 시장 4곳이 같은 행정동 값을 공유 · 일반일 38일 · 60대 비교는 유효 813시장 행 평균"),
}

def plot_stay_case(case):
    """공개 STAY 요약표의 사례별 두 지표를 비교한다."""
    filename, title, note = STAY_CASES[case]
    rows = STAY.loc[STAY["case"].eq(case)].to_dict(orient="records")
    if len(rows) != 2:
        raise ValueError(f"{case}: STAY 요약 지표 2개 필요")
    panels = [(row["metric"], row["case_label"], float(row["case_share"]) * 100,
               float(row["national_market_row_mean"]) * 100) for row in rows]
    return _stay_comparison(filename, title, panels, note)

# 공개 가능한 저장 요약값. 비공개 OD/STAY 원자료의 재집계값은 아니다.
STAY = pd.DataFrame([{'case': '중마시장',
  'case_label': '중마',
  'metric': '귀가 목적 STAY',
  'case_share': 0.7772841059988385,
  'national_market_row_mean': 0.6503731192832142,
  'national_valid_market_rows': 830,
  'source_snapshot': 'a1_stay_purpose_mix.csv',
  'scope': '2023-09-01~2023-10-15 중 일반일 38일; 전목적 STAY 시간별 인원 합의 목적 구성비'},
 {'case': '중마시장',
  'case_label': '중마',
  'metric': '쇼핑여가 목적 STAY',
  'case_share': 0.0071260744014798,
  'national_market_row_mean': 0.0372771102023517,
  'national_valid_market_rows': 830,
  'source_snapshot': 'a1_stay_purpose_mix.csv',
  'scope': '2023-09-01~2023-10-15 중 일반일 38일; 전목적 STAY 시간별 인원 합의 목적 구성비'},
 {'case': '신림 시장권',
  'case_label': '신림동',
  'metric': '20·30대 STAY',
  'case_share': 0.4856855047485557,
  'national_market_row_mean': 0.263374230820392,
  'national_valid_market_rows': 830,
  'source_snapshot': 'market_features_v4.csv',
  'scope': '2023-09-01~2023-10-15 중 일반일 38일; 전목적 STAY 시간별 인원 합의 연령 구성비'},
 {'case': '신림 시장권',
  'case_label': '신림동',
  'metric': '쇼핑여가 목적 STAY',
  'case_share': 0.0677834686344262,
  'national_market_row_mean': 0.0372771102023516,
  'national_valid_market_rows': 830,
  'source_snapshot': 'a1_hdong_time_purpose_profile.csv',
  'scope': '2023-09-01~2023-10-15 중 일반일 38일; 전목적 STAY 시간별 인원 합의 목적 구성비'},
 {'case': '국제시장권',
  'case_label': '광복동',
  'metric': '쇼핑여가 목적 STAY',
  'case_share': 0.2757356083427678,
  'national_market_row_mean': 0.0372771102023516,
  'national_valid_market_rows': 830,
  'source_snapshot': 'a1_hdong_time_purpose_profile.csv',
  'scope': '2023-09-01~2023-10-15 중 일반일 38일; 전목적 STAY 시간별 인원 합의 목적 구성비'},
 {'case': '국제시장권',
  'case_label': '광복동',
  'metric': '쇼핑여가 목적 STAY 중 60대 이상',
  'case_share': 0.2497139884578453,
  'national_market_row_mean': 0.1770037150134049,
  'national_valid_market_rows': 813,
  'source_snapshot': 'a1_hdong_time_purpose_profile.csv',
  'scope': '2023-09-01~2023-10-15 중 일반일 38일; 쇼핑여가 목적 STAY 시간별 인원 합의 연령 구성비'}])
HOUR = pd.DataFrame([{'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 8, 'hour_share': 0.01163113, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 9, 'hour_share': 0.0348934, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 10, 'hour_share': 0.05160144, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 11, 'hour_share': 0.08995943, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 12, 'hour_share': 0.11889853, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 13, 'hour_share': 0.12412542, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 14, 'hour_share': 0.13005722, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 15, 'hour_share': 0.12363797, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 16, 'hour_share': 0.0995658, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 17, 'hour_share': 0.07597357, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 18, 'hour_share': 0.0510765, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 19, 'hour_share': 0.03418098, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 20, 'hour_share': 0.02110252, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 21, 'hour_share': 0.01596562, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 22, 'hour_share': 0.01125618, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '주말', 'hour': 23, 'hour_share': 0.00607429, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 8, 'hour_share': 0.07446293, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 9, 'hour_share': 0.06726439, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 10, 'hour_share': 0.05871007, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 11, 'hour_share': 0.07845764, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 12, 'hour_share': 0.07844957, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 13, 'hour_share': 0.0729054, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 14, 'hour_share': 0.07777975, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 15, 'hour_share': 0.08576109, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 16, 'hour_share': 0.07091208, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 17, 'hour_share': 0.08354181, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 18, 'hour_share': 0.07866746, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 19, 'hour_share': 0.05774166, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 20, 'hour_share': 0.04315897, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 21, 'hour_share': 0.03628323, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 22, 'hour_share': 0.02454928, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'daytype': '평일', 'hour': 23, 'hour_share': 0.01135465, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 8, 'hour_share': 0.06063009, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 9, 'hour_share': 0.06271261, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 10, 'hour_share': 0.06422817, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 11, 'hour_share': 0.0659634, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 12, 'hour_share': 0.06879586, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 13, 'hour_share': 0.07075039, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 14, 'hour_share': 0.07132193, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 15, 'hour_share': 0.07098153, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 16, 'hour_share': 0.0691653, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 17, 'hour_share': 0.0659741, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 18, 'hour_share': 0.06163448, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 19, 'hour_share': 0.05763909, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 20, 'hour_share': 0.05470042, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 21, 'hour_share': 0.05289412, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 22, 'hour_share': 0.05180301, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 23, 'hour_share': 0.0508055, 'observed_days': 12},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 8, 'hour_share': 0.06413142, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 9, 'hour_share': 0.06554245, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 10, 'hour_share': 0.06631781, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 11, 'hour_share': 0.06782269, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 12, 'hour_share': 0.06923064, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 13, 'hour_share': 0.06878977, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 14, 'hour_share': 0.0676922, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 15, 'hour_share': 0.06725937, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 16, 'hour_share': 0.06571454, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 17, 'hour_share': 0.06380354, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 18, 'hour_share': 0.06056236, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 19, 'hour_share': 0.05727438, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 20, 'hour_share': 0.05550025, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 21, 'hour_share': 0.05429607, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 22, 'hour_share': 0.05360675, 'observed_days': 26},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 23, 'hour_share': 0.05245575, 'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 9,
  'hour_share': 0.0077868,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 10,
  'hour_share': 0.02172852,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 11,
  'hour_share': 0.06292403,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 12,
  'hour_share': 0.10251336,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 13,
  'hour_share': 0.12485866,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 14,
  'hour_share': 0.13195158,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 15,
  'hour_share': 0.12577097,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 16,
  'hour_share': 0.11028731,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 17,
  'hour_share': 0.10575144,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 18,
  'hour_share': 0.08553917,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 19,
  'hour_share': 0.06708727,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 20,
  'hour_share': 0.03862562,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 21,
  'hour_share': 0.01376182,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '주말',
  'hour': 22,
  'hour_share': 0.00141345,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 9,
  'hour_share': 0.01255057,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 10,
  'hour_share': 0.03608131,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 11,
  'hour_share': 0.06672278,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 12,
  'hour_share': 0.09975125,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 13,
  'hour_share': 0.10595744,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 14,
  'hour_share': 0.09565567,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 15,
  'hour_share': 0.09384658,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 16,
  'hour_share': 0.09595718,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 17,
  'hour_share': 0.11661097,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 18,
  'hour_share': 0.12164879,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 19,
  'hour_share': 0.08571823,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 20,
  'hour_share': 0.04919722,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 21,
  'hour_share': 0.01850549,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 도착 OD',
  'daytype': '평일',
  'hour': 22,
  'hour_share': 0.00179653,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 9,
  'hour_share': 0.00369138,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 10,
  'hour_share': 0.0163664,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 11,
  'hour_share': 0.04318745,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 12,
  'hour_share': 0.08005654,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 13,
  'hour_share': 0.10972809,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 14,
  'hour_share': 0.12613067,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 15,
  'hour_share': 0.12685021,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 16,
  'hour_share': 0.11908214,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 17,
  'hour_share': 0.11041782,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 18,
  'hour_share': 0.09572468,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 19,
  'hour_share': 0.07625233,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 20,
  'hour_share': 0.05366339,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 21,
  'hour_share': 0.02874763,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '주말',
  'hour': 22,
  'hour_share': 0.01010127,
  'observed_days': 12},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 9,
  'hour_share': 0.00674391,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 10,
  'hour_share': 0.02662842,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 11,
  'hour_share': 0.05890661,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 12,
  'hour_share': 0.09506043,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 13,
  'hour_share': 0.10898592,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 14,
  'hour_share': 0.10622795,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 15,
  'hour_share': 0.09840255,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 16,
  'hour_share': 0.09319948,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 17,
  'hour_share': 0.09351416,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 18,
  'hour_share': 0.0993502,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 19,
  'hour_share': 0.09009247,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 20,
  'hour_share': 0.06863999,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 21,
  'hour_share': 0.03971838,
  'observed_days': 26},
 {'case': '국제시장권',
  'measure': '쇼핑여가 목적 체류 STAY',
  'daytype': '평일',
  'hour': 22,
  'hour_share': 0.01452953,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 8,
  'hour_share': 0.02195541,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 9,
  'hour_share': 0.03364721,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 10,
  'hour_share': 0.04521798,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 11,
  'hour_share': 0.05056765,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 12,
  'hour_share': 0.05722447,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 13,
  'hour_share': 0.05715185,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 14,
  'hour_share': 0.06327613,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 15,
  'hour_share': 0.06201738,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 16,
  'hour_share': 0.0595241,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 17,
  'hour_share': 0.06741546,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 18,
  'hour_share': 0.0778001,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 19,
  'hour_share': 0.08610297,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 20,
  'hour_share': 0.08760379,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 21,
  'hour_share': 0.09481736,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 22,
  'hour_share': 0.0807049,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '주말',
  'hour': 23,
  'hour_share': 0.05497325,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 8,
  'hour_share': 0.0230061,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 9,
  'hour_share': 0.03036939,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 10,
  'hour_share': 0.03234709,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 11,
  'hour_share': 0.03565443,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 12,
  'hour_share': 0.0403361,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 13,
  'hour_share': 0.04130819,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 14,
  'hour_share': 0.03582203,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 15,
  'hour_share': 0.03715167,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 16,
  'hour_share': 0.04145344,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 17,
  'hour_share': 0.05716329,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 18,
  'hour_share': 0.11561152,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 19,
  'hour_share': 0.14538872,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 20,
  'hour_share': 0.10814767,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 21,
  'hour_share': 0.10311962,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 22,
  'hour_share': 0.09392389,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 도착 OD',
  'daytype': '평일',
  'hour': 23,
  'hour_share': 0.05919685,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 8,
  'hour_share': 0.07031341,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 9,
  'hour_share': 0.06767773,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 10,
  'hour_share': 0.0656333,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 11,
  'hour_share': 0.06356627,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 12,
  'hour_share': 0.06160531,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 13,
  'hour_share': 0.05947973,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 14,
  'hour_share': 0.05772918,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 15,
  'hour_share': 0.05678377,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 16,
  'hour_share': 0.05672001,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 17,
  'hour_share': 0.0574382,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 18,
  'hour_share': 0.05885081,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 19,
  'hour_share': 0.06084944,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 20,
  'hour_share': 0.06285967,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 21,
  'hour_share': 0.06511625,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 22,
  'hour_share': 0.06735603,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '주말',
  'hour': 23,
  'hour_share': 0.06802089,
  'observed_days': 12},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 8,
  'hour_share': 0.0760389,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 9,
  'hour_share': 0.06033368,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 10,
  'hour_share': 0.05372265,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 11,
  'hour_share': 0.05142089,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 12,
  'hour_share': 0.05030988,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 13,
  'hour_share': 0.04887491,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 14,
  'hour_share': 0.04781389,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 15,
  'hour_share': 0.04788551,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 16,
  'hour_share': 0.04894957,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 17,
  'hour_share': 0.0520188,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 18,
  'hour_share': 0.06098766,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 19,
  'hour_share': 0.07147965,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 20,
  'hour_share': 0.07776269,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 21,
  'hour_share': 0.0816254,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 22,
  'hour_share': 0.08515876,
  'observed_days': 26},
 {'case': '신림 시장권',
  'measure': '20·30대 체류 STAY',
  'daytype': '평일',
  'hour': 23,
  'hour_share': 0.08561715,
  'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 8, 'hour_share': 0.01103797, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 9, 'hour_share': 0.02278976, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 10, 'hour_share': 0.02928035, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 11, 'hour_share': 0.03706641, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 12, 'hour_share': 0.05208333, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 13, 'hour_share': 0.04891074, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 14, 'hour_share': 0.05044416, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 15, 'hour_share': 0.09508513, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 16, 'hour_share': 0.09523054, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 17, 'hour_share': 0.14562183, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 18, 'hour_share': 0.10724672, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 19, 'hour_share': 0.11935544, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 20, 'hour_share': 0.0768824, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 21, 'hour_share': 0.0571066, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 22, 'hour_share': 0.03220178, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '주말', 'hour': 23, 'hour_share': 0.01965683, 'observed_days': 12},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 8, 'hour_share': 0.0246836, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 9, 'hour_share': 0.00872252, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 10, 'hour_share': 0.00785828, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 11, 'hour_share': 0.01411034, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 12, 'hour_share': 0.02069544, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 13, 'hour_share': 0.02340621, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 14, 'hour_share': 0.0412349, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 15, 'hour_share': 0.05770609, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 16, 'hour_share': 0.08118397, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 17, 'hour_share': 0.27231642, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 18, 'hour_share': 0.21837927, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 19, 'hour_share': 0.0991307, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 20, 'hour_share': 0.04697684, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 21, 'hour_share': 0.04240689, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 22, 'hour_share': 0.02720888, 'observed_days': 26},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'daytype': '평일', 'hour': 23, 'hour_share': 0.01397965, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 8, 'hour_share': 0.06339827, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 9, 'hour_share': 0.06223555, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 10, 'hour_share': 0.06098389, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 11, 'hour_share': 0.05990228, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 12, 'hour_share': 0.05991205, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 13, 'hour_share': 0.05873071, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 14, 'hour_share': 0.05751608, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 15, 'hour_share': 0.05828986, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 16, 'hour_share': 0.05977052, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 17, 'hour_share': 0.06280328, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 18, 'hour_share': 0.06479442, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 19, 'hour_share': 0.06632006, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 20, 'hour_share': 0.06666957, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 21, 'hour_share': 0.06671183, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 22, 'hour_share': 0.06631574, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '주말', 'hour': 23, 'hour_share': 0.06564589, 'observed_days': 12},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 8, 'hour_share': 0.05984505, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 9, 'hour_share': 0.05578675, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 10, 'hour_share': 0.05489558, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 11, 'hour_share': 0.05631542, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 12, 'hour_share': 0.05743609, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 13, 'hour_share': 0.0565484, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 14, 'hour_share': 0.05710899, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 15, 'hour_share': 0.05879253, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 16, 'hour_share': 0.06097043, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 17, 'hour_share': 0.06640839, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 18, 'hour_share': 0.07055244, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 19, 'hour_share': 0.07076448, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 20, 'hour_share': 0.07018022, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 21, 'hour_share': 0.0695192, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 22, 'hour_share': 0.06807234, 'observed_days': 26},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'daytype': '평일', 'hour': 23, 'hour_share': 0.06680368, 'observed_days': 26}])
WEEK = pd.DataFrame([{'case': '구룡포시장', 'measure': '도착 OD 전체', 'day_of_week': 0, 'daily_index': 68.791665, 'observed_days': 4},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'day_of_week': 1, 'daily_index': 61.940193, 'observed_days': 5},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'day_of_week': 2, 'daily_index': 66.207992, 'observed_days': 6},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'day_of_week': 3, 'daily_index': 66.882529, 'observed_days': 5},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'day_of_week': 4, 'daily_index': 85.631695, 'observed_days': 6},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'day_of_week': 5, 'daily_index': 168.355988, 'observed_days': 6},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'day_of_week': 6, 'daily_index': 159.92428, 'observed_days': 6},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'day_of_week': 0, 'daily_index': 90.396294, 'observed_days': 4},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'day_of_week': 1, 'daily_index': 88.066938, 'observed_days': 5},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'day_of_week': 2, 'daily_index': 88.233286, 'observed_days': 6},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'day_of_week': 3, 'daily_index': 88.864477, 'observed_days': 5},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'day_of_week': 4, 'daily_index': 95.305537, 'observed_days': 6},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'day_of_week': 5, 'daily_index': 123.312455, 'observed_days': 6},
 {'case': '구룡포시장', 'measure': '체류 STAY 전체', 'day_of_week': 6, 'daily_index': 118.775012, 'observed_days': 6},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 0, 'daily_index': 73.68411, 'observed_days': 4},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 1, 'daily_index': 63.437131, 'observed_days': 5},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 2, 'daily_index': 70.320751, 'observed_days': 6},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 3, 'daily_index': 76.88252, 'observed_days': 5},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 4, 'daily_index': 83.858673, 'observed_days': 6},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 5, 'daily_index': 148.551028, 'observed_days': 6},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 6, 'daily_index': 164.547098, 'observed_days': 6},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 체류 STAY', 'day_of_week': 0, 'daily_index': 82.747833, 'observed_days': 4},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 체류 STAY', 'day_of_week': 1, 'daily_index': 68.802976, 'observed_days': 5},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 체류 STAY', 'day_of_week': 2, 'daily_index': 73.453305, 'observed_days': 6},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 체류 STAY', 'day_of_week': 3, 'daily_index': 80.601084, 'observed_days': 5},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 체류 STAY', 'day_of_week': 4, 'daily_index': 89.306982, 'observed_days': 6},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 체류 STAY', 'day_of_week': 5, 'daily_index': 142.735731, 'observed_days': 6},
 {'case': '국제시장권', 'measure': '쇼핑여가 목적 체류 STAY', 'day_of_week': 6, 'daily_index': 148.168711, 'observed_days': 6},
 {'case': '신림 시장권', 'measure': '20·30대 도착 OD', 'day_of_week': 0, 'daily_index': 101.602336, 'observed_days': 4},
 {'case': '신림 시장권', 'measure': '20·30대 도착 OD', 'day_of_week': 1, 'daily_index': 102.383781, 'observed_days': 5},
 {'case': '신림 시장권', 'measure': '20·30대 도착 OD', 'day_of_week': 2, 'daily_index': 96.920446, 'observed_days': 6},
 {'case': '신림 시장권', 'measure': '20·30대 도착 OD', 'day_of_week': 3, 'daily_index': 101.518091, 'observed_days': 5},
 {'case': '신림 시장권', 'measure': '20·30대 도착 OD', 'day_of_week': 4, 'daily_index': 98.745754, 'observed_days': 6},
 {'case': '신림 시장권', 'measure': '20·30대 도착 OD', 'day_of_week': 5, 'daily_index': 98.687654, 'observed_days': 6},
 {'case': '신림 시장권', 'measure': '20·30대 도착 OD', 'day_of_week': 6, 'daily_index': 101.326361, 'observed_days': 6},
 {'case': '신림 시장권', 'measure': '20·30대 체류 STAY', 'day_of_week': 0, 'daily_index': 90.732701, 'observed_days': 4},
 {'case': '신림 시장권', 'measure': '20·30대 체류 STAY', 'day_of_week': 1, 'daily_index': 92.152207, 'observed_days': 5},
 {'case': '신림 시장권', 'measure': '20·30대 체류 STAY', 'day_of_week': 2, 'daily_index': 93.122875, 'observed_days': 6},
 {'case': '신림 시장권', 'measure': '20·30대 체류 STAY', 'day_of_week': 3, 'daily_index': 91.465122, 'observed_days': 5},
 {'case': '신림 시장권', 'measure': '20·30대 체류 STAY', 'day_of_week': 4, 'daily_index': 93.519053, 'observed_days': 6},
 {'case': '신림 시장권', 'measure': '20·30대 체류 STAY', 'day_of_week': 5, 'daily_index': 112.994726, 'observed_days': 6},
 {'case': '신림 시장권', 'measure': '20·30대 체류 STAY', 'day_of_week': 6, 'daily_index': 120.193771, 'observed_days': 6},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'day_of_week': 0, 'daily_index': 110.832827, 'observed_days': 4},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'day_of_week': 1, 'daily_index': 116.940642, 'observed_days': 5},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'day_of_week': 2, 'daily_index': 110.143519, 'observed_days': 6},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'day_of_week': 3, 'daily_index': 112.108806, 'observed_days': 5},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'day_of_week': 4, 'daily_index': 105.284961, 'observed_days': 6},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'day_of_week': 5, 'daily_index': 73.983698, 'observed_days': 6},
 {'case': '중마시장', 'measure': '귀가 목적 도착 OD', 'day_of_week': 6, 'daily_index': 79.158063, 'observed_days': 6},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'day_of_week': 0, 'daily_index': 101.680306, 'observed_days': 4},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'day_of_week': 1, 'daily_index': 101.278863, 'observed_days': 5},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'day_of_week': 2, 'daily_index': 101.330912, 'observed_days': 6},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'day_of_week': 3, 'daily_index': 100.96668, 'observed_days': 5},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'day_of_week': 4, 'daily_index': 100.512633, 'observed_days': 6},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'day_of_week': 5, 'daily_index': 97.011643, 'observed_days': 6},
 {'case': '중마시장', 'measure': '체류 STAY 전체', 'day_of_week': 6, 'daily_index': 98.153323, 'observed_days': 6}])
TIME_FIGURES = {
    "구룡포시장": "fig_v4_1c_guryongpo_time_weekday.png",
    "중마시장": "fig_v4_2c_jungma_time_weekday.png",
    "신림 시장권": "fig_v4_3d_sillim_time_weekday.png",
    "국제시장권": "fig_v4_4d_international_time_weekday.png",
}

def _draw_time_weekday(case, filename):
    h = HOUR.loc[HOUR.case.eq(case)]
    w = WEEK.loc[WEEK.case.eq(case)]
    measures = list(h.measure.drop_duplicates())
    assert len(measures) == 2 and len(w) == 14, case
    assert "OD" in measures[0] and "STAY" in measures[1], measures
    fig = plt.figure(figsize=(12.5, 6.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, .8], hspace=.43, wspace=.28,
                          left=.07, right=.98, top=.88, bottom=.10)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_week = fig.add_subplot(gs[1, :])

    for ax, measure in zip(axes, measures):
        sub = h.loc[h.measure.eq(measure)]
        for daytype, color in [("평일", BLUE), ("주말", ORANGE)]:
            part = sub.loc[sub.daytype.eq(daytype)].sort_values("hour")
            ax.plot(part.hour, part.hour_share * 100, marker="o", markersize=3.5,
                    linewidth=2, color=color, label=daytype)
        ax.set(xlim=(7.8, 23.2), xticks=range(8, 24), xlabel="시각",
               ylabel="시간별 구성비 (%)")
        ax.set_title(measure, loc="left", fontsize=12, color=INK)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    for measure, color, marker in [(measures[0], BLUE, "o"), (measures[1], ORANGE, "D")]:
        part = w.loc[w.measure.eq(measure)].sort_values("day_of_week")
        weekday_share = part.daily_index / part.daily_index.sum() * 100
        assert abs(weekday_share.sum() - 100) < 1e-8
        ax_week.plot(part.day_of_week, weekday_share, color=color, marker=marker,
                     markersize=6, linewidth=1.6, label=measure)
    ax_week.set(xticks=range(7), xticklabels=DAYS,
                ylabel="요일별 일평균 구성비 (%)")
    ax_week.set_title("요일", loc="left", fontsize=12, color=INK)
    ax_week.spines[["top", "right"]].set_visible(False)
    ax_week.grid(axis="y", alpha=.15)
    ax_week.legend(frameon=False, fontsize=9, ncol=2, loc="upper right",
                   bbox_to_anchor=(1, 1.23))

    fig.suptitle(case, x=.07, y=.97,
                 ha="left", fontsize=16, color=INK, fontweight="bold")
    fig.savefig(FIGS / filename, dpi=175, bbox_inches="tight", pad_inches=.22)
    plt.close(fig)

def plot_time_weekday(case):
    """공개 요약값으로 사례의 시간·요일별 OD/STAY 그림을 저장한다."""
    return _draw_time_weekday(case, TIME_FIGURES[case])

COLORS = {
    "외곽형": "#819aab",
    "독립 근린형": "#50a3a2",
    "주거 귀가형": "#176b87",
    "쇼핑 밀집형": "#c99d5b",
}

REPRESENTATIVES = {
    "구룡포": ("시장명", "구룡포시장"),
    "중마": ("시장명", "중마시장"),
    "신림 시장권": ("행정기관코드", 1162069500),
    "국제시장권": ("행정기관코드", 2611057000),
}

def plot_pca(scores_path, classified_path, features_path, output_path):
    scores = pd.read_csv(scores_path, encoding="utf-8-sig")
    labels = pd.read_csv(classified_path, encoding="utf-8-sig")
    features = pd.read_csv(features_path, encoding="utf-8-sig")[
        ["market_uid", "시장명", "행정기관코드"]
    ]
    data = scores.merge(labels[["market_uid", "유형"]], on="market_uid", validate="one_to_one")
    data = data.merge(features, on="market_uid", validate="one_to_one")
    assert len(data) == 830 and data["유형"].notna().all()
    assert data["유형"].value_counts().to_dict() == {
        "외곽형": 204, "독립 근린형": 255, "주거 귀가형": 266, "쇼핑 밀집형": 105
    }
    coordinates = PCA(n_components=3, random_state=0).fit(
        data[scores.columns.drop("market_uid")].to_numpy()
    )
    points = coordinates.transform(data[scores.columns.drop("market_uid")].to_numpy())
    for i in range(3):
        data[f"PC{i + 1}"] = points[:, i]
    variance = coordinates.explained_variance_ratio_

    plt.rcParams.update({
        "font.family": "AppleGothic", "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(16.5, 7.2))
    left = fig.add_subplot(121)
    right = fig.add_subplot(122, projection="3d")
    for name, color in COLORS.items():
        group = data[data["유형"] == name]
        left.scatter(group.PC1, group.PC2, s=18, alpha=.64, color=color,
                     edgecolors="none", label=f"{name} ({len(group)})")
        right.scatter(group.PC1, group.PC2, group.PC3, s=13, alpha=.55,
                      color=color, edgecolors="none", depthshade=False)

    offsets = {"구룡포": (8, -13), "중마": (8, 8),
               "신림 시장권": (8, -12), "국제시장권": (8, 8)}
    for name, (key, value) in REPRESENTATIVES.items():
        group = data[data[key] == value]
        assert len(group) == (4 if "시장권" in name else 1), (name, len(group))
        left.scatter(group.PC1, group.PC2, s=82, facecolors="none",
                     edgecolors="#1d2831", linewidths=1.4, zorder=5)
        right.scatter(group.PC1, group.PC2, group.PC3, s=55, facecolors="none",
                      edgecolors="#1d2831", linewidths=1.2, depthshade=False)
        left.annotate(name, (group.PC1.mean(), group.PC2.mean()),
                      xytext=offsets[name], textcoords="offset points",
                      fontsize=10, fontweight="bold", color="#1d2831")

    left.set_title("2차원 투영", loc="left", fontsize=14, fontweight="bold")
    right.set_title("3차원 투영", loc="left", fontsize=14, fontweight="bold")
    for axis, index in [(left, 0), (left, 1), (right, 0), (right, 1), (right, 2)]:
        label = f"PC{index + 1} ({variance[index]:.1%})"
        if axis is left:
            (axis.set_xlabel if index == 0 else axis.set_ylabel)(label)
        else:
            (axis.set_xlabel if index == 0 else axis.set_ylabel if index == 1 else axis.set_zlabel)(label)
    left.grid(alpha=.2)
    left.spines[["top", "right"]].set_visible(False)
    left.legend(frameon=False, fontsize=10, loc="best")
    right.view_init(elev=22, azim=45)
    right.set_box_aspect((1.1, 1.0, .75))
    fig.suptitle("전국 830개 전통시장의 4유형", fontsize=17, fontweight="bold", x=.04, y=.97, ha="left")
    fig.subplots_adjust(left=.055, right=.98, bottom=.17, top=.87, wspace=.12)
    fig.text(.04, .045, "5개 지표 묶음의 결합 점수를 PCA로 투영 · 원은 대표 시장(권) · 유형 배정은 30차원 결합 점수 기준",
             fontsize=9, color="#50606d")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)
    print("PCA explained:", [round(float(v), 4) for v in variance])

SILLIM_CODES = {
    "가리봉동": "1153059500", "노량진제2동": "1159052000",
    "도림동": "1156058500", "사당제1동": "1159062000",
    "사당제4동": "1159065000", "하안2동": "4121063200",
}

SILLIM_MARKET_CODE = "1162069500"

def save(fig, name):
    fig.savefig(FIGS / name, dpi=175, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)

def market_row(features, name):
    rows = features.loc[features["시장명"].eq(name)]
    if len(rows) != 1:
        raise ValueError(f"{name}: 시장 행이 {len(rows)}개")
    return rows.iloc[0]

def guryongpo(features, hourly):
    row = market_row(features, "구룡포시장")
    values = hourly.loc[(hourly["목적"] == "전 목적") &
                        (hourly["일유형"] == "weekend")].sort_values("시간")
    share = values.set_index("시간")["구룡포_비중"]
    focus = share.loc[11:16].sum() * 100
    assert len(values) == 16 and abs(focus - 68.6) < .1
    car = float(row["OD교통_car"]) * 100
    national_car = features["OD교통_car"].mean() * 100
    parking = int(round(np.expm1(row["인접_parking_log1p"])))
    assert parking == 0

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(left=.065, right=.98, bottom=.23, top=.78, wspace=.31)
    ax.axvspan(10.5, 16.5, color="#e7f0f3", zorder=0)
    ax.plot(share.index, share.values * 100, color=BLUE, marker="o", linewidth=2)
    ax.set(xlim=(7.8, 23.2), ylim=(0, 15), xticks=[8, 10, 12, 14, 16, 18, 20, 22],
           xlabel="도착 시각", ylabel="주말 도착 OD 내 비중 (%)")
    ax.set_title(f"11~16시 도착 {focus:.1f}%", loc="left", color=INK, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    bx.barh([1, 0], [car, national_car], color=[BLUE, PALE], height=.45)
    bx.set(yticks=[1, 0], yticklabels=["구룡포", "전국 평균"], xlim=(0, 105))
    bx.set_title("차량 도착 OD 비중", loc="left", color=INK, fontsize=12)
    bx.spines[["top", "right", "left", "bottom"]].set_visible(False)
    bx.tick_params(axis="x", bottom=False, labelbottom=False)
    bx.tick_params(axis="y", length=0)
    for y, x in [(1, car), (0, national_car)]:
        bx.text(x + 2, y, f"{x:.1f}%", va="center", color=INK)
    bx.text(0, -.75, "반경 500m 공개 공영주차장: 0곳", fontsize=11, color=INK, fontweight="bold")
    fig.suptitle("구룡포시장 · 주말 운영 시간과 주차 검토 근거", x=.02, y=.97,
                 ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.text(.02, .045, "일반일 38일의 주말 · 08~23시 행정동 도착 OD · 주차장은 공개 위치 목록 기준", fontsize=8.5, color="#5c6974")
    save(fig, "fig_v4_1b_guryongpo_time_parking.png")

def geometry_points(coordinates):
    if isinstance(coordinates[0], (int, float)):
        yield coordinates
    else:
        for child in coordinates:
            yield from geometry_points(child)

def map_card(map_obj, title, legend):
    if Element is None:
        raise ImportError("folium과 branca가 필요합니다")
    style = "background:rgba(255,255,255,.96);padding:11px 15px;border-radius:8px;box-shadow:0 2px 8px #0003;font-family:Apple SD Gothic Neo,Malgun Gothic,sans-serif;color:#243447"
    map_obj.get_root().html.add_child(Element(
        f'<div style="position:fixed;z-index:1000;top:16px;left:55px;{style}">'
        f'<div style="font-size:19px;font-weight:700">{html.escape(title)}</div>'
        f'<div style="font-size:12px;margin-top:4px">{legend}</div></div>'
    ))

def sillim(candidates, boundaries, centroids, html_dir):
    data = candidates.copy()
    assert len(data) == 6 and (data["flow"] == 0).all()
    by_name = data.set_index("행정동")
    selected = {**SILLIM_CODES, "신림동": SILLIM_MARKET_CODE}
    shapes = {str(f["properties"]["adm_cd2"]): f for f in boundaries["features"]
              if str(f["properties"]["adm_cd2"]) in selected.values()}
    assert len(shapes) == len(selected), f"행정동 경계 누락: {set(selected.values()) - shapes.keys()}"
    centers = centroids.set_index("행정기관코드")
    center = centers.loc[SILLIM_MARKET_CODE]
    m = folium.Map(location=[center.hdong_lat, center.hdong_lon], tiles="OpenStreetMap",
                   width=1200, height=760, zoom_control=False, control_scale=True, zoom_snap=.25)
    all_points = []
    for name, code in selected.items():
        feature = shapes[code]
        all_points.extend(geometry_points(feature["geometry"]["coordinates"]))
        if name == "신림동":
            fill, label = "#dc5965", "신림 시장권"
            detail = "같은 행정동에 시장 4곳"
        else:
            row = by_name.loc[name]
            fill = "#207c97" if row.n_pass == 4 else ("#8abdd0" if row.n_pass == 2 else "#b5c0c8")
            label = f"{name} · {int(row.n_pass)}/4"
            detail = f"주민 구성 유사도 {row.cos:.3f} · 조건 통과 {int(row.n_pass)}/4 · 신림행 쇼핑 OD 미관측"
        folium.GeoJson(feature, style_function=lambda _, fill=fill: {
            "fillColor": fill, "color": fill, "weight": 2.5, "fillOpacity": .42
        }, tooltip=detail).add_to(m)
        point = centers.loc[code]
        folium.Marker([point.hdong_lat, point.hdong_lon], icon=folium.DivIcon(
            html=f'<div style="white-space:nowrap;display:inline-block;background:#fffE;border:1px solid {fill};'
                 f'border-radius:4px;padding:2px 4px;font:600 11px sans-serif;color:#243447">{html.escape(label)}</div>'
        )).add_to(m)
    lons, lats = zip(*all_points)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(12, 12))
    legend = ('<span style="color:#dc5965">●</span> 신림동　'
              '<span style="color:#207c97">●</span> 4/4　'
              '<span style="color:#8abdd0">●</span> 2/4　'
              '<span style="color:#b5c0c8">●</span> 0/4 (4개 시장 중 조건 통과 수)')
    map_card(m, "신림 시장권 · 배후 후보 동 6곳", legend)
    path = html_dir / "fig_v4_3b_sillim_hinterland.html"
    m.save(str(path))
    return path

def distance_km(lat, lon, ref_lat, ref_lon):
    lat, lon = np.radians(lat), np.radians(lon)
    ref_lat, ref_lon = np.radians(ref_lat), np.radians(ref_lon)
    dlat, dlon = lat - ref_lat, lon - ref_lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(ref_lat) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))

def international(features, html_dir):
    center = market_row(features, "국제시장")
    nearby = features.copy()
    nearby["dist_km"] = distance_km(nearby["위도"], nearby["경도"], center["위도"], center["경도"])
    nearby = nearby.loc[(nearby["dist_km"] <= 3) & (nearby["시장명"] != "국제시장")].copy()
    same = nearby["행정기관코드"].astype(str).eq(str(center["행정기관코드"]))
    assert len(nearby) == 22 and same.sum() == 3 and (~same).sum() == 19

    lat0, lon0 = float(center["위도"]), float(center["경도"])
    m = folium.Map(location=[lat0, lon0], tiles="OpenStreetMap", width=1000, height=1000,
                   zoom_control=False, control_scale=True, zoom_snap=.25)
    folium.Circle([lat0, lon0], radius=3000, color="#176b87", weight=2.5,
                  fill=False, dash_array="7 5", tooltip="국제시장 중심 직선거리 3km").add_to(m)
    for (_, row), is_same in zip(nearby.iterrows(), same):
        color = GOLD if is_same else BLUE
        folium.CircleMarker([row["위도"], row["경도"]], radius=5 if is_same else 4,
                            color="white", weight=1.2, fill=True, fill_color=color,
                            fill_opacity=.9, tooltip=str(row["시장명"])).add_to(m)
    folium.Marker([lat0, lon0], tooltip="국제시장",
                  icon=folium.Icon(color="red", icon="star")).add_to(m)
    dlat, dlon = 3 / 111.32, 3 / (111.32 * np.cos(np.radians(lat0)))
    m.fit_bounds([[lat0 - dlat, lon0 - dlon], [lat0 + dlat, lon0 + dlon]], padding=(8, 8))
    legend = ('<span style="color:#d9545f">●</span> 국제시장　'
              '<span style="color:#c68c40">●</span> 광복동 다른 시장 3곳　'
              '<span style="color:#176b87">●</span> 다른 행정동 등록 시장 19곳')
    map_card(m, "국제시장권 · 반경 3km 안 등록 시장", legend)
    path = html_dir / "fig_v4_4b_international_nearby.html"
    m.save(str(path))
    return path

def render_map_png(source_html, output_png, width, height, node="node"):
    """Folium 지도를 OpenStreetMap 타일 로딩 후 PNG로 저장한다."""
    renderer = Path(__file__).with_name("render_maps.js")
    subprocess.run([node, str(renderer), str(Path(source_html).resolve()),
                    str(Path(output_png).resolve()), str(width), str(height)], check=True)

