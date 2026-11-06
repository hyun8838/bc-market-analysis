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

def _case_comparison(path, title, panels, note):
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.4), squeeze=False)
    fig.subplots_adjust(left=.14 / len(panels), right=.97, bottom=.19, top=.72, wspace=.33)
    for ax, (label, names, values, unit) in zip(axes[0], panels):
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

CASE_COMPARISONS = {'구룡포시장': ['fig_v4_1b_guryongpo_distance.png',
           '구룡포시장 · 거리·접근',
           [('쇼핑 OD 중 10km 이상 출발', ['구룡포', '외곽형 평균', '전국 평균'], [80.5, 71.2, 29.3], '%'),
            ('전 목적 OD 가중 접근시간', ['구룡포', '외곽형 평균', '전국 평균'], [82.6, 63.7, 36.1], '분')],
           '출발지 거리는 일반일 38일, 행정동 중심점 간 직선거리 · 접근시간은 연휴 포함 45일, 도착 인원 가중 이동시간']}

def plot_case_comparison(case):
    """저장된 공개 요약값으로 사례와 전국·유형 평균을 비교한다."""
    if case not in CASE_COMPARISONS:
        raise KeyError(case)
    return _case_comparison(*CASE_COMPARISONS[case])


PURPOSE = {
    '구룡포시장': {'도착 OD': {'귀가': 0.111862, '업무': 0.074554, '학업': 0.004427, '쇼핑여가': 0.018716, '기타': 0.434684, '여행': 0.355757},
        '체류 STAY': {'귀가': 0.500979, '업무': 0.102237, '학업': 0.001893, '쇼핑여가': 0.007255, '기타': 0.23439, '여행': 0.153245}},
    '국제시장권': {'도착 OD': {'귀가': 0.160507, '업무': 0.117235, '학업': 0.000932, '쇼핑여가': 0.406507, '기타': 0.157782, '여행': 0.157036},
        '체류 STAY': {'귀가': 0.26869, '업무': 0.20817, '학업': 0.000506, '쇼핑여가': 0.275736, '기타': 0.142551, '여행': 0.104346}},
}
PURPOSE_COLORS = {"귀가": BLUE, "쇼핑여가": ORANGE, "여행": "#4f8a5b"}
PURPOSE_FIGURES = {
    "구룡포시장": ("fig_v4_1a_guryongpo_purpose.png", ["여행"]),
    "국제시장권": ("fig_v4_4a_international_purpose.png", ["쇼핑여가"]),
}
OTHER_GRAYS = ["#c9d2d9", "#b5c0c8", "#dbe1e6", "#a6b3bd", "#e6eaee"]


def _draw_donut(ax, title, shares, focus):
    names = sorted(shares, key=lambda name: (name not in focus, -shares[name]))
    grays = iter(OTHER_GRAYS)
    colors = [PURPOSE_COLORS[name] if name in focus else next(grays) for name in names]
    wedges, _ = ax.pie([shares[name] for name in names], colors=colors, startangle=90, counterclock=False, normalize=True,
                       wedgeprops={"width": .36, "edgecolor": "white", "linewidth": 2})
    labels = [f"{name} {shares[name] * 100:.1f}%" for name in names]
    legend = ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, .5), frameon=False, fontsize=10)
    for text, name in zip(legend.get_texts(), names):
        text.set_color(INK)
        text.set_fontweight("bold" if name in focus else "normal")
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)


def plot_purpose(case):
    """사례의 도착·체류 목적 구성을 도넛으로 그린다."""
    filename, focus = PURPOSE_FIGURES[case]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.5))
    for ax, (title, shares) in zip(axes, PURPOSE[case].items()):
        _draw_donut(ax, f"{title} 목적", shares, focus)
    fig.suptitle(f"{case} · 방문·체류 목적", x=.02, y=1.02, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / filename, dpi=175, bbox_inches="tight", pad_inches=.22)
    plt.close(fig)


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
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 9, 'hour_share': 0.00931677, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 10, 'hour_share': 0.02795031, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 11, 'hour_share': 0.05693582, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 12, 'hour_share': 0.14596273, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 13, 'hour_share': 0.08281573, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 14, 'hour_share': 0.05072464, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 15, 'hour_share': 0.10973085, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 16, 'hour_share': 0.07556936, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 17, 'hour_share': 0.14389234, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 18, 'hour_share': 0.23291925, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 19, 'hour_share': 0.04140787, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '주말', 'hour': 21, 'hour_share': 0.02277433, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 9, 'hour_share': 0.00562500, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 10, 'hour_share': 0.01312500, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 11, 'hour_share': 0.05218750, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 12, 'hour_share': 0.01250000, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 13, 'hour_share': 0.02156250, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 14, 'hour_share': 0.31937500, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 15, 'hour_share': 0.20718750, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 16, 'hour_share': 0.10437500, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 17, 'hour_share': 0.10593750, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 18, 'hour_share': 0.10156250, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 19, 'hour_share': 0.05312500, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'daytype': '평일', 'hour': 20, 'hour_share': 0.00343750, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 9, 'hour_share': 0.00718743, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 10, 'hour_share': 0.03732207, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 11, 'hour_share': 0.06824556, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 12, 'hour_share': 0.10053643, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 13, 'hour_share': 0.10134282, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 14, 'hour_share': 0.08936961, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 15, 'hour_share': 0.08503962, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 16, 'hour_share': 0.08209452, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 17, 'hour_share': 0.09625903, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 18, 'hour_share': 0.11477105, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 19, 'hour_share': 0.10388472, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 20, 'hour_share': 0.06961293, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 21, 'hour_share': 0.03499053, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '주말', 'hour': 22, 'hour_share': 0.00934366, 'observed_days': 12},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 9, 'hour_share': 0.01127360, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 10, 'hour_share': 0.03433945, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 11, 'hour_share': 0.06969571, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 12, 'hour_share': 0.09370330, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 13, 'hour_share': 0.08109215, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 14, 'hour_share': 0.08395832, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 15, 'hour_share': 0.08670165, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 16, 'hour_share': 0.08518668, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 17, 'hour_share': 0.08883081, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 18, 'hour_share': 0.11102998, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 19, 'hour_share': 0.10887353, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 20, 'hour_share': 0.08207483, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 21, 'hour_share': 0.04783775, 'observed_days': 26},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 체류 STAY', 'daytype': '평일', 'hour': 22, 'hour_share': 0.01540225, 'observed_days': 26}])
WEEK = pd.DataFrame([{'case': '구룡포시장', 'measure': '여행 목적 체류 STAY', 'day_of_week': 0, 'daily_index': 60.971, 'observed_days': 4},
 {'case': '구룡포시장', 'measure': '여행 목적 체류 STAY', 'day_of_week': 1, 'daily_index': 51.044, 'observed_days': 5},
 {'case': '구룡포시장', 'measure': '여행 목적 체류 STAY', 'day_of_week': 2, 'daily_index': 56.568, 'observed_days': 6},
 {'case': '구룡포시장', 'measure': '여행 목적 체류 STAY', 'day_of_week': 3, 'daily_index': 59.93, 'observed_days': 5},
 {'case': '구룡포시장', 'measure': '여행 목적 체류 STAY', 'day_of_week': 4, 'daily_index': 82.342, 'observed_days': 6},
 {'case': '구룡포시장', 'measure': '여행 목적 체류 STAY', 'day_of_week': 5, 'daily_index': 207.203, 'observed_days': 6},
 {'case': '구룡포시장', 'measure': '여행 목적 체류 STAY', 'day_of_week': 6, 'daily_index': 181.941, 'observed_days': 6},
 {'case': '구룡포시장', 'measure': '도착 OD 전체', 'day_of_week': 0, 'daily_index': 68.791665, 'observed_days': 4},
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
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 0, 'daily_index': 77.988478, 'observed_days': 4},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 1, 'daily_index': 113.470955, 'observed_days': 5},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 2, 'daily_index': 128.460554, 'observed_days': 6},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 3, 'daily_index': 131.896303, 'observed_days': 5},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 4, 'daily_index': 101.552248, 'observed_days': 6},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 5, 'daily_index': 79.660746, 'observed_days': 6},
 {'case': '중마시장', 'measure': '쇼핑여가 목적 도착 OD', 'day_of_week': 6, 'daily_index': 67.194751, 'observed_days': 6},
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
}
TIME_PANELS = {
    "구룡포시장": "도착 OD 전체",
    "중마시장": "쇼핑여가 목적 도착 OD",
}
WEEKEND_RATIO_NATIONAL = {"도착 OD 전체": 1.03, "여행 목적 체류 STAY": 1.88}
JUNGMA_DAILY_SHOPPING = {"평일": 123.1, "주말": 80.5}


def _draw_hours(ax, part, measure):
    for daytype, color in [("평일", BLUE), ("주말", ORANGE)]:
        line = part.loc[part.daytype.eq(daytype)].sort_values("hour")
        ax.plot(line.hour, line.hour_share * 100, marker="o", markersize=3.5,
                linewidth=2, color=color, label=daytype)
    ax.set(xlim=(7.8, 23.2), xticks=range(8, 24), xlabel="시각", ylabel="시간별 구성비 (%)")
    ax.set_title(measure, loc="left", fontsize=12, color=INK)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def _draw_guryongpo_weekend_ratios(ax, week):
    measures = ["도착 OD 전체", "여행 목적 체류 STAY"]
    for i, measure in enumerate(measures):
        part = week.loc[week.measure.eq(measure)]
        assert len(part) == 7, measure
        weekday = part.loc[part.day_of_week.lt(5)]
        weekend = part.loc[part.day_of_week.ge(5)]
        weekday_mean = np.average(weekday.daily_index, weights=weekday.observed_days)
        weekend_mean = np.average(weekend.daily_index, weights=weekend.observed_days)
        ratio = weekend_mean / weekday_mean
        ax.barh(i - .15, ratio, height=.3, color=BLUE, label="구룡포" if i == 0 else None)
        ax.barh(i + .15, WEEKEND_RATIO_NATIONAL[measure], height=.3, color="#aebcc7",
                label="전국 평균" if i == 0 else None)
        ax.text(ratio + .06, i - .15, f"{ratio:.2f}배", va="center", fontsize=10)
        ax.text(WEEKEND_RATIO_NATIONAL[measure] + .06, i + .15,
                f"{WEEKEND_RATIO_NATIONAL[measure]:.2f}배", va="center", fontsize=10)
    ax.set(yticks=range(len(measures)), yticklabels=measures, xlim=(0, 3.7), xlabel="주말/평일")
    ax.invert_yaxis()
    ax.set_title("주말/평일 일평균 배수", loc="left", fontsize=12, color=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.legend(frameon=False, fontsize=9, loc="lower right")


def _draw_jungma_daily_counts(ax, week):
    part = week.loc[week.measure.eq("쇼핑여가 목적 도착 OD")]
    assert len(part) == 7
    # 건수는 비공개 원자료 값이고, 저장된 요일 지수의 주말/평일 비율과 맞는지만 확인한다.
    weekday = part.loc[part.day_of_week.lt(5)]
    weekend = part.loc[part.day_of_week.ge(5)]
    ratio = (np.average(weekend.daily_index, weights=weekend.observed_days)
             / np.average(weekday.daily_index, weights=weekday.observed_days))
    assert abs(ratio - JUNGMA_DAILY_SHOPPING["주말"] / JUNGMA_DAILY_SHOPPING["평일"]) < .005
    names = list(JUNGMA_DAILY_SHOPPING)
    values = [JUNGMA_DAILY_SHOPPING[name] for name in names]
    ax.bar(range(2), values, width=.5, color=[BLUE, ORANGE])
    for i, value in enumerate(values):
        ax.text(i, value + 3, f"{value:.1f}건", ha="center", fontsize=10, color=INK)
    ax.set(xticks=range(2), xticklabels=names, xlim=(-.7, 1.7), ylim=(0, 145), ylabel="일평균 도착 건수")
    ax.set_title("쇼핑여가 목적 도착 OD · 일평균", loc="left", fontsize=12, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)


def plot_time_weekday(case):
    """공개 요약값으로 사례의 시간·요일 그림을 저장한다."""
    measure = TIME_PANELS[case]
    hours = HOUR.loc[HOUR.case.eq(case)]
    week = WEEK.loc[WEEK.case.eq(case)]
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 2, wspace=.28, left=.07, right=.98, top=.80, bottom=.17)
    ax_hour, ax_week = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    _draw_hours(ax_hour, hours.loc[hours.measure.eq(measure)], measure)
    if case == "구룡포시장":
        _draw_guryongpo_weekend_ratios(ax_week, week)
    else:
        _draw_jungma_daily_counts(ax_week, week)
    fig.suptitle(f"{case} · 요일·시간", x=.07, y=.97, ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.savefig(FIGS / TIME_FIGURES[case], dpi=175, bbox_inches="tight", pad_inches=.22)
    plt.close(fig)

COLORS = {
    "외곽형": "#819aab",
    "독립 근린형": "#50a3a2",
    "주거 귀가형": "#176b87",
    "쇼핑 밀집형": "#c99d5b",
}

SILLIM_MARKETS = {"관악신사시장": ("신사동", "1162068500"), "관악종합시장": ("신원동", "1162065500"),
                  "신림현대종합상가": ("서림동", "1162066500"), "우림시장": ("난곡동", "1162077500")}

REPRESENTATIVES = {
    "구룡포": ("시장명", ["구룡포시장"]),
    "중마": ("시장명", ["중마시장"]),
    "신림 시장권": ("시장명", list(SILLIM_MARKETS)),
    "국제시장권": ("행정기관코드", [2611057000]),
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
        group = data[data[key].isin(value)]
        assert len(group) == {"국제시장권": 4, "신림 시장권": 4}.get(name, 1), (name, len(group))
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

def market_row(features, name):
    rows = features.loc[features["시장명"].eq(name)]
    if len(rows) != 1:
        raise ValueError(f"{name}: 시장 행이 {len(rows)}개")
    return rows.iloc[0]

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

def sillim(features, boundaries, centroids, html_dir):
    shapes = {str(f["properties"]["adm_cd2"]): f for f in boundaries["features"]}
    first = market_row(features, next(iter(SILLIM_MARKETS)))
    m = folium.Map(location=[first["위도"], first["경도"]], tiles="OpenStreetMap", width=1000, height=1000,
                   zoom_control=False, control_scale=True, zoom_snap=.25)
    all_points = []
    for dong, code in SILLIM_MARKETS.values():
        feature = shapes[code]
        all_points.extend(geometry_points(feature["geometry"]["coordinates"]))
        folium.GeoJson(feature, style_function=lambda _: {
            "fillColor": GOLD, "color": GOLD, "weight": 2, "fillOpacity": .24}).add_to(m)
    for market, (dong, _) in SILLIM_MARKETS.items():
        row = market_row(features, market)
        folium.CircleMarker([row["위도"], row["경도"]], radius=11, color="white", weight=2, fill=True,
                            fill_color=BLUE, fill_opacity=1, tooltip=market).add_to(m)
        folium.Marker([row["위도"] - .0006, row["경도"]], icon=folium.DivIcon(
            icon_anchor=(0, 0),
            html=f'<div style="white-space:nowrap;display:inline-block;background:#fffE;border:1px solid {GOLD};'
                 f'border-radius:4px;padding:2px 5px;font:600 15px sans-serif;color:#243447">{market} · {dong}</div>')).add_to(m)
    lons, lats = zip(*all_points)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(16, 16))
    legend = ('<span style="color:#176b87">●</span> 분석 대상 시장　'
              '<span style="color:#c68c40">■</span> 시장 소재 4개 동')
    map_card(m, "신림 시장권 · 시장 4곳과 소재 행정동", legend)
    path = html_dir / "fig_v4_3b_sillim_map.html"
    m.save(str(path))
    return path

def distance_km(lat, lon, ref_lat, ref_lon):
    lat, lon = np.radians(lat), np.radians(lon)
    ref_lat, ref_lon = np.radians(ref_lat), np.radians(ref_lon)
    dlat, dlon = lat - ref_lat, lon - ref_lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(ref_lat) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))

JUNGMA_ORIGINS = {"금호동": "4623055000", "골약동": "4623051500", "광영동": "4623054000"}


def jungma(features, boundaries, centroids, html_dir):
    center = market_row(features, "중마시장")
    lat0, lon0 = float(center["위도"]), float(center["경도"])
    nearby = features.copy()
    nearby["dist_km"] = distance_km(nearby["위도"], nearby["경도"], lat0, lon0)
    assert (nearby.loc[nearby["시장명"].ne("중마시장"), "dist_km"] > 3).all()
    shapes = {str(f["properties"]["adm_cd2"]): f for f in boundaries["features"]}
    m = folium.Map(location=[lat0, lon0], tiles="OpenStreetMap", width=1000, height=1000,
                   zoom_control=False, control_scale=True, zoom_snap=.25)
    points = centroids.set_index("행정기관코드")
    all_points = []
    for name, code in JUNGMA_ORIGINS.items():
        feature = shapes[code]
        all_points.extend(geometry_points(feature["geometry"]["coordinates"]))
        folium.GeoJson(feature, style_function=lambda _: {
            "fillColor": GOLD, "color": GOLD, "weight": 2, "fillOpacity": .28}).add_to(m)
        shift = .012 if name == "광영동" else 0
        folium.Marker([points.loc[code].hdong_lat + shift, points.loc[code].hdong_lon], icon=folium.DivIcon(
            html=f'<div style="white-space:nowrap;display:inline-block;background:#fffE;border:1px solid {GOLD};'
                 f'border-radius:4px;padding:2px 4px;font:600 15px sans-serif;color:#243447">{name}</div>')).add_to(m)
    folium.Circle([lat0, lon0], radius=3000, color="#176b87", weight=2.5, fill=False, dash_array="7 5").add_to(m)
    lons, lats = zip(*all_points)
    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    inside = nearby.loc[nearby["시장명"].ne("중마시장") & nearby["위도"].between(*[b[0] for b in bounds])
                        & nearby["경도"].between(*[b[1] for b in bounds])]
    for _, row in inside.iterrows():
        folium.CircleMarker([row["위도"], row["경도"]], radius=9, color="white", weight=1.5, fill=True,
                            fill_color=BLUE, fill_opacity=.95, tooltip=str(row["시장명"])).add_to(m)
    folium.CircleMarker([lat0, lon0], radius=12, color="white", weight=2, fill=True,
                        fill_color="#d9545f", fill_opacity=1, tooltip="중마시장").add_to(m)
    m.fit_bounds(bounds, padding=(16, 16))
    legend = ('<span style="color:#d9545f">●</span> 중마시장　'
              '<span style="color:#176b87">●</span> 다른 등록 시장　'
              '<span style="color:#c68c40">■</span> 근처 3개 동　--- 반경 3km')
    map_card(m, "중마시장 · 근처 3개 동과 반경 3km", legend)
    path = html_dir / "fig_v4_2b_jungma_map.html"
    m.save(str(path))
    return path


def plot_jungma_nearby(map_png):
    """지도와 근처 3개 동의 중마동행 비중을 한 그림으로 묶는다."""
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.75, 1], wspace=.12, left=.01, right=.97, top=.92, bottom=.03)
    ax_map, ax = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[1, 1])
    ax_map.imshow(plt.imread(str(map_png)))
    ax_map.axis("off")
    names, values = ["중마", "전국 평균"], [84.7, 16.3]
    bars = ax.barh(range(2), values, color=[FOCUS, REF], height=.55)
    ax.set_yticks(range(2))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_title("근처 3개 동에서 시장 소재 동으로 향한\n쇼핑 OD 중 중마동 도착", color=INK, fontsize=12, loc="left", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for item, value in zip(bars, values):
        ax.text(item.get_width() + 2, item.get_y() + item.get_height() / 2, f"{value:.1f}%",
                va="center", fontsize=10, color=INK)
    fig.suptitle("중마시장 · 입지·인근 이동", x=.02, y=.97, ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.savefig(FIGS / "fig_v4_2b_jungma_nearby.png", dpi=175, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)


def international(features, html_dir):
    center = market_row(features, "국제시장")
    nearby = features.copy()
    nearby["dist_km"] = distance_km(nearby["위도"], nearby["경도"], center["위도"], center["경도"])
    nearby = nearby.loc[(nearby["dist_km"] <= 2) & (nearby["시장명"] != "국제시장")].copy()
    target = nearby["dist_km"].lt(0.6)
    assert len(nearby) == 19 and target.sum() == 7

    lat0, lon0 = float(center["위도"]), float(center["경도"])
    m = folium.Map(location=[lat0, lon0], tiles="OpenStreetMap", width=1000, height=1000,
                   zoom_control=False, control_scale=True, zoom_snap=.25)
    folium.Circle([lat0, lon0], radius=2000, color="#176b87", weight=2.5,
                  fill=False, dash_array="7 5", tooltip="국제시장 중심 직선거리 2km").add_to(m)
    for (_, row), is_target in sorted(zip(nearby.iterrows(), target), key=lambda item: item[1]):
        folium.CircleMarker([row["위도"], row["경도"]], radius=10 if is_target else 9,
                            color="white", weight=1.5, fill=True,
                            fill_color=GOLD if is_target else BLUE,
                            fill_opacity=.95, tooltip=str(row["시장명"])).add_to(m)
    folium.CircleMarker([lat0, lon0], radius=12, color="white", weight=2, fill=True,
                        fill_color="#d9545f", fill_opacity=1, tooltip="국제시장").add_to(m)
    dlat, dlon = 2 / 111.32, 2 / (111.32 * np.cos(np.radians(lat0)))
    m.fit_bounds([[lat0 - dlat, lon0 - dlon], [lat0 + dlat, lon0 + dlon]], padding=(8, 8))
    legend = ('<span style="color:#d9545f">●</span> 국제시장　'
              '<span style="color:#c68c40">●</span> 시범 대상 7곳(약 0.5km 안)　'
              '<span style="color:#176b87">●</span> 그 밖의 2km 안 시장 12곳')
    map_card(m, "국제시장권 · 반경 2km 안 분석 대상 시장", legend)
    path = html_dir / "fig_v4_4b_international_nearby.html"
    m.save(str(path))
    return path


def _nearby_bar(ax, title, names, values):
    bars = ax.barh(range(len(names)), values, color=[FOCUS] * (len(names) - 1) + [REF], height=.55)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.3)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for item, value in zip(bars, values):
        ax.text(item.get_width() + max(values) * .025, item.get_y() + item.get_height() / 2,
                f"{value:.1f}%", va="center", fontsize=10, color=INK)


def plot_international_nearby(map_png):
    """지도와 근거리 출발·옆 동네 쇼핑 비중을 한 그림으로 묶는다."""
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], wspace=.12, hspace=.35,
                          left=.01, right=.97, top=.90, bottom=.12)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.imshow(plt.imread(str(map_png)))
    ax_map.axis("off")
    _nearby_bar(fig.add_subplot(gs[0, 1]), "쇼핑 OD 중 2km 안 출발", ["국제시장권", "전국 평균"], [41.1, 25.1])
    _nearby_bar(fig.add_subplot(gs[1, 1]), "옆 동네 쇼핑여가 목적 도착 OD", ["남포동", "부평동", "전국 평균"], [31.1, 21.5, 5.9])
    fig.suptitle("국제시장권 · 거리·접근", x=.02, y=.97, ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.savefig(FIGS / "fig_v4_4b_international_nearby.png", dpi=175, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)


# 신림 시장 4곳이 있는 행정동의 공개 요약값(일반일 38일). 전국은 시장 행 평균.
SILLIM_DONGS = ["신사동", "신원동", "서림동", "난곡동"]
SILLIM_SHARES = {
    "도착 OD 중 귀가": ([82.2, 78.6, 91.0, 93.1], 55.7),
    "체류 STAY 중 귀가": ([78.4, 83.5, 83.8, 89.3], 65.0),
}
SILLIM_HOME_HOURS = {"신사동": [0.077501, 0.034217, 0.030877, 0.030745, 0.036933, 0.040562, 0.044097, 0.059259, 0.073802, 0.075233, 0.108492, 0.115234, 0.087275, 0.086147, 0.067306, 0.03232], "신원동": [0.060605, 0.034636, 0.036223, 0.035753, 0.043424, 0.044131, 0.046259, 0.063602, 0.084764, 0.086381, 0.093187, 0.09503, 0.087758, 0.079226, 0.074036, 0.034986], "서림동": [0.050458, 0.035224, 0.033072, 0.038074, 0.045998, 0.051827, 0.049904, 0.066195, 0.076841, 0.078539, 0.100591, 0.093952, 0.080037, 0.080428, 0.076969, 0.041891], "난곡동": [0.061915, 0.037413, 0.03613, 0.034995, 0.038149, 0.044457, 0.055213, 0.073302, 0.088688, 0.085471, 0.114736, 0.111514, 0.079804, 0.064291, 0.052947, 0.020975], "전국 평균": [0.069514, 0.033779, 0.028465, 0.031279, 0.038804, 0.045498, 0.051712, 0.067077, 0.087562, 0.108299, 0.130322, 0.091874, 0.076906, 0.066382, 0.050532, 0.021994]}
SILLIM_HOME_EVENING = {"신사동": {"평일": 2370, "주말": 1866}, "신원동": {"평일": 3795, "주말": 3289}, "서림동": {"평일": 2569, "주말": 1938}, "난곡동": {"평일": 2411, "주말": 1750}}


def plot_sillim_purpose():
    """네 동의 귀가 목적 도착·체류 비중을 전국 평균과 비교한다."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    for ax, (title, (values, national)) in zip(axes, SILLIM_SHARES.items()):
        _nearby_bar(ax, title, SILLIM_DONGS + ["전국 평균"], values + [national])
    fig.suptitle("신림 시장권 · 귀가 목적 도착·체류", x=.02, y=1.0, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_v4_3a_sillim_purpose.png", dpi=175, bbox_inches="tight", pad_inches=.22)
    plt.close(fig)


def plot_sillim_map(map_png):
    """시장 4곳과 소재 행정동 지도를 저장한다."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(plt.imread(str(map_png)))
    ax.axis("off")
    fig.savefig(FIGS / "fig_v4_3b_sillim_map.png", dpi=175, bbox_inches="tight", pad_inches=.1)
    plt.close(fig)


def plot_sillim_time():
    """평일 귀가 도착의 시간 분포와 19~23시 일평균 건수를 그린다."""
    fig, (ax, ax_bar) = plt.subplots(1, 2, figsize=(12.5, 4.1), gridspec_kw={"width_ratios": [1.5, 1], "wspace": .24})
    colors = dict(zip(SILLIM_DONGS, [BLUE, ORANGE, "#4f8a5b", "#8a6fa8"]))
    hours = list(range(8, 24))
    for name, shares in SILLIM_HOME_HOURS.items():
        national = name == "전국 평균"
        ax.plot(hours, np.array(shares) * 100, color=REF if national else colors[name], linewidth=2,
                linestyle="--" if national else "-", marker=None if national else "o", markersize=3, label=name)
    ax.axvspan(18.5, 23.2, color=PALE, alpha=.18, linewidth=0)
    ax.set(xlim=(7.8, 23.2), xticks=hours, xlabel="시각", ylabel="시간별 구성비 (%)")
    ax.set_title("평일 귀가 목적 도착 OD (음영: 19~23시)", loc="left", fontsize=12, color=INK)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    positions = np.arange(len(SILLIM_DONGS))
    for offset, daytype, color in [(-.19, "평일", BLUE), (.19, "주말", ORANGE)]:
        values = [SILLIM_HOME_EVENING[name][daytype] for name in SILLIM_DONGS]
        bars = ax_bar.barh(positions + offset, values, height=.36, color=color, label=daytype)
        for item, value in zip(bars, values):
            ax_bar.text(value + 60, item.get_y() + item.get_height() / 2, f"{value:,}", va="center", fontsize=9, color=INK)
    ax_bar.set_yticks(positions)
    ax_bar.set_yticklabels(SILLIM_DONGS)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 4700)
    ax_bar.set_title("19~23시 귀가 도착 일평균(건)", loc="left", fontsize=12, color=INK)
    ax_bar.spines[["top", "right", "left"]].set_visible(False)
    ax_bar.tick_params(axis="y", length=0, labelsize=10)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("신림 시장권 · 요일·시간", x=.07, y=1.0, ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.savefig(FIGS / "fig_v4_3c_sillim_time.png", dpi=175, bbox_inches="tight", pad_inches=.22)
    plt.close(fig)


def render_map_png(source_html, output_png, width, height, node="node"):
    """Folium 지도를 OpenStreetMap 타일 로딩 후 PNG로 저장한다."""
    renderer = Path(__file__).with_name("render_maps.js")
    subprocess.run([node, str(renderer), str(Path(source_html).resolve()),
                    str(Path(output_png).resolve()), str(width), str(height)], check=True)

