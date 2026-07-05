"""
Two charts about the manual labeling set for the presentation slide.
Dark theme: transparent background, white text/lines.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT   = 'presentation'
WHITE = 'white'
ANNO  = '#cccccc'

SENTIMENT_COLORS = {
    'very positive': '#2ecc71',
    'positive':      '#a8e6a3',
    'neutral':       '#95a5a6',
    'negative':      '#e8a09a',
    'very negative': '#e74c3c',
}
SENT_ORDER = ['very negative', 'negative', 'neutral', 'positive', 'very positive']

plt.rcParams.update({
    'font.size':         11,
    'figure.dpi':        150,
    'text.color':        WHITE,
    'axes.labelcolor':   WHITE,
    'axes.edgecolor':    WHITE,
    'xtick.color':       WHITE,
    'ytick.color':       WHITE,
    'legend.edgecolor':  WHITE,
    'legend.labelcolor': WHITE,
    'axes.facecolor':    'none',
    'figure.facecolor':  'none',
})

def dark_ax(ax):
    for spine in ax.spines.values():
        spine.set_color(WHITE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def save(name):
    plt.savefig(f'{OUT}/{name}', bbox_inches='tight', transparent=True, facecolor='none')
    plt.close()
    print(f'Saved {name}')


# ── DATA ──────────────────────────────────────────────────────────────────────
# Calibration: 30 comments each, 5 annotators
# Batch: 34 comments each (Jovan did 3 rounds)
annotators = ['Alice', 'Francesco', 'Jovan', 'Pierpaolo', 'Sergio']

calib_total    = {'Alice': 30, 'Francesco': 30, 'Jovan': 30, 'Pierpaolo': 30, 'Sergio': 30}
calib_relevant = {'Alice':  6, 'Francesco':  4, 'Jovan':  6, 'Pierpaolo':  6, 'Sergio':  6}

# Jovan has 3 batch rounds; combine them
batch_total    = {'Alice': 34, 'Francesco': 34, 'Jovan': 34+34+34, 'Pierpaolo': 34, 'Sergio': 33}
batch_relevant = {'Alice':  9, 'Francesco': 10, 'Jovan': 13+8+13,  'Pierpaolo': 16, 'Sergio': 10}

# eval_final.csv — 71 ground-truth rows
ef = pd.read_csv('eval/eval_final.csv')
ef['label_sentiment'] = ef['label_sentiment'].str.strip()
sent_counts = ef['label_sentiment'].value_counts().reindex(SENT_ORDER).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Per-annotator labeling contribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

y     = np.arange(len(annotators))
h     = 0.35
ct    = [calib_total[a]    for a in annotators]
cr    = [calib_relevant[a] for a in annotators]
bt    = [batch_total[a]    for a in annotators]
br    = [batch_relevant[a] for a in annotators]

# calibration row (top half)
ax.barh(y + h/2, ct, h, color='#5dade2', alpha=0.4, edgecolor=WHITE, linewidth=0.5, label='Calibration — labeled')
ax.barh(y + h/2, cr, h, color='#5dade2', alpha=0.9, edgecolor=WHITE, linewidth=0.5, label='Calibration — relevant')

# batch row (bottom half)
ax.barh(y - h/2, bt, h, color='#a8e6a3', alpha=0.4, edgecolor=WHITE, linewidth=0.5, label='Batch — labeled')
ax.barh(y - h/2, br, h, color='#2ecc71', alpha=0.9, edgecolor=WHITE, linewidth=0.5, label='Batch — relevant')

# value labels
for i, a in enumerate(annotators):
    ax.text(ct[i] + 1, i + h/2, f'{cr[i]}/{ct[i]} relevant', va='center', fontsize=9, color=ANNO)
    ax.text(bt[i] + 1, i - h/2, f'{br[i]}/{bt[i]} relevant', va='center', fontsize=9, color=ANNO)

ax.set_yticks(y)
ax.set_yticklabels(annotators, fontsize=11)
ax.set_xlabel('Comments')
ax.set_title('Manual labeling — per-annotator contribution\n'
             '30 shared calibration comments + 34 unique batch comments each', fontsize=12)
leg = ax.legend(loc='lower right', fontsize=9, framealpha=0, ncol=2)
for t in leg.get_texts(): t.set_color(WHITE)
ax.set_xlim(0, 130)
dark_ax(ax)
plt.tight_layout()
save('wsb_labeling_contribution.png')


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Final eval set sentiment distribution (71 rows)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

total = sent_counts.sum()
bars = ax.bar(
    sent_counts.index, sent_counts.values,
    color=[SENTIMENT_COLORS[s] for s in sent_counts.index],
    edgecolor=WHITE, linewidth=0.5, width=0.6,
)
for bar, val in zip(bars, sent_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f'{val}\n({val/total*100:.0f}%)',
            ha='center', va='bottom', fontsize=10, color=WHITE)

ax.set_ylabel('Comments')
ax.set_title(f'Ground truth eval set — sentiment distribution\n'
             f'{int(total)} relevant, manually labeled comments  ·  5 annotators', fontsize=12)
ax.set_ylim(0, sent_counts.max() * 1.3)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
dark_ax(ax)
# class imbalance note
ax.text(0.98, 0.95, 'Class imbalance: 82% negative/very negative',
        transform=ax.transAxes, ha='right', va='top', fontsize=9, color=ANNO, style='italic')
plt.tight_layout()
save('wsb_eval_set_distribution.png')

print('\nDone.')
