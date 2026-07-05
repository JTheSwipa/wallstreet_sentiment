"""
WSB sentiment charts + eval accuracy evolution — June 1-4, 2026
Dark-presentation style: transparent background, white text and lines.
Outputs to presentation/wsb_*.png and presentation/wsb_accuracy_evolution.png
"""

import warnings
warnings.filterwarnings('ignore')

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DB_PATH = 'data/sentiment.duckdb'
OUT_DIR = 'presentation'
TOP_N        = 15
MIN_MENTIONS = 50

SENTIMENT_COLORS = {
    'very positive': '#2ecc71',
    'positive':      '#a8e6a3',
    'neutral':       '#95a5a6',
    'negative':      '#e8a09a',
    'very negative': '#e74c3c',
}
SENT_ORDER = ['very negative', 'negative', 'neutral', 'positive', 'very positive']

# ── dark-theme defaults ───────────────────────────────────────────────────────
WHITE = 'white'
ANNO  = '#cccccc'   # secondary annotation text

plt.rcParams.update({
    'font.size':          11,
    'figure.dpi':         150,
    'text.color':         WHITE,
    'axes.labelcolor':    WHITE,
    'axes.edgecolor':     WHITE,
    'xtick.color':        WHITE,
    'ytick.color':        WHITE,
    'legend.edgecolor':   WHITE,
    'legend.labelcolor':  WHITE,
    'axes.facecolor':     'none',
    'figure.facecolor':   'none',
})

def dark_ax(ax):
    """Apply white spines and remove top/right for a dark background."""
    for spine in ax.spines.values():
        spine.set_color(WHITE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def save(name):
    plt.savefig(f'{OUT_DIR}/{name}', bbox_inches='tight',
                transparent=True, facecolor='none')
    plt.close()
    print(f'Saved {name}')


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
con = duckdb.connect(DB_PATH, read_only=True)
df = con.execute("""
    SELECT date, ticker, sentiment_label, sentiment_score, reddit_score
    FROM ticker_sentiment WHERE subreddit = 'wallstreetbets'
""").df()
con.close()

df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df):,} WSB rows, {df['date'].min().date()} → {df['date'].max().date()}")

ticker_summary = (
    df.groupby('ticker')
    .agg(mentions=('sentiment_score', 'count'), avg_sentiment=('sentiment_score', 'mean'))
    .sort_values('mentions', ascending=False)
    .reset_index()
)

daily = (
    df.groupby('date')
    .agg(mentions=('sentiment_score', 'count'), avg_sentiment=('sentiment_score', 'mean'))
    .reset_index()
    .sort_values('date')
)

sent_dist = (
    df['sentiment_label'].value_counts()
    .reindex(SENT_ORDER).dropna().reset_index()
)
sent_dist.columns = ['sentiment_label', 'count']

pivot_raw = (
    df.groupby(['ticker', 'sentiment_label']).size()
    .unstack(fill_value=0)
    .reindex(columns=[s for s in SENT_ORDER if s in df['sentiment_label'].unique()])
)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Top tickers by volume
# ─────────────────────────────────────────────────────────────────────────────
top = ticker_summary.head(TOP_N)
colors = [
    '#e74c3c' if s < -0.5 else
    '#e8a09a' if s < -0.1 else
    '#95a5a6' if s < 0.1 else
    '#a8e6a3' if s < 0.5 else '#2ecc71'
    for s in top['avg_sentiment']
]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top['ticker'][::-1], top['mentions'][::-1],
               color=colors[::-1], edgecolor=WHITE, linewidth=0.5)

for bar, sent in zip(bars, top['avg_sentiment'][::-1]):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
            f'{sent:+.2f}', va='center', fontsize=9, color=ANNO)

ax.set_xlabel('Mentions (ticker-comment pairs)')
ax.set_title(f'r/wallstreetbets — Top {TOP_N} tickers by volume\n'
             'June 1–4, 2026  ·  color = avg sentiment score', fontsize=12)
ax.set_xlim(0, top['mentions'].max() * 1.18)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
dark_ax(ax)
plt.tight_layout()
save('wsb_top_tickers.png')


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Most bullish / most bearish
# ─────────────────────────────────────────────────────────────────────────────
qualified = ticker_summary[ticker_summary['mentions'] >= MIN_MENTIONS].copy()
bullish = qualified.nlargest(10, 'avg_sentiment')
bearish = qualified.nsmallest(10, 'avg_sentiment').sort_values('avg_sentiment')

fig, (ax_b, ax_be) = plt.subplots(1, 2, figsize=(14, 5))

ax_b.barh(bullish['ticker'], bullish['avg_sentiment'], color='#2ecc71', edgecolor=WHITE)
for i, (_, row) in enumerate(bullish.iterrows()):
    ax_b.text(row['avg_sentiment'] + 0.02, i,
              f"{row['avg_sentiment']:+.2f} ({int(row['mentions']):,})",
              va='center', fontsize=8.5, color=ANNO)
ax_b.set_xlim(0, bullish['avg_sentiment'].max() * 1.6)
ax_b.set_title('Most Bullish Tickers', fontsize=12, color='#2ecc71')
ax_b.axvline(0, color=ANNO, linewidth=0.8)
ax_b.set_xlabel('Avg sentiment score')
dark_ax(ax_b)

ax_be.barh(bearish['ticker'], bearish['avg_sentiment'], color='#e74c3c', edgecolor=WHITE)
for i, (_, row) in enumerate(bearish.iterrows()):
    ax_be.text(row['avg_sentiment'] - 0.02, i,
               f"{row['avg_sentiment']:+.2f} ({int(row['mentions']):,})",
               va='center', fontsize=8.5, ha='right', color=ANNO)
ax_be.set_xlim(bearish['avg_sentiment'].min() * 1.35, 0.1)
ax_be.set_title('Most Bearish Tickers', fontsize=12, color='#e74c3c')
ax_be.axvline(0, color=ANNO, linewidth=0.8)
ax_be.set_xlabel('Avg sentiment score')
dark_ax(ax_be)

fig.suptitle(f'r/wallstreetbets — Sentiment Rankings  ·  June 1–4, 2026  ·  min {MIN_MENTIONS} mentions',
             fontsize=11)
plt.tight_layout()
save('wsb_bullish_bearish.png')


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Sentiment distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
total = sent_dist['count'].sum()
bars = ax.bar(
    sent_dist['sentiment_label'], sent_dist['count'],
    color=[SENTIMENT_COLORS[s] for s in sent_dist['sentiment_label']],
    edgecolor=WHITE, linewidth=0.5,
)
for bar, val in zip(bars, sent_dist['count']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
            f'{val:,}\n({val/total*100:.1f}%)', ha='center', fontsize=9, color=WHITE)

ax.set_title('r/wallstreetbets — Sentiment distribution (all relevant ticker mentions)\n'
             'June 1–4, 2026', fontsize=12)
ax.set_ylabel('Ticker-comment pairs')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
dark_ax(ax)
plt.tight_layout()
save('wsb_sentiment_distribution.png')


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Daily volume + avg sentiment
# ─────────────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()

x = np.arange(len(daily))
bar_colors = ['#e74c3c' if s < 0 else '#2ecc71' for s in daily['avg_sentiment']]
ax1.bar(x, daily['mentions'], color=bar_colors, alpha=0.75, width=0.6, label='Mentions')
ax2.plot(x, daily['avg_sentiment'], color=WHITE, marker='o',
         linewidth=2, markersize=7, label='Avg sentiment', zorder=5)
ax2.axhline(0, color=ANNO, linewidth=0.8, linestyle='--', alpha=0.6)

ax1.set_xticks(x)
ax1.set_xticklabels([d.strftime('%b %d') for d in daily['date']])
ax1.set_ylabel('Ticker-comment pairs')
ax2.set_ylabel('Avg sentiment score')
ax1.set_xlabel('Date')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax1.set_title('r/wallstreetbets — Daily volume & sentiment  ·  June 1–4, 2026', fontsize=12)

# twin axis spine colors
for spine in ax2.spines.values():
    spine.set_color(WHITE)
ax2.spines['top'].set_visible(False)
dark_ax(ax1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
leg = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                 fontsize=9, framealpha=0)
for text in leg.get_texts():
    text.set_color(WHITE)
plt.tight_layout()
save('wsb_daily_volume_sentiment.png')


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — Stacked sentiment breakdown for top 8 tickers
# ─────────────────────────────────────────────────────────────────────────────
top8 = ticker_summary.head(8)['ticker'].tolist()
pivot = pivot_raw.loc[top8]
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 5.5))
bottom = np.zeros(len(pivot_pct))
for label in pivot_pct.columns:
    vals = pivot_pct[label].values
    ax.bar(pivot_pct.index, vals, bottom=bottom,
           label=label, color=SENTIMENT_COLORS.get(label, '#ccc'),
           edgecolor=WHITE, linewidth=0.4)
    bottom += vals

ax.set_ylabel('% of mentions')
ax.set_title('r/wallstreetbets — Sentiment breakdown for top 8 tickers\nJune 1–4, 2026', fontsize=12)
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                fontsize=9, ncol=5, frameon=False)
for text in leg.get_texts():
    text.set_color(WHITE)
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
dark_ax(ax)

for i, ticker in enumerate(pivot_pct.index):
    n = ticker_summary.loc[ticker_summary['ticker'] == ticker, 'mentions'].values[0]
    ax.text(i, 103, f'n={n:,}', ha='center', fontsize=8, color=ANNO)

plt.tight_layout()
save('wsb_stacked_breakdown.png')


# ─────────────────────────────────────────────────────────────────────────────
# CHART 6 — Eval accuracy evolution (v1 → v5)
# ─────────────────────────────────────────────────────────────────────────────
versions = [
    ('v1',  39.7, 58, 'baseline'),
    ('v2',  60.3, 58, 'domain anchoring\n+ severity scale'),
    ('v3',  69.0, 58, 'few-shot examples\n+ edge case rules'),
    ('v4',  63.8, 58, 'stricter\nrelevance filter'),
    ('v4c', 65.5, 58, 'calibration\nadjustment'),
    ('v5',  77.5, 71, 'corrected\nground truth'),
]

labels  = [v[0] for v in versions]
accs    = [v[1] for v in versions]
ns      = [v[2] for v in versions]
changes = [v[3] for v in versions]

bar_colors = [
    '#e74c3c' if a < 50 else
    '#e8a09a' if a < 65 else
    '#a8e6a3' if a < 70 else '#2ecc71'
    for a in accs
]

fig, ax = plt.subplots(figsize=(13, 5.5))
x = np.arange(len(labels))
bars = ax.bar(x, accs, color=bar_colors, edgecolor=WHITE, linewidth=0.5, width=0.6)

# 70% target line
ax.axhline(70, color='#5dade2', linewidth=1.5, linestyle='--', alpha=0.9, zorder=3)
ax.text(len(labels) - 0.5, 71.2, '70% target', color='#5dade2', fontsize=9, ha='right')

# accuracy labels on bars
for bar, acc, n in zip(bars, accs, ns):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.8,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color=WHITE)
    ax.text(bar.get_x() + bar.get_width() / 2, acc / 2,
            f'n={n}', ha='center', va='center', fontsize=8, color='#222', fontweight='bold')

# change labels below x-axis
ax.set_xticks(x)
ax.set_xticklabels([f'{l}\n{c}' for l, c in zip(labels, changes)], fontsize=9)

ax.set_ylim(0, 92)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Eval accuracy by prompt version  ·  r/wallstreetbets comments', fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0f}%'))
dark_ax(ax)
plt.tight_layout()
save('wsb_accuracy_evolution.png')


print('\nAll charts saved to presentation/')
