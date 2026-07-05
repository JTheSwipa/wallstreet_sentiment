"""
Regenerate all presentation/chart_*.png with dark theme:
transparent background, white text, white lines.
"""

import warnings
warnings.filterwarnings('ignore')

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix
import seaborn as sns
import yfinance as yf

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
    'font.size':        11,
    'figure.dpi':       150,
    'text.color':       WHITE,
    'axes.labelcolor':  WHITE,
    'axes.edgecolor':   WHITE,
    'xtick.color':      WHITE,
    'ytick.color':      WHITE,
    'legend.edgecolor': WHITE,
    'legend.labelcolor': WHITE,
    'axes.facecolor':   'none',
    'figure.facecolor': 'none',
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
con = duckdb.connect('data/sentiment.duckdb', read_only=True)

# overall sentiment counts (all subreddits, from sentiment table)
overall = con.execute("""
    SELECT sentiment_label, COUNT(*) as n FROM sentiment
    WHERE is_relevant = true GROUP BY sentiment_label
""").df()

# news ticker_sentiment for top-tickers and price charts
ts_news = con.execute("""
    SELECT date, ticker, sentiment_label, sentiment_score, reddit_score
    FROM ticker_sentiment WHERE subreddit = 'news'
""").df()
ts_news['date'] = pd.to_datetime(ts_news['date'])

# WSB ticker_sentiment for wsb charts
ts_wsb = con.execute("""
    SELECT date, ticker, sentiment_label, sentiment_score
    FROM ticker_sentiment WHERE subreddit = 'wallstreetbets'
""").df()
ts_wsb['date'] = pd.to_datetime(ts_wsb['date'])

# daily aggregates (news) for price overlay
daily_news = con.execute("""
    SELECT date, ticker, AVG(sentiment_score) as avg_sentiment,
           SUM(sentiment_score) as sum_sentiment, COUNT(*) as n_comments,
           SUM(CAST(reddit_score AS DOUBLE)) as weighted_signal
    FROM ticker_sentiment WHERE subreddit = 'news'
    GROUP BY date, ticker
""").df()
daily_news['date'] = pd.to_datetime(daily_news['date'])

con.close()

# eval data
mv = pd.read_csv('eval/model_vs_labels.csv')
mv['match'] = mv['label_sentiment'] == mv['model_sentiment']

# classification reports
reports = {}
for v in ['v1', 'v2', 'v3', 'v4', 'v4c', 'v5']:
    try:
        reports[v] = pd.read_csv(f'eval/{v}_classification_report.csv', index_col=0)
    except FileNotFoundError:
        pass

# kappa data (from score_report_v3.txt)
KAPPA_DATA = [
    ('Sergio\n↔ Alice',       0.12),
    ('Sergio\n↔ Jovan',       0.05),
    ('Sergio\n↔ Francesco',   0.02),
    ('Sergio\n↔ Pierpaolo',   0.11),
    ('Alice\n↔ Jovan',       -0.01),
    ('Alice\n↔ Francesco',    0.06),
    ('Alice\n↔ Pierpaolo',    0.08),
    ('Jovan\n↔ Francesco',    0.03),
    ('Jovan\n↔ Pierpaolo',    0.05),
    ('Francesco\n↔ Pierpaolo', 0.09),
]

print('Data loaded.')


# ─────────────────────────────────────────────────────────────────────────────
# 1. Accuracy evolution
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
bar_colors = ['#e74c3c' if a < 50 else '#e8a09a' if a < 65 else '#a8e6a3' if a < 70 else '#2ecc71' for a in accs]

fig, ax = plt.subplots(figsize=(13, 5.5))
x = np.arange(len(labels))
bars = ax.bar(x, accs, color=bar_colors, edgecolor=WHITE, linewidth=0.5, width=0.6)
ax.axhline(70, color='#5dade2', linewidth=1.5, linestyle='--', alpha=0.9, zorder=3)
ax.text(len(labels) - 0.55, 71.5, '70% target', color='#5dade2', fontsize=9, ha='right')
for bar, acc, n in zip(bars, accs, ns):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.8,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color=WHITE)
    ax.text(bar.get_x() + bar.get_width() / 2, acc / 2,
            f'n={n}', ha='center', va='center', fontsize=8, color='#222', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{l}\n{c}' for l, c in zip(labels, changes)], fontsize=9)
ax.set_ylim(0, 92)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Eval accuracy by prompt version  ·  r/wallstreetbets comments', fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0f}%'))
dark_ax(ax)
plt.tight_layout()
save('chart_accuracy_evolution.png')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inter-annotator kappa
# ─────────────────────────────────────────────────────────────────────────────
pairs  = [d[0] for d in KAPPA_DATA]
kappas = [d[1] for d in KAPPA_DATA]
kappa_colors = ['#e74c3c' if k < 0 else '#e8a09a' for k in kappas]

fig, ax = plt.subplots(figsize=(13, 4.5))
bars = ax.bar(range(len(pairs)), kappas, color=kappa_colors, edgecolor=WHITE, linewidth=0.5, width=0.6)
ax.axhline(0.4, color='#5dade2', linewidth=1.5, linestyle='--', alpha=0.9)
ax.text(len(pairs) - 0.5, 0.42, 'Moderate threshold (κ=0.4)', color='#5dade2', fontsize=9, ha='right')
ax.axhline(0, color=ANNO, linewidth=0.8, alpha=0.5)
for bar, k in zip(bars, kappas):
    ypos = k + 0.005 if k >= 0 else k - 0.012
    va   = 'bottom' if k >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width() / 2, ypos, f'{k:.2f}', ha='center', va=va, fontsize=9, fontweight='bold', color=WHITE)
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels(pairs, fontsize=9)
ax.set_ylabel("Cohen's Kappa")
ax.set_title("Inter-Annotator Agreement — All Annotator Pairs", fontsize=12)
ax.set_ylim(-0.12, 0.58)
dark_ax(ax)
plt.tight_layout()
save('chart_annotator_kappa.png')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-annotator label distribution (calibration)
# ─────────────────────────────────────────────────────────────────────────────
annotators = ['Alice', 'Francesco', 'Jovan', 'Pierpaolo', 'Sergio']
ann_data = {}
for name in annotators:
    df = pd.read_csv(f'eval/calibration_{name}.csv')
    df['label_sentiment'] = df['label_sentiment'].str.strip()
    ann_data[name] = df[df['label_is_relevant'] == True]['label_sentiment'].value_counts()

pivot = pd.DataFrame(ann_data).fillna(0).reindex(
    [s for s in SENT_ORDER if any(s in ann_data[a].index for a in annotators)], fill_value=0
)

fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(annotators))
width = 0.15
for i, label in enumerate(pivot.index):
    offset = (i - len(pivot.index) / 2 + 0.5) * width
    ax.bar(x + offset, pivot[annotators].loc[label], width,
           label=label, color=SENTIMENT_COLORS.get(label, '#ccc'), edgecolor=WHITE, linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels(annotators)
ax.set_ylabel('Count (relevant comments only)')
ax.set_title('Per-annotator sentiment label distribution (calibration set)', fontsize=12)
leg = ax.legend(loc='upper right', fontsize=9, framealpha=0)
for t in leg.get_texts(): t.set_color(WHITE)
dark_ax(ax)
plt.tight_layout()
save('chart_annotator_label_distribution.png')


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calibration agreement distribution
# ─────────────────────────────────────────────────────────────────────────────
agreement_pcts = [13.3, 6.7, 3.3, 13.3, 0.0, 6.7, 10.0, 3.3, 6.7, 10.0]
pair_labels    = [p.replace('\n', ' ') for p in pairs]

fig, ax = plt.subplots(figsize=(13, 4))
bar_cols = ['#e74c3c' if p == 0 else '#e8a09a' if p < 10 else '#a8e6a3' for p in agreement_pcts]
bars = ax.bar(range(len(pair_labels)), agreement_pcts, color=bar_cols, edgecolor=WHITE, linewidth=0.5, width=0.6)
ax.axhline(60, color='#5dade2', linewidth=1.5, linestyle='--', alpha=0.9)
ax.text(len(pair_labels) - 0.5, 62, '60% agreement threshold', color='#5dade2', fontsize=9, ha='right')
for bar, p in zip(bars, agreement_pcts):
    ax.text(bar.get_x() + bar.get_width() / 2, p + 0.5,
            f'{p:.1f}%', ha='center', va='bottom', fontsize=9, color=WHITE)
ax.set_xticks(range(len(pair_labels)))
ax.set_xticklabels(pair_labels, fontsize=8.5, rotation=15, ha='right')
ax.set_ylabel('Exact sentiment agreement (%)')
ax.set_title('Calibration — Pairwise Sentiment Agreement', fontsize=12)
ax.set_ylim(0, 80)
dark_ax(ax)
plt.tight_layout()
save('chart_calibration_agreement_distribution.png')


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pie — relevance
# ─────────────────────────────────────────────────────────────────────────────
total_sent = 230386
relevant   = 56583
not_rel    = total_sent - relevant

fig, ax = plt.subplots(figsize=(7, 5))
wedges, texts, autotexts = ax.pie(
    [relevant, not_rel],
    labels=['Relevant\n(mentions a ticker\n+ expresses a view)', 'Not relevant'],
    autopct='%1.1f%%',
    colors=['#2ecc71', '#95a5a6'],
    startangle=90,
    wedgeprops={'edgecolor': WHITE, 'linewidth': 1.5},
)
for t in texts + autotexts:
    t.set_color(WHITE)
    t.set_fontsize(10)
ax.set_title(f'Comment relevance — {total_sent:,} comments scored', fontsize=12)
plt.tight_layout()
save('chart_pie_relevance.png')


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pie — sentiment distribution (relevant comments)
# ─────────────────────────────────────────────────────────────────────────────
overall_ordered = overall.set_index('sentiment_label').reindex(SENT_ORDER).dropna()

fig, ax = plt.subplots(figsize=(7, 5))
wedges, texts, autotexts = ax.pie(
    overall_ordered['n'],
    labels=overall_ordered.index,
    autopct='%1.1f%%',
    colors=[SENTIMENT_COLORS[s] for s in overall_ordered.index],
    startangle=90,
    wedgeprops={'edgecolor': WHITE, 'linewidth': 1.0},
)
for t in texts + autotexts:
    t.set_color(WHITE)
    t.set_fontsize(10)
ax.set_title('Sentiment distribution — all relevant comments', fontsize=12)
plt.tight_layout()
save('chart_pie_sentiment_distribution.png')


# ─────────────────────────────────────────────────────────────────────────────
# 7. Overall sentiment bar chart
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
total_rel = overall_ordered['n'].sum()
bars = ax.bar(overall_ordered.index, overall_ordered['n'],
              color=[SENTIMENT_COLORS[s] for s in overall_ordered.index],
              edgecolor=WHITE, linewidth=0.5)
for bar, val in zip(bars, overall_ordered['n']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f'{val:,}\n({val/total_rel*100:.1f}%)', ha='center', fontsize=9, color=WHITE)
ax.set_title('Overall sentiment distribution (all relevant comments)', fontsize=12)
ax.set_ylabel('Ticker-comment pairs')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
dark_ax(ax)
plt.tight_layout()
save('chart_overall_sentiment.png')


# ─────────────────────────────────────────────────────────────────────────────
# 8. Top tickers (news data)
# ─────────────────────────────────────────────────────────────────────────────
ticker_sum_news = (
    ts_news.groupby('ticker')
    .agg(mentions=('sentiment_score', 'count'), avg_sentiment=('sentiment_score', 'mean'))
    .sort_values('mentions', ascending=False)
    .reset_index()
)
top_news = ticker_sum_news.head(10)
bar_cols = ['#e74c3c' if s < -0.5 else '#e8a09a' if s < -0.1 else '#95a5a6' if s < 0.1 else '#a8e6a3' if s < 0.5 else '#2ecc71' for s in top_news['avg_sentiment']]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(top_news['ticker'][::-1], top_news['mentions'][::-1], color=bar_cols[::-1], edgecolor=WHITE, linewidth=0.5)
for bar, sent in zip(bars, top_news['avg_sentiment'][::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f'{sent:+.2f}', va='center', fontsize=9, color=ANNO)
ax.set_xlabel('Mentions')
ax.set_title('Top 10 tickers by volume  ·  r/news  ·  Jan 2024–Jan 2025\ncolor = avg sentiment', fontsize=12)
ax.set_xlim(0, top_news['mentions'].max() * 1.2)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
dark_ax(ax)
plt.tight_layout()
save('chart_top_tickers.png')


# ─────────────────────────────────────────────────────────────────────────────
# 9. Sentiment breakdown — stacked bar for top 8 news tickers
# ─────────────────────────────────────────────────────────────────────────────
top8_news = ticker_sum_news.head(8)['ticker'].tolist()
pivot_news = (
    ts_news.groupby(['ticker', 'sentiment_label']).size()
    .unstack(fill_value=0)
    .reindex(columns=[s for s in SENT_ORDER if s in ts_news['sentiment_label'].unique()])
)
pivot_news_pct = pivot_news.loc[top8_news].div(pivot_news.loc[top8_news].sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 5.5))
bottom = np.zeros(len(pivot_news_pct))
for label in pivot_news_pct.columns:
    vals = pivot_news_pct[label].values
    ax.bar(pivot_news_pct.index, vals, bottom=bottom,
           label=label, color=SENTIMENT_COLORS.get(label, '#ccc'),
           edgecolor=WHITE, linewidth=0.4)
    bottom += vals
ax.set_ylabel('% of mentions')
ax.set_title('Sentiment breakdown for top 8 tickers  ·  r/news  ·  Jan 2024–Jan 2025', fontsize=12)
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=9, ncol=5, frameon=False)
for t in leg.get_texts(): t.set_color(WHITE)
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
dark_ax(ax)
for i, ticker in enumerate(pivot_news_pct.index):
    n = ticker_sum_news.loc[ticker_sum_news['ticker'] == ticker, 'mentions'].values[0]
    ax.text(i, 103, f'n={n:,}', ha='center', fontsize=8, color=ANNO)
plt.tight_layout()
save('chart_sentiment_breakdown.png')


# ─────────────────────────────────────────────────────────────────────────────
# 10. Per-sentiment recall across prompt versions
# ─────────────────────────────────────────────────────────────────────────────
recall_versions = ['v1', 'v2', 'v3', 'v5']
sent_classes    = ['very negative', 'negative', 'neutral', 'positive']
version_colors  = {'v1': '#e74c3c', 'v2': '#e8a09a', 'v3': '#a8e6a3', 'v5': '#2ecc71'}

fig, axes = plt.subplots(1, len(sent_classes), figsize=(16, 5), sharey=True)
for ax, sent in zip(axes, sent_classes):
    recalls = []
    for v in recall_versions:
        if v in reports and sent in reports[v].index:
            recalls.append(reports[v].loc[sent, 'recall'] * 100)
        else:
            recalls.append(0)
    bars = ax.bar(recall_versions, recalls,
                  color=[version_colors[v] for v in recall_versions],
                  edgecolor=WHITE, linewidth=0.5, width=0.6)
    for bar, r in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 1,
                f'{r:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color=WHITE)
    ax.set_title(sent, color=SENTIMENT_COLORS.get(sent, WHITE), fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    dark_ax(ax)

axes[0].set_ylabel('Recall')
fig.suptitle('Per-Sentiment Recall Across Prompt Versions', fontsize=13, y=1.01)
plt.tight_layout()
save('chart_recall_by_sentiment.png')


# ─────────────────────────────────────────────────────────────────────────────
# 11. Confusion matrix — v5
# ─────────────────────────────────────────────────────────────────────────────
LABEL_ORDER = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
present = [l for l in LABEL_ORDER if l in mv['label_sentiment'].values or l in mv['model_sentiment'].values]
cm = confusion_matrix(mv['label_sentiment'], mv['model_sentiment'], labels=present)
accuracy = mv['match'].mean()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=present, yticklabels=present, ax=ax,
            linewidths=0.5, linecolor=WHITE,
            annot_kws={'color': WHITE, 'fontsize': 12})
ax.set_xlabel('Model prediction')
ax.set_ylabel('Human label')
ax.set_title(f'Sentiment — Human vs Model (v5, accuracy {accuracy:.1%})', fontsize=12)
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
# colorbar tick colors
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color=WHITE)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)
plt.tight_layout()
save('chart_v5_confusion_matrix.png')


# ─────────────────────────────────────────────────────────────────────────────
# 12. Confusion matrix — v3 (reconstruct from classification report proportions)
#     We don't have raw v3 predictions, so show the v3 classification report
#     as a precision/recall heatmap instead.
# ─────────────────────────────────────────────────────────────────────────────
v3 = reports['v3'].loc[[s for s in SENT_ORDER if s in reports['v3'].index], ['precision', 'recall', 'f1-score']]
v3 = v3.drop(index=[r for r in v3.index if r in ['accuracy', 'macro avg', 'weighted avg']], errors='ignore')

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(v3.astype(float), annot=True, fmt='.2f', cmap='Blues',
            ax=ax, linewidths=0.5, linecolor=WHITE,
            annot_kws={'color': WHITE, 'fontsize': 12},
            vmin=0, vmax=1)
ax.set_title('v3 prompt — Precision / Recall / F1 per class\n(overall accuracy 69.0%)', fontsize=12)
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color=WHITE)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)
plt.tight_layout()
save('chart_v3_confusion_matrix.png')


# ─────────────────────────────────────────────────────────────────────────────
# 13-18. Sentiment vs price overlays (WSB June 1-4, 2026)
# ─────────────────────────────────────────────────────────────────────────────
PRICE_TICKERS = ['SPCE', 'AVGO', 'NVDA', 'MRVL', 'MU', 'GOOGL']

daily_wsb = (
    ts_wsb.groupby(['date', 'ticker'])
    .agg(avg_sentiment=('sentiment_score', 'mean'),
         sum_sentiment=('sentiment_score', 'sum'),
         n_comments=('sentiment_score', 'count'))
    .reset_index()
)

date_min = ts_wsb['date'].min().strftime('%Y-%m-%d')
date_max = (ts_wsb['date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

for ticker in PRICE_TICKERS:
    data = daily_wsb[daily_wsb['ticker'] == ticker].sort_values('date')
    if len(data) == 0:
        print(f'  No data for {ticker}, skipping')
        continue

    # fetch price
    try:
        price = yf.download(ticker, start=date_min, end=date_max, progress=False, auto_adjust=True)['Close']
        price.index = pd.to_datetime(price.index)
        if hasattr(price, 'squeeze'):
            price = price.squeeze()
    except Exception as e:
        print(f'  yfinance error for {ticker}: {e}')
        price = pd.Series(dtype=float)

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax2 = ax1.twinx()

    bar_cols = ['#e74c3c' if s < 0 else '#2ecc71' for s in data['sum_sentiment']]
    x = np.arange(len(data))
    ax1.bar(x, data['sum_sentiment'], color=bar_cols, alpha=0.7, width=0.6, label='Sentiment sum')

    if len(price) > 0:
        price_aligned = price.reindex(data['date']).interpolate()
        if price_aligned.notna().sum() >= 1:
            ax2.plot(x, price_aligned.values, color=WHITE, linewidth=2, marker='o', markersize=6,
                     label=f'{ticker} price (USD)', zorder=5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([d.strftime('%b %d') for d in data['date']])
    ax1.set_ylabel('Sentiment signal (sum)')
    ax2.set_ylabel('Price (USD)')
    ax1.set_xlabel('Date')
    ax1.axhline(0, color=ANNO, linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.set_title(f'{ticker} — WSB sentiment vs stock price  ·  June 1–4, 2026', fontsize=12)

    for spine in ax2.spines.values():
        spine.set_color(WHITE)
    ax2.spines['top'].set_visible(False)
    dark_ax(ax1)

    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(lines1 + lines2, l1 + l2, loc='upper left', fontsize=9, framealpha=0)
    for t in leg.get_texts(): t.set_color(WHITE)

    plt.tight_layout()
    save(f'chart_{ticker}_sentiment_price.png')


print('\nAll presentation charts regenerated.')
