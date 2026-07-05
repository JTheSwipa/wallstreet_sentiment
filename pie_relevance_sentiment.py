"""
Pie chart: not-relevant vs relevant-by-sentiment across all 230k scored comments.
Dark theme: transparent background, white text.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT   = 'presentation'
WHITE = 'white'

plt.rcParams.update({
    'font.size':       11,
    'figure.dpi':      150,
    'text.color':      WHITE,
    'figure.facecolor': 'none',
    'axes.facecolor':  'none',
})

# ── DATA ──────────────────────────────────────────────────────────────────────
total       = 230386
not_rel     = 173803   # neutral/irrelevant non-relevant
sent_counts = {
    'very negative': 8841,
    'negative':      26380,
    'neutral':       6792,
    'positive':      11542,
    'very positive': 3028,
}
COLORS = {
    'not relevant':  '#4a4a4a',
    'very negative': '#e74c3c',
    'negative':      '#e8a09a',
    'neutral':       '#95a5a6',
    'positive':      '#a8e6a3',
    'very positive': '#2ecc71',
}

labels = ['not relevant'] + list(sent_counts.keys())
sizes  = [not_rel] + list(sent_counts.values())
colors = [COLORS[l] for l in labels]

# explode the relevant slices slightly to separate them from the grey mass
explode = [0] + [0.04] * len(sent_counts)

# ── CHART ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=None,
    autopct=lambda pct: f'{pct:.1f}%' if pct > 1.5 else '',
    colors=colors,
    explode=explode,
    startangle=140,
    wedgeprops={'edgecolor': WHITE, 'linewidth': 1.2},
    pctdistance=0.78,
)
for at in autotexts:
    at.set_color(WHITE)
    at.set_fontsize(10)
    at.set_fontweight('bold')

# ── LEGEND with counts ────────────────────────────────────────────────────────
legend_entries = [
    mpatches.Patch(color=COLORS[l], label=f'{l}  —  {n:,}  ({n/total*100:.1f}%)')
    for l, n in zip(labels, sizes)
]
leg = ax.legend(
    handles=legend_entries,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    fontsize=10,
    frameon=False,
)
for t in leg.get_texts():
    t.set_color(WHITE)

ax.set_title(
    f'All {total:,} scored comments — relevance & sentiment breakdown',
    fontsize=13, pad=18,
)

plt.tight_layout()
plt.savefig(f'{OUT}/wsb_pie_relevance_sentiment.png',
            bbox_inches='tight', transparent=True, facecolor='none')
plt.close()
print('Saved wsb_pie_relevance_sentiment.png')
