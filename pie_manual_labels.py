"""
Pie chart: manual labeling set (all batch files) — not relevant vs relevant by sentiment.
Dark theme: transparent background, white text.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT   = 'presentation'
WHITE = 'white'

plt.rcParams.update({
    'font.size':        11,
    'figure.dpi':       150,
    'text.color':       WHITE,
    'figure.facecolor': 'none',
    'axes.facecolor':   'none',
})

# ── DATA (from batch_*_labeled.csv files) ────────────────────────────────────
total       = 238
not_rel     = 158   # label_is_relevant == False (+ 1 unanswered)
sent_counts = {
    'very negative': 28,
    'negative':      37,
    'neutral':        7,
    'positive':       5,
    'very positive':  0,   # none labeled
}
# drop zero entries
sent_counts = {k: v for k, v in sent_counts.items() if v > 0}

COLORS = {
    'not relevant':  '#4a4a4a',
    'very negative': '#e74c3c',
    'negative':      '#e8a09a',
    'neutral':       '#95a5a6',
    'positive':      '#a8e6a3',
}

labels  = ['not relevant'] + list(sent_counts.keys())
sizes   = [not_rel + 1] + list(sent_counts.values())   # 158 + 1 unanswered = 159
colors  = [COLORS[l] for l in labels]
explode = [0] + [0.05] * len(sent_counts)

# ── CHART ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=None,
    autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
    colors=colors,
    explode=explode,
    startangle=130,
    wedgeprops={'edgecolor': WHITE, 'linewidth': 1.2},
    pctdistance=0.75,
)
for at in autotexts:
    at.set_color(WHITE)
    at.set_fontsize(10)
    at.set_fontweight('bold')

# ── LEGEND ────────────────────────────────────────────────────────────────────
legend_entries = [
    mpatches.Patch(color=COLORS[l], label=f'{l}  —  {n}  ({n/total*100:.1f}%)')
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
    f'Manual labeling set — {total} comments\nrelevance & sentiment breakdown',
    fontsize=13, pad=16,
)

plt.tight_layout()
plt.savefig(f'{OUT}/wsb_pie_manual_labels.png',
            bbox_inches='tight', transparent=True, facecolor='none')
plt.close()
print('Saved wsb_pie_manual_labels.png')
