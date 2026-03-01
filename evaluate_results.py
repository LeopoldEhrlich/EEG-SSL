#!/usr/bin/env python3
"""
Model Comparison Script for LOSMO Validation Results
Usage: python compare_models.py <folder_path>
       Expects .xlsx files named after models (e.g., FULL.xlsx, BASE.xlsx, LITE.xlsx)
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from itertools import combinations

# ── Configuration ────────────────────────────────────────────────────────────
METRICS = {
    'bac':      {'label': 'BAC',      'higher_better': True},
    'Accuracy': {'label': 'Accuracy', 'higher_better': True},
}
PALETTE = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_models(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.xlsx')))
    if not files:
        sys.exit(f"No .xlsx files found in {folder}")
    dfs = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_excel(f)
        df['_model'] = name
        dfs[name] = df
        print(f"  Loaded {name}: {len(df)} subjects, cols={df.columns.tolist()}")
    return dfs

def compute_stats(dfs):
    rows = []
    for model, df in dfs.items():
        row = {'Model': model}
        for col, meta in METRICS.items():
            if col not in df.columns:
                continue
            v = df[col].dropna()
            row[f'{col}_mean']   = v.mean()
            row[f'{col}_std']    = v.std()
            row[f'{col}_median'] = v.median()
            row[f'{col}_min']    = v.min()
            row[f'{col}_max']    = v.max()
            row[f'{col}_sem']    = stats.sem(v)
            row[f'{col}_95ci_lo'], row[f'{col}_95ci_hi'] = stats.t.interval(
                0.95, df=len(v)-1, loc=v.mean(), scale=stats.sem(v))
        rows.append(row)
    return pd.DataFrame(rows)

def pairwise_tests(dfs, metric):
    """Return dict of (m1,m2) -> (t_stat, p_value) via paired t-test."""
    models = list(dfs.keys())
    results = {}
    for m1, m2 in combinations(models, 2):
        d1 = dfs[m1][metric].dropna().reset_index(drop=True)
        d2 = dfs[m2][metric].dropna().reset_index(drop=True)
        n  = min(len(d1), len(d2))
        t, p = stats.ttest_rel(d1[:n], d2[:n])
        results[(m1, m2)] = (t, p)
    return results

# ── Plotting ──────────────────────────────────────────────────────────────────
STYLE = {
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#e0e0e0',
    'grid.linewidth':    0.7,
    'axes.labelsize':    11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'figure.facecolor':  'white',
    'axes.facecolor':    '#fafafa',
}

def add_shared_legend(fig, models, colors, ncol=None):
    """Add a clean shared legend centred at the bottom of the figure."""
    handles = [mpatches.Patch(facecolor=colors[m], edgecolor='#444', linewidth=0.8, label=m)
               for m in models]
    fig.legend(handles=handles, loc='lower center',
               ncol=ncol or len(models),
               frameon=True, framealpha=0.9, edgecolor='#cccccc',
               fontsize=10, title='Model', title_fontsize=10,
               bbox_to_anchor=(0.5, -0.04))

def plot_all(dfs, stats_df, out_dir):
    models  = list(dfs.keys())
    colors  = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    metrics = [m for m in METRICS if m in list(dfs.values())[0].columns]
    plt.rcParams.update(STYLE)

    # ── Figure 1: Bar chart with error bars ──────────────────────────────────
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5.2),
                             constrained_layout=False)
    if len(metrics) == 1: axes = [axes]
    x      = np.arange(len(models))
    width  = 0.55
    for ax, metric in zip(axes, metrics):
        means = [stats_df.loc[stats_df.Model == m, f'{metric}_mean'].values[0] for m in models]
        sems  = [stats_df.loc[stats_df.Model == m, f'{metric}_sem'].values[0]  for m in models]
        best_idx = int(np.argmax(means) if METRICS[metric]['higher_better'] else np.argmin(means))
        for i, (m, mean, sem) in enumerate(zip(models, means, sems)):
            lw   = 2.5 if i == best_idx else 0.7
            ec   = '#FFD700' if i == best_idx else '#444444'
            bar  = ax.bar(i, mean, width, yerr=sem, capsize=5,
                          color=colors[m], edgecolor=ec, linewidth=lw,
                          error_kw=dict(elinewidth=1.5, ecolor='#333333', capthick=1.5))
            ax.text(i, mean + sem + 0.002, f'{mean:.4f}',
                    ha='center', va='bottom', fontsize=8.5, color='#222')
        ax.set_title(METRICS[metric]['label'])
        ax.set_ylabel('Value')
        ax.set_xticks([])          # no x-tick labels — legend does the job
        ax.set_xlim(-0.6, len(models) - 0.4)
        ymin = min(means) * 0.96
        ax.set_ylim(bottom=ymin)
    fig.suptitle('Model Comparison — Mean ± SEM', fontsize=14, fontweight='bold', y=1.01)
    add_shared_legend(fig, models, colors)
    fig.subplots_adjust(bottom=0.14, wspace=0.35)
    fig.savefig(os.path.join(out_dir, 'fig1_bar_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: Violin + box plots ─────────────────────────────────────────
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5.2),
                             constrained_layout=False)
    if len(metrics) == 1: axes = [axes]
    for ax, metric in zip(axes, metrics):
        data   = [dfs[m][metric].dropna().values for m in models]
        vparts = ax.violinplot(data, positions=range(len(models)),
                               showmeans=True, showmedians=False, widths=0.7)
        for body, m in zip(vparts['bodies'], models):
            body.set_facecolor(colors[m])
            body.set_alpha(0.55)
            body.set_edgecolor(colors[m])
            body.set_linewidth(1)
        for part in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
            if part in vparts:
                vparts[part].set_color('#444444')
                vparts[part].set_linewidth(1.5)
        ax.boxplot(data, positions=range(len(models)), widths=0.12,
                   medianprops=dict(color='#111111', linewidth=2.5),
                   whiskerprops=dict(linewidth=1.3, color='#444'),
                   capprops=dict(linewidth=1.3, color='#444'),
                   flierprops=dict(marker='o', markersize=3.5, alpha=0.45, color='#555'),
                   boxprops=dict(linewidth=1.3, color='#444'))
        ax.set_xticks([])
        ax.set_title(METRICS[metric]['label'])
        ax.set_ylabel('Value')
    fig.suptitle('Model Comparison — Distributions', fontsize=14, fontweight='bold', y=1.01)
    add_shared_legend(fig, models, colors)
    fig.subplots_adjust(bottom=0.14, wspace=0.35)
    fig.savefig(os.path.join(out_dir, 'fig2_violin_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3: Per-subject scatter ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5.2),
                             constrained_layout=False)
    if len(metrics) == 1: axes = [axes]
    for ax, metric in zip(axes, metrics):
        for m in models:
            vals = dfs[m][metric].dropna().reset_index(drop=True)
            ax.scatter(range(len(vals)), vals, alpha=0.6, s=22,
                       color=colors[m], edgecolors='none')
        ax.set_title(METRICS[metric]['label'])
        ax.set_xlabel('Subject Index')
        ax.set_ylabel('Value')
    fig.suptitle('Per-Subject Validation Values', fontsize=14, fontweight='bold', y=1.01)
    add_shared_legend(fig, models, colors)
    fig.subplots_adjust(bottom=0.14, wspace=0.35)
    fig.savefig(os.path.join(out_dir, 'fig3_per_subject_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 4: BAC vs Accuracy scatter ────────────────────────────────────
    if 'bac' in metrics and 'Accuracy' in metrics:
        fig, ax = plt.subplots(figsize=(5.5, 5.2), constrained_layout=False)
        for m in models:
            ax.scatter(dfs[m]['bac'], dfs[m]['Accuracy'],
                       alpha=0.7, s=35, color=colors[m], edgecolors='none')
        ax.set_xlabel('BAC')
        ax.set_ylabel('Accuracy')
        ax.set_title('BAC vs Accuracy per Subject')
        add_shared_legend(fig, models, colors)
        fig.subplots_adjust(bottom=0.18)
        fig.savefig(os.path.join(out_dir, 'fig4_bac_vs_accuracy.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # ── Figure 5: p-value heatmaps ───────────────────────────────────────────
    for metric in metrics:
        pw = pairwise_tests(dfs, metric)
        model_list = list(dfs.keys())
        n = len(model_list)
        mat = np.ones((n, n))
        for (m1, m2), (_, p) in pw.items():
            i, j = model_list.index(m1), model_list.index(m2)
            mat[i, j] = p
            mat[j, i] = p
        fig, ax = plt.subplots(figsize=(max(4.5, n * 1.4), max(3.8, n * 1.2)),
                               constrained_layout=True)
        im = ax.imshow(mat, vmin=0, vmax=0.1, cmap='RdYlGn_r')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('p-value', fontsize=10)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(model_list, rotation=30, ha='right', fontsize=10)
        ax.set_yticklabels(model_list, fontsize=10)
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, '—', ha='center', va='center', fontsize=11, color='#555')
                else:
                    sig = ('***' if mat[i,j] < 0.001 else '**' if mat[i,j] < 0.01 else
                           '*'   if mat[i,j] < 0.05  else 'ns')
                    txt = f'{mat[i,j]:.3f}\n{sig}'
                    ax.text(j, i, txt, ha='center', va='center', fontsize=8.5,
                            color='white' if mat[i,j] < 0.03 else '#111')
        ax.set_title(f'Paired t-test p-values: {METRICS[metric]["label"]}')
        fig.savefig(os.path.join(out_dir, f'fig5_pvalue_heatmap_{metric}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved figures to {out_dir}")

# ── LaTeX Tables ──────────────────────────────────────────────────────────────
def find_best(stats_df, metric):
    col = f'{metric}_mean'
    if col not in stats_df.columns:
        return None
    return (stats_df.loc[stats_df[col].idxmax(), 'Model']
            if METRICS[metric]['higher_better']
            else stats_df.loc[stats_df[col].idxmin(), 'Model'])

def fmt_cell(val, bold):
    s = f'{val:.4f}'
    return r'\textbf{' + s + '}' if bold else s

def build_latex_summary(stats_df, dfs):
    metrics = [m for m in METRICS if f'{m}_mean' in stats_df.columns]
    bests   = {m: find_best(stats_df, m) for m in metrics}

    # ── Table 1: Mean ± Std per metric ──
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Model Comparison: Mean $\pm$ Std (best in \textbf{bold})}',
        r'\label{tab:mean_std}',
        r'\begin{tabular}{l' + 'c' * len(metrics) + '}',
        r'\toprule',
        'Model & ' + ' & '.join(METRICS[m]['label'] for m in metrics) + r' \\',
        r'\midrule',
    ]
    for _, row in stats_df.iterrows():
        model = row['Model']
        cells = []
        for m in metrics:
            mu  = row[f'{m}_mean']
            sd  = row[f'{m}_std']
            bold = (model == bests[m])
            s   = f'{mu:.4f} $\\pm$ {sd:.4f}'
            cells.append(r'\textbf{' + s + '}' if bold else s)
        lines.append(model + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']

    # ── Table 2: Full descriptive stats ──
    stat_cols = ['mean', 'std', 'median', 'min', 'max', '95ci_lo', '95ci_hi']
    stat_labels = ['Mean', 'Std', 'Median', 'Min', 'Max', '95\% CI Low', '95\% CI High']
    for metric in metrics:
        lines += [
            r'\begin{table}[ht]',
            r'\centering',
            r'\caption{Descriptive Statistics: ' + METRICS[metric]['label'] + '}',
            r'\label{tab:desc_' + metric + '}',
            r'\begin{tabular}{l' + 'c' * len(stat_cols) + '}',
            r'\toprule',
            'Model & ' + ' & '.join(stat_labels) + r' \\',
            r'\midrule',
        ]
        for _, row in stats_df.iterrows():
            model = row['Model']
            vals  = [row[f'{metric}_{s}'] for s in stat_cols]
            best_val = row[f'{metric}_mean']
            is_best  = (model == bests[metric])
            cells    = [fmt_cell(v, is_best) for v in vals]
            lines.append(model + ' & ' + ' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']

    # ── Table 3: Pairwise t-test p-values per metric ──
    model_list = list(dfs.keys())
    for metric in metrics:
        pw = pairwise_tests(dfs, metric)
        lines += [
            r'\begin{table}[ht]',
            r'\centering',
            r'\caption{Pairwise Paired t-test p-values: ' + METRICS[metric]['label'] + '}',
            r'\label{tab:pval_' + metric + '}',
            r'\begin{tabular}{l' + 'c' * len(model_list) + '}',
            r'\toprule',
            'Model & ' + ' & '.join(model_list) + r' \\',
            r'\midrule',
        ]
        for m1 in model_list:
            cells = []
            for m2 in model_list:
                if m1 == m2:
                    cells.append('—')
                else:
                    key = (m1, m2) if (m1, m2) in pw else (m2, m1)
                    _, p = pw[key]
                    sig = (r'$^{***}$' if p < 0.001 else r'$^{**}$' if p < 0.01 else
                           r'$^{*}$' if p < 0.05 else '')
                    cells.append(f'{p:.3f}{sig}')
            lines.append(m1 + ' & ' + ' & '.join(cells) + r' \\')
        lines += [
            r'\bottomrule',
            r'\multicolumn{' + str(len(model_list) + 1) + r'}{l}{\small $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$}\\',
            r'\end{tabular}',
            r'\end{table}', '',
        ]

    header = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{amsmath}
\begin{document}
"""
    footer = r"\end{document}"
    return header + '\n'.join(lines) + '\n' + footer

# ── Main ──────────────────────────────────────────────────────────────────────
def main(folder):
    out_dir = os.path.join(folder, 'comparison_output')
    os.makedirs(out_dir, exist_ok=True)

    print("[1] Loading models…")
    dfs = load_models(folder)

    print("[2] Computing statistics…")
    stats_df = compute_stats(dfs)
    stats_path = os.path.join(out_dir, 'stats_summary.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"  Stats saved to {stats_path}")
    print(stats_df.to_string(index=False))

    print("[3] Generating figures…")
    plot_all(dfs, stats_df, out_dir)

    print("[4] Generating LaTeX tables…")
    latex = build_latex_summary(stats_df, dfs)
    tex_path = os.path.join(out_dir, 'tables.tex')
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f"  LaTeX saved to {tex_path}")

    print("\n✓ Done. All outputs in:", out_dir)
    return out_dir, dfs, stats_df, latex

if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(folder)
