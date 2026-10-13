"""Gráficos e testes estatísticos da Análise Exploratória de Dados (EDA)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from src.config import (
    COLOR_HIGH,
    COLOR_LOW,
    FEATURE_LABELS,
    FEATURES,
    GRID,
    INK,
    INK_SECONDARY,
    TARGET,
)

# Mapa divergente: laranja (correlação negativa) → cinza neutro → azul (positiva)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "laranja_azul", ["#c4501f", "#f3b89c", "#efeeea", "#9cc2ef", "#1f5fae"]
)
CLASS_PALETTE = {0: COLOR_LOW, 1: COLOR_HIGH}
CLASS_NAMES = {0: "Baixa/Média", 1: "Alta"}


def set_style():
    """Aplica um estilo limpo e consistente a todos os gráficos."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.rcParams.update({
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_quality_distribution(df, threshold, path=None):
    """Distribuição das notas originais e da classe binária resultante."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    counts = df["quality"].value_counts().sort_index()
    colors = [COLOR_HIGH if q >= threshold else COLOR_LOW for q in counts.index]
    axes[0].bar(counts.index.astype(str), counts.values, color=colors,
                edgecolor="white", linewidth=2)
    for x, v in zip(counts.index.astype(str), counts.values):
        axes[0].annotate(str(v), (x, v), ha="center", va="bottom",
                         fontsize=9, color=INK_SECONDARY,
                         xytext=(0, 2), textcoords="offset points")
    axes[0].set_title("Notas atribuídas pelos especialistas")
    axes[0].set_xlabel("Nota (quality)")
    axes[0].set_ylabel("Quantidade de vinhos")

    target_counts = df[TARGET].value_counts().sort_index()
    labels = [CLASS_NAMES[i] for i in target_counts.index]
    pct = target_counts / target_counts.sum() * 100
    axes[1].barh(labels, target_counts.values,
                 color=[CLASS_PALETTE[i] for i in target_counts.index],
                 edgecolor="white", linewidth=2)
    for i, (v, p) in enumerate(zip(target_counts.values, pct.values)):
        axes[1].annotate(f"{v} ({p:.1f}%)", (v, i), va="center", fontsize=10,
                         color=INK, xytext=(4, 0), textcoords="offset points")
    axes[1].set_title(f"Classe binária (alta = nota ≥ {threshold})")
    axes[1].set_xlabel("Quantidade de vinhos")
    axes[1].set_xlim(0, target_counts.max() * 1.2)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_histograms(df, features=None, path=None):
    """Histogramas de todas as variáveis, separados por classe."""
    features = features or FEATURES
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.4 * nrows))
    axes = axes.ravel()
    for ax, col in zip(axes, features):
        for cls in (0, 1):
            sns.histplot(df.loc[df[TARGET] == cls, col], ax=ax, stat="density",
                         color=CLASS_PALETTE[cls], alpha=0.45, bins=30,
                         kde=True, label=CLASS_NAMES[cls], edgecolor="white")
        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[len(features):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.0), title="Qualidade")
    _save(fig, path)
    return fig


def plot_boxplots(df, features=None, path=None):
    """Boxplots por classe: mostram diferenças de mediana e outliers."""
    features = features or FEATURES
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.4 * nrows))
    axes = axes.ravel()
    plot_df = df.assign(Classe=df[TARGET].map(CLASS_NAMES))
    order = [CLASS_NAMES[0], CLASS_NAMES[1]]
    palette = {CLASS_NAMES[k]: v for k, v in CLASS_PALETTE.items()}
    for ax, col in zip(axes, features):
        sns.boxplot(data=plot_df, x="Classe", y=col, hue="Classe", order=order,
                    palette=palette, ax=ax, width=0.5, linewidth=1,
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
                    legend=False)
        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_correlation_heatmap(df, columns, method="pearson", path=None):
    """Matriz de correlação (triângulo inferior) com mapa divergente."""
    corr = df[columns].corr(method=method)
    labels = [FEATURE_LABELS.get(c, c) for c in columns]
    corr.index, corr.columns = labels, labels
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap=DIVERGING_CMAP, vmin=-1, vmax=1,
                center=0, annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=1.5, linecolor="white", square=True,
                cbar_kws={"shrink": 0.7, "label": f"Correlação ({method})"},
                ax=ax)
    ax.set_title(f"Matriz de correlação de {method.capitalize()}")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_target_correlation(df, features=None, path=None):
    """Correlação de cada variável com a nota original (Spearman)."""
    features = features or FEATURES
    corr = df[features].corrwith(df["quality"], method="spearman").sort_values()
    labels = [FEATURE_LABELS.get(c, c) for c in corr.index]
    colors = [COLOR_HIGH if v > 0 else COLOR_LOW for v in corr.values]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(labels, corr.values, color=colors, edgecolor="white", linewidth=2)
    ax.axvline(0, color=INK_SECONDARY, lw=0.8)
    for i, v in enumerate(corr.values):
        ax.annotate(f"{v:+.2f}", (v, i), va="center",
                    ha="left" if v >= 0 else "right", fontsize=9,
                    color=INK, xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points")
    ax.set_xlim(-0.6, 0.6)
    ax.set_title("Correlação de cada variável com a nota de qualidade")
    ax.set_xlabel("Correlação de Spearman")
    fig.tight_layout()
    _save(fig, path)
    return fig


def statistical_tests(df, features=None, alpha=0.05):
    """Shapiro-Wilk (normalidade) e Mann-Whitney (diferença entre classes).

    - Shapiro-Wilk: H0 = a variável segue distribuição normal.
    - Mann-Whitney U (não paramétrico, adequado quando não há normalidade):
      H0 = as duas classes têm a mesma distribuição.
    - Como são feitos vários testes ao mesmo tempo (um por variável), o
      p-valor é ajustado pela **correção de Bonferroni** (p × nº de testes),
      reduzindo o risco de "falsas descobertas".
    - Com amostras grandes, até diferenças pequenas ficam "significativas".
      Por isso também reportamos o **tamanho do efeito** (correlação
      rank-biserial, de −1 a +1; positivo = maior nos vinhos de alta
      qualidade). Referência: |r| < 0,3 pequeno; 0,3 a 0,5 médio; ≥ 0,5 grande.

    Returns:
        DataFrame com medianas por classe, p-valores, tamanho do efeito e
        interpretação.
    """
    features = features or FEATURES
    n_tests = len(features)
    rows = []
    high = df[df[TARGET] == 1]
    low = df[df[TARGET] == 0]
    for col in features:
        shapiro_p = stats.shapiro(df[col]).pvalue
        u_stat, mw_p = stats.mannwhitneyu(high[col], low[col])
        effect = 2 * u_stat / (len(high) * len(low)) - 1
        p_bonf = min(mw_p * n_tests, 1.0)
        size = abs(effect)
        rows.append({
            "variavel": FEATURE_LABELS.get(col, col),
            "mediana_alta": high[col].median(),
            "mediana_baixa_media": low[col].median(),
            "normal? (Shapiro)": "sim" if shapiro_p > alpha else "não",
            "shapiro_p": shapiro_p,
            "mann_whitney_p": mw_p,
            "p_bonferroni": p_bonf,
            "significativa (Bonferroni)?": "sim" if p_bonf < alpha else "não",
            "tamanho_efeito": effect,
            "efeito": "grande" if size >= 0.5 else "médio" if size >= 0.3 else "pequeno",
        })
    result = pd.DataFrame(rows)
    result = result.reindex(result["tamanho_efeito"].abs().sort_values(ascending=False).index)
    return result.reset_index(drop=True)


def combined_quality_table(df, alcohol_bins, alcohol_labels, va_bins, va_labels):
    """% de vinhos de alta qualidade e quantidade por faixa de álcool × acidez volátil."""
    alc = pd.cut(df["alcohol"], alcohol_bins, right=False, labels=alcohol_labels)
    va = pd.cut(df["volatile acidity"], va_bins, right=False, labels=va_labels)
    grouped = df.groupby([va, alc], observed=False)[TARGET]
    return grouped.mean().unstack(), grouped.size().unstack()


def plot_combined_heatmap(pct, counts, path=None):
    """Mapa de calor: % de vinhos de alta qualidade em cada combinação."""
    cmap = LinearSegmentedColormap.from_list(
        "seq_azul", ["#f2f6fc", "#9cc2ef", "#2a78d6", "#123f80"])
    annot = pct.apply(lambda col: col.map(lambda v: f"{v:.0%}")).astype(str) \
        + "\n(" + counts.astype(str) + " vinhos)"
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pct, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=0.6,
                linewidths=2, linecolor="white",
                cbar_kws={"label": "% de vinhos de alta qualidade"},
                annot_kws={"size": 10}, ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6])
    cbar.set_ticklabels(["0%", "20%", "40%", "60%"])
    ax.set_xlabel("Teor alcoólico (%)")
    ax.set_ylabel("Acidez volátil (g/L)")
    ax.set_title("% de vinhos de alta qualidade por combinação de álcool e acidez volátil")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, path)
    return fig


def quality_rate_by_bins(df, column, bins, labels):
    """% de vinhos de alta qualidade (e quantidade) em cada faixa de uma variável."""
    faixas = pd.cut(df[column], bins, right=False, labels=labels)
    table = df.groupby(faixas, observed=False)[TARGET].agg(["mean", "size"])
    table.columns = ["% alta qualidade", "vinhos"]
    table.index.name = FEATURE_LABELS.get(column, column)
    return table


def plot_quality_rate_by_bins(tables, path=None):
    """Barras com a % de vinhos de alta qualidade por faixa, uma variável por painel."""
    n = len(tables)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4.6))
    for ax, (title, table, color) in zip(np.atleast_1d(axes), tables):
        x = table.index.astype(str)
        vals = table["% alta qualidade"].values * 100
        ax.bar(x, vals, color=color, edgecolor="white", linewidth=2)
        for xi, v, nv in zip(x, vals, table["vinhos"].values):
            ax.annotate(f"{v:.0f}%\n({nv} vinhos)", (xi, v), ha="center",
                        va="bottom", fontsize=9, color=INK,
                        xytext=(0, 3), textcoords="offset points")
        ax.set_ylim(0, 55)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(table.index.name)
        ax.set_ylabel("% de vinhos de alta qualidade")
    fig.tight_layout()
    _save(fig, path)
    return fig
