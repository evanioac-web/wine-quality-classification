"""Métricas, comparação de modelos e gráficos de avaliação."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate

from src.config import (
    FEATURE_LABELS,
    INK_SECONDARY,
    MODEL_COLORS,
    RANDOM_STATE,
)


def evaluate_model(model, X_test, y_test):
    """Calcula as métricas de classificação de um modelo na base de teste.

    Returns:
        Dicionário com acurácia, precisão, recall, F1 e ROC AUC.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob),
    }


def compare_models(fitted, X_test, y_test):
    """Monta uma tabela comparativa com as métricas de teste de cada modelo."""
    rows = {name: evaluate_model(search.best_estimator_, X_test, y_test)
            for name, search in fitted.items()}
    return pd.DataFrame(rows).T.round(3)


def cross_validation_summary(fitted, X_train, y_train, cv):
    """Resume a validação cruzada (média ± desvio) do melhor modelo de cada tipo."""
    scoring = {"F1": "f1", "ROC AUC": "roc_auc", "Recall": "recall",
               "Precisão": "precision"}
    rows = {}
    for name, search in fitted.items():
        scores = cross_validate(search.best_estimator_, X_train, y_train,
                                cv=cv, scoring=scoring)
        rows[name] = {
            f"{metric} (CV)": f"{scores['test_' + metric].mean():.3f} "
                              f"± {scores['test_' + metric].std():.3f}"
            for metric in scoring
        }
    return pd.DataFrame(rows).T


def plot_confusion_matrices(fitted, X_test, y_test, path=None):
    """Plota as matrizes de confusão de todos os modelos (grade de 2 linhas)."""
    n = len(fitted)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, search) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay.from_estimator(
            search.best_estimator_, X_test, y_test,
            display_labels=["Baixa/Média", "Alta"], cmap="Blues",
            colorbar=False, ax=ax, text_kw={"fontsize": 15},
        )
        ax.set_title(name, fontsize=13)
        ax.grid(False)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(fitted, X_test, y_test, path=None):
    """Plota a curva ROC de todos os modelos no mesmo gráfico."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for color, (name, search) in zip(MODEL_COLORS, fitted.items()):
        y_prob = search.best_estimator_.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})".replace(".", ","))
    ax.plot([0, 1], [0, 1], ls="--", color=INK_SECONDARY, lw=1,
            label="Aleatório (AUC = 0,5)")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos (Recall)")
    ax.set_title("Curva ROC — base de teste")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def feature_importance(model, X_test, y_test, n_repeats=30):
    """Importância das variáveis por permutação (medida pela queda no ROC AUC).

    A importância por permutação funciona para qualquer modelo e é calculada
    na base de teste, refletindo o que o modelo realmente usa para generalizar.
    """
    result = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                    n_repeats=n_repeats,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({
        "variavel": X_test.columns,
        "importancia": result.importances_mean,
        "desvio": result.importances_std,
    })
    imp["rotulo"] = imp["variavel"].map(FEATURE_LABELS).fillna(imp["variavel"])
    return imp.sort_values("importancia", ascending=False).reset_index(drop=True)


def overfitting_table(fitted, X_train, y_train, X_test, y_test):
    """Compara o desempenho no treino, na validação cruzada e no teste.

    Uma diferença grande entre treino e validação/teste indica que o modelo
    "decorou" os dados de treino (overfitting) em vez de aprender padrões
    que se generalizam para vinhos novos.
    """
    rows = {}
    for name, search in fitted.items():
        model = search.best_estimator_
        f1_train = f1_score(y_train, model.predict(X_train))
        f1_test = f1_score(y_test, model.predict(X_test))
        rows[name] = {
            "F1 treino": f1_train,
            "F1 validação cruzada": search.best_score_,
            "F1 teste": f1_test,
            "Diferença treino − validação": f1_train - search.best_score_,
        }
    return pd.DataFrame(rows).T.round(3)


def gains_curve(y_true, y_prob):
    """Curva de triagem (ganho acumulado).

    Ordena os lotes da maior para a menor probabilidade prevista e calcula,
    para cada fração de lotes avaliados, a fração de vinhos de alta qualidade
    encontrados.

    Returns:
        DataFrame com ``pct_lotes`` e ``pct_excelentes`` (0 a 1).
    """
    order = np.argsort(-np.asarray(y_prob))
    hits = np.asarray(y_true)[order]
    n = len(hits)
    return pd.DataFrame({
        "pct_lotes": np.concatenate([[0], np.arange(1, n + 1) / n]),
        "pct_excelentes": np.concatenate([[0], np.cumsum(hits) / hits.sum()]),
    })


def plot_gains_curve(gains, marks=(0.1, 0.2, 0.3), path=None):
    """Plota a curva de triagem com a linha de seleção aleatória."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(gains["pct_lotes"] * 100, gains["pct_excelentes"] * 100,
            color=MODEL_COLORS[0], lw=2.5, label="Triagem com o modelo")
    ax.plot([0, 100], [0, 100], ls="--", color=INK_SECONDARY, lw=1,
            label="Sem modelo (lotes escolhidos ao acaso)")
    for m in marks:
        idx = (gains["pct_lotes"] - m).abs().idxmin()
        x, y = gains.loc[idx, "pct_lotes"] * 100, gains.loc[idx, "pct_excelentes"] * 100
        ax.scatter([x], [y], s=60, color=MODEL_COLORS[0], zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(f"{x:.0f}% dos lotes → {y:.0f}% dos excelentes", (x, y),
                    xytext=(14, -24), textcoords="offset points", fontsize=9.5)
    ax.set_xlabel("% dos lotes enviados para avaliação dos especialistas")
    ax.set_ylabel("% dos vinhos de alta qualidade encontrados")
    ax.set_title("Curva de triagem — base de teste")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def wilson_interval(successes, total, z=1.96):
    """Intervalo de confiança de 95% (Wilson) para uma proporção.

    Útil para comunicar a incerteza de taxas medidas em amostras pequenas
    (ex.: 21 de 27 vinhos excelentes encontrados).
    """
    if total == 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return (center - half, center + half)
