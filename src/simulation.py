"""Simulações de negócio com o modelo: cenários "e se?" e lotes de exemplo.

As probabilidades vêm do modelo calibrado (ver ``modeling.calibrate_model``),
para que representem a chance real estimada de um lote ser de alta qualidade.

Importante: o modelo aprende associações nos dados. As simulações mostram
como a *previsão* muda quando uma medida muda — não provam que alterar o
processo causará esse efeito. Mudanças devem ser validadas em lotes-piloto.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FEATURE_LABELS, FEATURES, INK_SECONDARY, MODEL_COLORS


def scenario_table(model, reference, scenarios):
    """Probabilidade prevista para um lote de referência sob vários cenários.

    Args:
        model: modelo com ``predict_proba`` (de preferência calibrado).
        reference: Series com as medidas do lote de referência.
        scenarios: dicionário ``{nome: {variavel: novo_valor}}``.

    Returns:
        DataFrame com as medidas alteradas e a probabilidade de alta qualidade.
    """
    rows = []
    for name, changes in scenarios.items():
        lot = reference.copy()
        for col, value in changes.items():
            lot[col] = value
        prob = model.predict_proba(lot[FEATURES].to_frame().T)[0, 1]
        rows.append({
            "Cenário": name,
            "Teor alcoólico (%)": lot["alcohol"],
            "Acidez volátil (g/L)": lot["volatile acidity"],
            "Sulfatos (g/L)": lot["sulphates"],
            "Chance de alta qualidade": prob,
        })
    return pd.DataFrame(rows).set_index("Cenário")


def partial_dependence(model, X, feature, grid):
    """Probabilidade média prevista quando ``feature`` assume cada valor de ``grid``.

    Para cada valor, a variável é fixada em todos os vinhos de ``X`` e a média
    das probabilidades previstas é calculada (dependência parcial).
    """
    values = []
    for v in grid:
        X_mod = X.copy()
        X_mod[feature] = v
        values.append(model.predict_proba(X_mod)[:, 1].mean())
    return pd.Series(values, index=grid)


def plot_partial_dependence(model, X, features_grids, path=None):
    """Painéis de dependência parcial para as variáveis mais importantes."""
    n = len(features_grids)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), sharey=True)
    for ax, (feature, grid) in zip(np.atleast_1d(axes), features_grids.items()):
        pdp = partial_dependence(model, X, feature, grid)
        ax.plot(pdp.index, pdp.values * 100, color=MODEL_COLORS[0], lw=2.5)
        q10, q90 = X[feature].quantile([0.1, 0.9])
        ax.axvspan(q10, q90, color="#e4e3df", alpha=0.5, lw=0,
                   label="Faixa de 80% dos vinhos")
        ax.set_title(FEATURE_LABELS.get(feature, feature), fontsize=12)
        units = {"alcohol": " (%)", "volatile acidity": " (g/L)", "sulphates": " (g/L)"}
        ax.set_xlabel(FEATURE_LABELS.get(feature, feature) + units.get(feature, ""))
    np.atleast_1d(axes)[0].set_ylabel("Chance média prevista de alta qualidade (%)")
    np.atleast_1d(axes)[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Como a chance prevista muda com cada medida (demais medidas mantidas)",
                 fontsize=12, fontweight="bold", color=INK_SECONDARY)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


BANDS = [(0.4, 1.01, "Prioridade (chance ≥ 40%)"),
         (0.1, 0.4, "Avaliar (chance entre 10% e 40%)"),
         (0.0, 0.1, "Baixo potencial (chance < 10%)")]

RECOMMENDATIONS = {
    BANDS[0][2]: "Enviar primeiro à avaliação sensorial; candidato à linha premium",
    BANDS[1][2]: "Avaliar em seguida; revisar álcool, acidez volátil e sulfatos",
    BANDS[2][2]: "Avaliação simplificada; investigar o processo (ex.: acidez volátil alta)",
}


def assign_band(prob):
    """Classifica uma probabilidade em uma das faixas de ação de negócio."""
    for low, high, name in BANDS:
        if low <= prob < high:
            return name
    return BANDS[-1][2]


def band_table(model, X, y):
    """Resultado real dos lotes em cada faixa de chance prevista.

    Returns:
        DataFrame com quantidade de lotes, vinhos excelentes encontrados e a
        taxa real de excelência em cada faixa — mostra se a previsão é
        confiável o suficiente para orientar decisões.
    """
    prob = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    band = prob.map(assign_band)
    table = pd.DataFrame({"band": band, "y": y}).groupby("band")["y"].agg(
        ["size", "sum", "mean"])
    table = table.reindex([b[2] for b in BANDS])
    table.columns = ["Lotes", "Excelentes (real)", "Taxa real de excelência"]
    table["% dos lotes"] = table["Lotes"] / table["Lotes"].sum()
    table["% dos excelentes"] = table["Excelentes (real)"] / table["Excelentes (real)"].sum()
    table["Ação recomendada"] = [RECOMMENDATIONS[b] for b in table.index]
    table.index.name = "Faixa de chance prevista"
    return table


def example_lots(model, X, y, quality):
    """Um lote real por faixa, escolhido por critério objetivo (sem seleção manual).

    - Lote A: o lote com a **maior** chance prevista.
    - Lote B: o lote cuja chance prevista é a **mais próxima de 20%** (meio da faixa intermediária).
    - Lote C: o lote com a **menor** chance prevista.

    Returns:
        DataFrame com as principais medidas, a chance prevista, a faixa,
        a ação recomendada e o resultado real dos especialistas.
    """
    prob = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    picks = {"Lote A": prob.idxmax(),
             "Lote B": (prob - 0.20).abs().idxmin(),
             "Lote C": prob.idxmin()}
    rows = []
    for label, idx in picks.items():
        band = assign_band(prob[idx])
        rows.append({
            "Lote": label,
            "Teor alcoólico (%)": X.loc[idx, "alcohol"],
            "Acidez volátil (g/L)": X.loc[idx, "volatile acidity"],
            "Sulfatos (g/L)": X.loc[idx, "sulphates"],
            "Chance prevista": prob[idx],
            "Faixa": band,
            "Ação recomendada": RECOMMENDATIONS[band],
            "Nota real dos especialistas": int(quality.loc[idx]),
        })
    return pd.DataFrame(rows).set_index("Lote")


def financial_scenario(n_lots, share_simplified, premium_missed_per_lot,
                       cost_full, cost_simplified, premium_value):
    """Resultado financeiro ilustrativo da triagem (Modo B) para ``n_lots`` lotes.

    Todos os valores monetários são PREMISSAS a serem substituídas pelos dados
    reais da vinícola — a base não contém custos nem preços.

    Args:
        n_lots: quantidade de lotes avaliados por safra.
        share_simplified: fração de lotes que vai para avaliação simplificada.
        premium_missed_per_lot: excelentes na faixa de baixo potencial por lote
            avaliado (pior caso: a avaliação simplificada não os identifica).
        cost_full: custo de uma avaliação completa (painel de especialistas).
        cost_simplified: custo de uma avaliação simplificada.
        premium_value: valor perdido por um vinho premium não identificado.

    Returns:
        Dicionário com economia, perda potencial e resultado líquido.
    """
    simplified = n_lots * share_simplified
    savings = simplified * (cost_full - cost_simplified)
    missed = n_lots * premium_missed_per_lot
    loss = missed * premium_value
    return {
        "Lotes com avaliação simplificada": simplified,
        "Economia em avaliações (R$)": savings,
        "Premium possivelmente não identificados": missed,
        "Perda potencial (R$)": loss,
        "Resultado líquido (R$)": savings - loss,
    }


def breakeven_premium_value(share_simplified, premium_missed_per_lot,
                            cost_full, cost_simplified):
    """Valor máximo de um vinho premium perdido para a triagem ainda compensar.

    Não depende do número de lotes: é a economia por lote dividida pelos
    premium "perdidos" por lote.
    """
    return (share_simplified * (cost_full - cost_simplified)
            / premium_missed_per_lot)
