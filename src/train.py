"""Executa a pipeline completa de modelagem pelo terminal.

Uso (a partir da raiz do repositório):
    python -m src.train
"""

import warnings

from src import data_preprocessing as prep
from src import evaluation, modeling
from src.config import FEATURES, RESULTS_DIR


def main():
    """Carrega os dados, treina os modelos, avalia e salva as métricas."""
    warnings.filterwarnings("ignore")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = prep.clean_data(prep.create_target(prep.load_data()))
    print(f"Vinhos após limpeza: {len(df)} | "
          f"alta qualidade: {df['alta_qualidade'].mean():.1%}")

    X_train, X_test, y_train, y_test = prep.split_data(df, FEATURES)
    cv = modeling.get_cv()
    buscas = modeling.tune_models(modeling.build_models(), X_train, y_train,
                                  cv=cv)

    metricas = evaluation.compare_models(buscas, X_test, y_test)
    metricas.to_csv(RESULTS_DIR / "metricas_teste.csv")
    print("\nMétricas na base de teste:")
    print(metricas.to_string())

    # A escolha usa a validação cruzada (treino); o teste só confirma.
    melhor = max(buscas, key=lambda nome: buscas[nome].best_score_)
    print(f"\nModelo escolhido (maior F1 na validação cruzada): {melhor}")


if __name__ == "__main__":
    main()
