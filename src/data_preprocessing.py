"""Funções de carga, limpeza, criação da variável alvo e feature engineering."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    FEATURES,
    QUALITY_THRESHOLD,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def load_data(path=DATA_PATH):
    """Carrega o Wine Quality Dataset a partir de um arquivo CSV.

    Args:
        path: caminho do arquivo CSV.

    Returns:
        DataFrame com os dados brutos.
    """
    return pd.read_csv(path)


def clean_data(df):
    """Remove a coluna de identificação e as amostras duplicadas.

    A coluna ``Id`` é apenas um identificador da linha e não carrega
    informação físico-química. Amostras com todas as medidas e a nota
    idênticas são removidas para evitar que o mesmo vinho apareça ao mesmo
    tempo no treino e no teste (o que inflaria artificialmente as métricas).

    Args:
        df: DataFrame bruto.

    Returns:
        DataFrame limpo, com índice reiniciado.
    """
    df = df.drop(columns=["Id"], errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def create_target(df, threshold=QUALITY_THRESHOLD):
    """Transforma a nota de qualidade em uma classe binária.

    Args:
        df: DataFrame com a coluna ``quality``.
        threshold: nota mínima para o vinho ser considerado de alta qualidade.

    Returns:
        Cópia do DataFrame com a coluna alvo (1 = alta, 0 = baixa/média).
    """
    df = df.copy()
    df[TARGET] = (df["quality"] >= threshold).astype(int)
    return df


def add_features(df):
    """Cria novas variáveis a partir das originais (feature engineering).

    - ``free_so2_ratio``: fração do dióxido de enxofre que está livre
      (a parte que efetivamente protege o vinho).
    - ``total_acidity``: soma das acidezes fixa, volátil e do ácido cítrico.
    - ``alcohol_x_sulphates``: interação entre as duas variáveis mais
      associadas à alta qualidade na análise exploratória.

    Args:
        df: DataFrame com as variáveis físico-químicas originais.

    Returns:
        Cópia do DataFrame com as novas colunas.
    """
    df = df.copy()
    df["free_so2_ratio"] = df["free sulfur dioxide"] / df["total sulfur dioxide"]
    df["total_acidity"] = (
        df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
    )
    df["alcohol_x_sulphates"] = df["alcohol"] * df["sulphates"]
    return df


ENGINEERED_FEATURES = ["free_so2_ratio", "total_acidity", "alcohol_x_sulphates"]


def split_data(df, features=None, test_size=TEST_SIZE,
               random_state=RANDOM_STATE):
    """Separa os dados em treino e teste de forma estratificada.

    O parâmetro ``stratify`` garante que a proporção de vinhos de alta
    qualidade seja a mesma no treino e no teste — essencial em bases
    desbalanceadas.

    Args:
        df: DataFrame com as variáveis e a coluna alvo.
        features: lista de colunas usadas como entrada do modelo.
        test_size: proporção reservada para teste.
        random_state: semente aleatória.

    Returns:
        Tupla ``(X_train, X_test, y_train, y_test)``.
    """
    features = features or FEATURES
    X = df[features]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def iqr_outlier_counts(df, features=None):
    """Conta, por variável, os valores fora do intervalo [Q1 - 1,5·IQR, Q3 + 1,5·IQR].

    Args:
        df: DataFrame com as variáveis numéricas.
        features: colunas a analisar.

    Returns:
        DataFrame com a quantidade e o percentual de outliers por variável.
    """
    features = features or FEATURES
    data = df[features]
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    mask = (data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)
    result = pd.DataFrame({
        "outliers": mask.sum(),
        "percentual": (mask.mean() * 100).round(1),
    })
    return result.sort_values("outliers", ascending=False)
