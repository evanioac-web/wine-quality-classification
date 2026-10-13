"""Configurações centrais do projeto (caminhos, sementes e constantes)."""

from pathlib import Path

# Raiz do repositório (pasta que contém data/, src/, notebooks/ e results/)
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "WineQT.csv"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Semente aleatória fixa para garantir reprodutibilidade dos resultados
RANDOM_STATE = 42

# Proporção da base reservada para teste (hold-out)
TEST_SIZE = 0.2

# Nota de corte: vinhos com quality >= 7 são considerados de alta qualidade
QUALITY_THRESHOLD = 7

TARGET = "alta_qualidade"

# Variáveis físico-químicas originais do dataset
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Tradução das variáveis para gráficos e relatórios em português
FEATURE_LABELS = {
    "fixed acidity": "Acidez fixa",
    "volatile acidity": "Acidez volátil",
    "citric acid": "Ácido cítrico",
    "residual sugar": "Açúcar residual",
    "chlorides": "Cloretos",
    "free sulfur dioxide": "SO₂ livre",
    "total sulfur dioxide": "SO₂ total",
    "density": "Densidade",
    "pH": "pH",
    "sulphates": "Sulfatos",
    "alcohol": "Teor alcoólico",
    "free_so2_ratio": "Proporção SO₂ livre",
    "total_acidity": "Acidez total",
    "alcohol_x_sulphates": "Álcool × sulfatos",
}

# Paleta (azul = alta qualidade, laranja = baixa/média) — validada para daltonismo
COLOR_HIGH = "#2a78d6"
COLOR_LOW = "#eb6834"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
