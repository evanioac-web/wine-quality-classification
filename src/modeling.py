"""Construção, ajuste de hiperparâmetros e treinamento dos modelos."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE


def get_cv(n_splits=5, random_state=RANDOM_STATE):
    """Validação cruzada K-Fold estratificada e embaralhada (shuffle=True)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=random_state)


def build_models(random_state=RANDOM_STATE):
    """Monta os pipelines e as grades de hiperparâmetros de cada modelo.

    Modelos baseados em distância (KNN, SVM) ou em coeficientes (Regressão
    Logística) recebem padronização (StandardScaler) dentro do Pipeline, de
    modo que a média e o desvio padrão sejam calculados apenas no treino de
    cada fold (sem vazamento de dados). Modelos de árvore não precisam de
    escalonamento.

    ``class_weight='balanced'`` dá mais peso à classe minoritária (vinhos de
    alta qualidade) durante o treino, compensando o desbalanceamento.

    Returns:
        Dicionário ``{nome: (pipeline, param_grid)}``.
    """
    return {
        "Regressão Logística": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced",
                                             max_iter=2000,
                                             random_state=random_state)),
            ]),
            {"model__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "KNN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(class_weight="balanced", probability=True,
                              random_state=random_state)),
            ]),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.01, 0.1, 1],
            },
        ),
        "Árvore de Decisão": (
            Pipeline([
                ("model", DecisionTreeClassifier(class_weight="balanced",
                                                 random_state=random_state)),
            ]),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [3, 4, 5, 6, 8],
                "model__min_samples_leaf": [5, 10, 20],
            },
        ),
        "Random Forest": (
            Pipeline([
                ("model", RandomForestClassifier(class_weight="balanced",
                                                 random_state=random_state,
                                                 n_jobs=-1)),
            ]),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 6, 8, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
    }


def tune_models(models, X_train, y_train, scoring="f1", cv=None):
    """Ajusta os hiperparâmetros de cada modelo com GridSearchCV.

    Args:
        models: dicionário retornado por :func:`build_models`.
        X_train: variáveis de treino.
        y_train: alvo de treino.
        scoring: métrica otimizada (F1 da classe de alta qualidade).
        cv: estratégia de validação cruzada.

    Returns:
        Dicionário ``{nome: GridSearchCV ajustado}``.
    """
    cv = cv or get_cv()
    fitted = {}
    for name, (pipeline, grid) in models.items():
        search = GridSearchCV(pipeline, grid, scoring=scoring, cv=cv,
                              n_jobs=-1)
        search.fit(X_train, y_train)
        fitted[name] = search
    return fitted


def calibrate_model(model, X_train, y_train, cv=None):
    """Calibra as probabilidades de um modelo já ajustado (método sigmoide).

    Com ``class_weight='balanced'`` o modelo "superestima" a chance da classe
    rara. A calibração, feita com validação cruzada apenas no treino, ajusta
    as probabilidades para que reflitam a frequência real de vinhos de alta
    qualidade — essencial para comunicar "chance" a gestores.

    Returns:
        ``CalibratedClassifierCV`` ajustado.
    """
    cv = cv or get_cv()
    return CalibratedClassifierCV(model, cv=cv, method="sigmoid").fit(
        X_train, y_train)


def build_smote_models(random_state=RANDOM_STATE):
    """Versões dos modelos com oversampling (SMOTE) no lugar de class_weight.

    O SMOTE cria vinhos de alta qualidade "sintéticos" (interpolando vizinhos
    reais) para equilibrar as classes. Ele fica DENTRO do Pipeline do
    imbalanced-learn, então só é aplicado aos dados de treino de cada fold —
    a validação continua sendo feita apenas com vinhos reais.

    Returns:
        Dicionário ``{nome: (pipeline, param_grid)}``.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    return {
        "Regressão Logística + SMOTE": (
            ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=random_state)),
                ("model", LogisticRegression(max_iter=2000,
                                             random_state=random_state)),
            ]),
            {"model__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "Random Forest + SMOTE": (
            ImbPipeline([
                ("smote", SMOTE(random_state=random_state)),
                ("model", RandomForestClassifier(random_state=random_state,
                                                 n_jobs=-1)),
            ]),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 6, 8, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
    }
