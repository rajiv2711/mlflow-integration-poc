import mlflow, numpy as np, matplotlib
matplotlib.use("Agg")             # headless backend for Docker
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


def train_linear(alpha: float = 0.01, epochs: int = 50):
    mlflow.set_experiment("demo-linear1")

    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xtr, Xte = scaler.fit_transform(Xtr), scaler.transform(Xte)

    with mlflow.start_run() as run:
        mlflow.log_params({"alpha": alpha, "epochs": epochs})

        model = SGDRegressor(
            alpha=alpha,
            max_iter=1,
            tol=None,                       # ✔️  no early-stop check
            learning_rate="constant",
            eta0=0.01,
            random_state=42,
        )

        losses = []
        for epoch in range(epochs):
            model.partial_fit(Xtr, ytr)
            mse = mean_squared_error(ytr, model.predict(Xtr))
            losses.append(mse)
            mlflow.log_metric("train_mse", mse, step=epoch)

        test_mse = mean_squared_error(yte, model.predict(Xte))
        mlflow.log_metric("test_mse", test_mse)

        # save learning curve
        plt.figure()
        plt.plot(range(epochs), losses)
        plt.xlabel("epoch")
        plt.ylabel("train_mse")
        plt.title("Learning Curve")
        plt.tight_layout()
        plt.savefig("learning_curve.png")
        plt.close()
        mlflow.log_artifact("learning_curve.png")

        return {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id}
