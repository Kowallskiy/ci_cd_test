from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import joblib
import pandas as pd

def main():
    model = joblib.load("models/model.pkl")
    dataset = pd.read_csv("data/dataset.csv")

    train_X, _, train_y, _ = train_test_split(dataset['feature'], dataset['target'], random_state=3)

    train_X = train_X.values.reshape(-1, 1)
    train_y = train_y.values.reshape(-1, 1)

    model.fit(train_X, train_y)

    joblib.dump(model, "models/model1.pkl")

if __name__ == "__main__":
    main()
