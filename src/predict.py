from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import pandas as pd

def main():

    model = joblib.load("models/model1.pkl")
    dataset = pd.read_csv("data/dataset.csv")

    _, test_X, _, test_y = train_test_split(dataset['feature'], dataset['target'], random_state=3)

    test_X = test_X.values.reshape(-1, 1)
    test_y = test_y.values.reshape(-1, 1)

    predictions = model.predict(test_X)

    mape = mean_absolute_percentage_error(test_y, predictions)

    print(f"Our mean absolute percentage error is {mape}%")

if __name__ == "__main__":
    main()