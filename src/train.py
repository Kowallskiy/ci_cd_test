from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def main():
    model = joblib.load("models/model.pkl")
    dataset = pd.read_csv("data/dataset.csv")

    train_X, test_X, train_y, test_y = train_test_split(dataset['feature'], dataset['target'], random_state=3)

    model.fit(train_X, train_y)

    joblib.dump(model, "models/model1.pkl")

if __name__ == "__main__":
    main()
