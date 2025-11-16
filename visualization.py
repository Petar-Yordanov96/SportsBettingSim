import matplotlib.pyplot as plt
import pandas as pd
import pickle
from data_preprocessing import load_and_clean_data
from feature_engineering import create_features

def visualize_results():
    df, _ = load_and_clean_data("data/matches.csv")
    X, y = create_features(df)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X)

    accuracy = (preds == y).cumsum() / (range(1, len(y)+1))
    plt.plot(accuracy)
    plt.xlabel("Match #")
    plt.ylabel("Cumulative Accuracy")
    plt.title("Prediction Accuracy Over Time")
    plt.show()

if __name__ == "__main__":
    visualize_results()
