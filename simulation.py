import pickle
import pandas as pd
from data_preprocessing import load_and_clean_data
from feature_engineering import create_features

def simulate_betting(bet_amount=10, threshold=0.6):
    df, _ = load_and_clean_data("data/matches_test.csv")
    X, y = create_features(df)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    probs = model.predict_proba(X)     #returns a 2D array w/ each row = game, each col = probality for the classes [-1, 0 , 1] 
    preds = model.predict(X)           #returns a 1D array w/ the predicted classes for each game (-1 away win, 1 home win, 0 draw)

    balance = 1000
    for i, pred in enumerate(preds):
        prob = max(probs[i])           #saves the max probability among the 3 classes for each game
        if prob > threshold:           #places bet if the probability is larger than the threshold for betting    
            if pred == y.iloc[i]:
                balance += bet_amount * (df.iloc[i]['home_odds'] - 1)  #increases balance if bet is correct
            else:
                balance -= bet_amount                                  #decreases balance if bet is wrong

    print(f"Final balance: {balance:.2f}")

if __name__ == "__main__":
    simulate_betting()
