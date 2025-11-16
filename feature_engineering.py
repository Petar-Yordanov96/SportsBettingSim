import pandas as pd
import numpy as np

def create_features(df):
    # creates a new column odds_diff
    df['odds_diff'] = df['home_odds'] - df['away_odds']       #if <0 => home team is expected to win

    X = df[['home_id', 'away_id', 'odds_diff', 'home_odds', 'draw_odds', 'away_odds']]  #creates array based on that feature
    y = df['result']                                          #creates vector with the results, which the model would predict

    return X, y
