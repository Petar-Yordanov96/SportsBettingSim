import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from data_preprocessing import load_and_clean_data
from feature_engineering import create_features

def train_model():
    df, _ = load_and_clean_data("data/matches.csv")
    X, y = create_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #splits the data into training and testing sets

    model = LogisticRegression(max_iter=1000) #choose model type
    model.fit(X_train, y_train)               #trains the model

    preds = model.predict(X_test)              #predictions based on the model for matches it has not seen
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)               #saves the trained model

if __name__ == "__main__":
    train_model()
