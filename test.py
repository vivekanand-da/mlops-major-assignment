from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def main():
    # 1. Load saved model
    model = joblib.load("savedmodel.pth")
    print("Loaded model from savedmodel.pth")

    # 2. Load dataset again
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    # 3. Same split as training (VERY IMPORTANT: same random_state & test_size)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4. Predict on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
