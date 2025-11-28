from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    # 1. Load the Olivetti faces dataset
    data = fetch_olivetti_faces()
    X = data.data      # Flattened images
    y = data.target    # Labels (person IDs)

    # 2. Train-test split: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. Train DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluate on train just to see it works
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy: {train_acc:.4f}")

    # 5. Save model using joblib as savedmodel.pth
    joblib.dump(clf, "savedmodel.pth")
    print("Model saved as savedmodel.pth")

if __name__ == "__main__":
    main()
