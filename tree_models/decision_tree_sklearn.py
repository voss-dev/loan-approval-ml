from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def main():
    # 1. Load benchmark dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Decision Tree Classifier (With entropy and depth regularization)
    dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=5, random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_preds = dt_clf.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)

    print("1. Decision Tree Accuracy:", f"{dt_acc * 100:.2f}%")
    print("\n--- Tree Structure Rules ---")
    print(export_text(dt_clf, feature_names=list(feature_names), ))

    # 3. Random Forest Classifier (Bagging + Feature Subsampling)
    rf_clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_preds = rf_clf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)


    # 4. AdaBoost Classifier (Boosting with decision stumps)
    ada_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
    ada_clf.fit(X_train, y_train)
    ada_acc = accuracy_score(y_test, ada_clf.predict(X_test))

    # 5. Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_clf.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb_clf.predict(X_test))

    print("\n" + "=" * 45)
    print(f"Random Tree Accuracy:     {dt_acc * 100:.2f}%")
    print(f"Random Forest Accuracy:   {rf_acc * 100:.2f}%")
    print(f"AdaBoost Accuracy:        {ada_acc * 100:.2f}%")
    print(f"Gradient Boosting Accuracy: {gb_acc * 100:.2f}%")


if __name__ == "__main__":
    main()