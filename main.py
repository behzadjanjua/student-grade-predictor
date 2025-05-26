import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
def load_data(path="data/student-mat.csv"):
    df = pd.read_csv(path, sep=';')
    return df

# Preprocess dataset
def preprocess_data(df):
    df = df.copy()
    
    # Turn final grade into binary: pass (>=10) = 1, fail (<10) = 0
    df["G3"] = (df["G3"] >= 10).astype(int)

    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.drop("G3", axis=1)
    y = df["G3"]
    
    return X, y, label_encoders

# Train and evaluate models
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='binary'):.2f}")
        
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"{name} Confusion Matrix")
        plt.show()

if __name__ == "__main__":
    df = load_data()
    X, y, encoders = preprocess_data(df)
    train_and_evaluate(X, y)