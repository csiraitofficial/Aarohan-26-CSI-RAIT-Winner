import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# load dataset
df = pd.read_csv("pose_attention_dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]

# encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1
)

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# save model
joblib.dump(model, "xgboost_model12.pkl")
joblib.dump(encoder, "label_encoder.pkl")