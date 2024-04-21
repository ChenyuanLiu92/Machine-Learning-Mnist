from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import numpy as np
import time

start_time = time.time()


# Load MNIST
mnist = fetch_openml('mnist_784',version=1)
X = mnist.data.astype('float32')
y = np.array(mnist.target.astype('int'))

# Pre-operation
X /= 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training
DT_model = DecisionTreeClassifier(random_state=32)
DT_model.fit(X_train, y_train)

# prediction
y_pred = DT_model.predict(X_test)

accuracy_score = accuracy_score(y_test, y_pred)
print(f"Error is {1-accuracy_score}")

end_time = time.time()
print(f"Running time is {end_time-start_time} s")



