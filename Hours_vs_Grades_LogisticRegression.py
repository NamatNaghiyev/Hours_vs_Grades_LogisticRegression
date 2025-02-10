import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix

# 📌 Create the dataset: Study hours vs. grades (pass/fail)
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Number of hours studied
    'Grades': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]   # Binary outcome (0 = fail, 1 = pass)
}

df = pd.DataFrame(data)  # Convert dictionary to DataFrame

# 📌 Define features (X) and target variable (y)
X = df[['Hours']]  # Independent variable (study hours)
y = df['Grades']   # Dependent variable (grades: pass/fail)

# 📌 Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 📌 Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 📌 Predict outcomes for the test set
y_pred = model.predict(X_test)

# 📌 Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'The accuracy of the model: {accuracy:.2f}')

# 📌 Generate the confusion matrix to evaluate performance
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)

# 📌 Plot the data points and logistic regression curve
plt.scatter(X, y, color='red', label='Real data')  # Actual data points
plt.plot(X, model.predict_proba(X)[:, 1], color='blue', label='Logistic Regression')  # Regression curve

plt.xlabel('Hours')   # X-axis label
plt.ylabel('Grades')  # Y-axis label
plt.legend()          # Show legend
plt.tight_layout()    # Adjust layout for better visualization
plt.show()            # Display the plot
