import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,confusion_matrix


data = {
    'Hours':[1,2,3,4,5,6,7,8,9,10],
    'Grades':[0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Grades']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'The accuracy of the model: {accuracy:.2f}')

cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix')
print(cm)

plt.scatter(X,y, color='red',label='Real data')
plt.plot(X,model.predict_proba(X)[:,1],color='blue',label='Logistic Regression')
plt.xlabel('Hours')
plt.ylabel('Grades')
plt.legend()
plt.tight_layout()
plt.show()