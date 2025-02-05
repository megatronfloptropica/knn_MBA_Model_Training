#เอา race แบบquartile

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# อ่านข้อมูลจากไฟล์
xlsx = pd.ExcelFile("MBAdataset.xlsx")
df = pd.read_excel(xlsx)

# Preprocess ข้อมูล
df = df.dropna(subset=['race'])
df.drop(['international'], axis=1, inplace=True)
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['major'] = df['major'].map({'Humanities': 0, 'STEM': 1, 'Business': 2})
df['race'] = df['race'].map({'White': 0, 'Asian': 1, 'Black': 2, 'Hispanic': 3, 'Other': 3})
df['work_industry'] = df['work_industry'].map({'Consulting': 0, 'PE/VC': 0, 'Investment Banking': 0, 'Investment Management': 0, 'Financial Services': 0,
                                               'Technology': 1, 'Health Care': 1, 'Media/Entertainment': 1,
                                               'Nonprofit/Gov': 2, 'Real Estate': 2, 'Energy': 2, 'CPG': 2, 'Retail': 2,
                                               'Other': 3})
df['admission'] = df['admission'].fillna(0)
df['admission'] = df['admission'].map({0: 0, 'Waitlist': 0, 'Admit': 1})
#แบ่ง Quartile
df['gpa'] = pd.qcut(df['gpa'], q=4, labels=['0', '1', '2', '3'])
df['gmat'] = pd.qcut(df['gmat'], q=4, labels=['0', '1', '2', '3'])

# แยก features และ target
X = df.drop(columns=['application_id', 'admission'])
y = df['admission']

# Normalize ข้อมูล
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# แบ่งข้อมูลเป็น train และ test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# สร้าง kNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ทำนายผล
y_pred = knn.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print("\nAccuracy:", accuracy)

