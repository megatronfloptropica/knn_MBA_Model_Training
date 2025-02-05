#เอา inter ลบ race แบบnormalize ข้อมูล

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# อ่านข้อมูลจากไฟล์
xlsx = pd.ExcelFile("MBAdataset.xlsx")
df = pd.read_excel(xlsx)

# Preprocess ข้อมูล
df.drop(['race'], axis=1, inplace=True)
# เปลี่ยนข้อมูล categorical เป็นตัวเลข
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['major'] = df['major'].map({'Humanities': 0, 'STEM': 1, 'Business': 2})
df['international'] = df['international'].map({False: 0, True: 1})
df['work_industry'] = df['work_industry'].map({
    'Consulting': 0, 'PE/VC': 0, 'Investment Banking': 0, 'Investment Management': 0, 'Financial Services': 0,
    'Technology': 1, 'Health Care': 1, 'Media/Entertainment': 1,
    'Nonprofit/Gov': 2, 'Real Estate': 2, 'Energy': 2, 'CPG': 2, 'Retail': 2,
    'Other': 3
})
df['admission'] = df['admission'].fillna(0)
df['admission'] = df['admission'].map({0: 0, 'Waitlist': 0, 'Admit': 1})
# Normalize ข้อมูล
cols_to_normalize = ['gpa', 'gmat', 'work_exp']
scaler = MinMaxScaler()
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# แยก features และ target
x = df.drop(columns=['application_id', 'admission'])
y = df['admission']

# แบ่งข้อมูลเป็น train และ test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# สร้าง kNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# ทำนายผล
y_pred = knn.predict(x_test)

# คำนวณความแม่นยำ (Accuracy)
accuracy = accuracy_score(y_test, y_pred)

# ประเมินความถูกต้องของโมเดล
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# แสดง confusion matrix เป็นรูปหรือกราฟ
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(f"ความแม่นยำ (Accuracy) ของโมเดล kNN คือ {accuracy:}")
print("\nClassification Report:")
print(class_report)
