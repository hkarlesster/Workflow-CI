import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import KNNImputer
import missingno as msno
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from warnings import filterwarnings
warnings.filterwarnings("ignore")

df = pd.read_csv("heart_disease.csv")
df.head()

df.shape

df.info()

cat_cols = df.select_dtypes(include=["object"]).columns
print("Categorical Columns:", cat_cols)

for col in cat_cols:
    print(f"{col}: {df[col].unique()}")

fig, axes = plt.subplots(4, 3, figsize=(20, 20))
fig.tight_layout(pad=5.0)
axes = axes.flatten()

for i, col in enumerate(cat_cols[:12]):
    sns.countplot(x=col, data=df, hue=col, palette='Greens', dodge=False, ax=axes[i])
    axes[i].set_title(f'Plot Hitungan dari {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Sembunyikan subplot yang tidak digunakan
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.show()

num_cols = df.select_dtypes(include=[ "number" ]).columns
plt.figure(figsize=( 15 ,  5 ))
df[num_cols].boxplot()
plt.xticks(rotation= 90 )
plt.show()

df.isna(). sum ()

"""Periksa apakah ada duplikat"""
df.duplicated(). sum ()

"""Mengkodekan 12 variabel kategorikal (tipe objek) sebagai variabel numerik."""
from sklearn.preprocessing import LabelEncoder

categorical_cols = df.select_dtypes(include=['object']).columns

encoder = LabelEncoder()
label_mappings = {}

for col in categorical_cols:
    mask = df[col].notna()

    df.loc[mask, col] = encoder.fit_transform(df.loc[mask, col])

    label_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

print("Data info:")
print(df.info())

# Display label mappings
for col, mapping in label_mappings.items():
    print(f"Column: {col}")
    for label, code in mapping.items():
        print(f"{code} -> {label}")
    print()

"""Menangani data yang hilang menggunakan KNNImputer dari Scikit-learn. KNNImputer (n_neighbors=5) mengganti nilai yang hilang dengan menemukan lima tetangga terdekat untuk setiap entri yang hilang dan mengisi nilai yang hilang berdasarkan rata-ratanya."""
knn_imputer = KNNImputer(n_neighbors=5)

df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
print(df_imputed)
df = df_imputed

"""Membuat heatmap untuk memvisualisasikan korelasi antara fitur numerik dalam dataset. Ini membantu mengidentifikasi hubungan antar variabel, tetapi dalam kasus ini, tidak ada korelasi signifikan antara fitur-fitur tersebut."""
matriks_korelasi = df.corr()

plt.figure(figsize=(12, 6))
sns.heatmap(matriks_korelasi, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Peta Panas Korelasi Fitur")
plt.show()

"""Menskalakan semua fitur numerik ke rentang [0, 1] menggunakan MinMaxScaler untuk memastikan kontribusi yang sama terhadap pelatihan model. Variabel target (Status Penyakit Jantung) dipertahankan tanpa penskalaan karena merupakan label klasifikasi. Fitur yang telah diskalakan kemudian digabungkan kembali dengan kolom target asli untuk analisis lebih lanjut."""
scaler = MinMaxScaler()

Grade_column = df[ 'Heart Disease Status' ]
df_scaled = scaler.fit_transform(df)
df = pd.DataFrame(df_scaled, columns=df.columns)

df[ 'Heart Disease Status' ] = Grade_column

df

"""Buat diagram batang untuk menggambarkan distribusi variabel target Status Penyakit Jantung."""
print(df["Heart Disease Status"].value_counts())

sns.countplot(x=df["Heart Disease Status"])
plt.title("Target Variable Distribution")
plt.show()

"""Kita dapat dengan jelas melihat ketidakseimbangan kelas di mana sebagian besar sampel termasuk dalam kategori tidak sakit (0). Ketidakseimbangan ini dapat menyebabkan model yang bias yang menguntungkan kelas mayoritas, mengurangi kinerja prediksi untuk kelas minoritas (kasus penyakit jantung). Jadi, dengan menggunakan pandas, kami melakukan upsampling manual untuk kelas 1.0. Kami menyeimbangkan data menggunakan duplikasi. Ini memastikan model mempelajari kedua kelas secara setara, meningkatkan akurasi, recall, dan F1-score untuk kelas minoritas. Penyeimbangan data mencegah bias, membuat model lebih andal dan efektif dalam memprediksi kedua kelas."""
from sklearn.model_selection import train_test_split

# Pisahkan kelas mayoritas dan minoritas
df_majority = df[df[ 'Heart Disease Status' ] ==  0 ]
df_minority = df[df[ 'Heart Disease Status' ] ==  1 ]

# Tingkatkan jumlah sampel kelas minoritas
df_minority_upsampled = df_minority.sample(n= int ( len (df_majority) *  0.5 ), replace= True , random_state= 42 )

# Gabungkan keduanya
df_balanced = pd.concat([df_majority, df_minority_upsampled ])

# Acak dataset
df_balanced = df_balanced.sample(frac= 1 , random_state= 42 )

# Pisahkan menjadi fitur dan label
X = df_balanced.drop(columns=[ 'Heart Disease Status' ])
y = df_balanced[ 'Heart Disease Status' ]

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size= 0.2 , random_state= 42 )

# menyimpan data preprocessing ke dalam csv
df.to_csv( 'heart_disease_preprocessed.csv' , index= False )
print ( 'Data berhasil disimpan !' )

accuracy = {}

"""Random Forest"""
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10]
}



rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="accuracy", n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
accuracy["Random Forest"]=grid_search_rf.best_score_
best_rf = grid_search_rf.best_params_
best_rf.update({
    'random_state' : 42
})

print("Best Parameters for Random Forest:", best_rf)
print("Best Accuracy:", accuracy["Random Forest"])

"""Logistic Regression"""
param_grid_lr = {
    "C": [0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"]
}


lr = LogisticRegression(max_iter=3000, random_state=42)
grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring="accuracy", n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
accuracy["Logistic Regression"] = grid_search_lr.best_score_
best_lr = grid_search_lr.best_params_
best_lr.update({
    'max_iter': 3000,
    'random_state': 42
})

print("Best Parameters for Logistic Regression:", best_lr)
print("Best Accuracy:", accuracy["Logistic Regression"])

"""KNN"""
param_grid_knn = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"]
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring="accuracy", n_jobs=-1)
grid_search_knn.fit(X_train, y_train)
accuracy["KNN"] = grid_search_knn.best_score_
best_knn = grid_search_knn.best_params_


print("Best Parameters for KNN:", best_knn)
print("Best Accuracy:", accuracy["KNN"])

"""SVM"""
param_grid_svm = {
    "C": [0.1, 1, 10],  # Reduced range for speed
    "kernel": ["linear", "rbf"]  # Removed "poly" for faster execution
}

svm = SVC(random_state=42, max_iter=1000)  # Added max_iter to prevent infinite training
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)  # Added verbose
grid_search_svm.fit(X_train, y_train)
accuracy["SVM"] = grid_search_svm.best_score_
best_svm = grid_search_svm.best_params_
best_svm.update({
    'max_iter': 3000,
    'random_state': 42
})


print("Best Parameters for SVM:", best_svm)
print("Best Accuracy:", grid_search_svm.best_score_)

"""### Naïve Bayes"""
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy["Naive Bayes"] = accuracy_nb
print("Naïve Bayes Accuracy:", accuracy_nb)

"""### Decision Tree"""
param_grid_dt = {
    "max_depth": [5, 10, 20],
    "criterion": ["gini", "entropy"]
}

dt = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring="accuracy", n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
accuracy["Decision Tree"] = grid_search_dt.best_score_
best_dt = grid_search_dt.best_params_
best_dt.update({
    'random_state': 42
})


print("Best Parameters for Decision Tree:", best_dt)
print("Best Accuracy:", accuracy["Decision Tree"])

"""Mengekstraksi Presisi, Recall, dan F1 Score, menggunakan parameter terbaik."""
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

def cross_validate_model(model, x_train, y_train, cv=5):
    scoring = {
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1_score': make_scorer(f1_score, average='weighted', zero_division=0)
    }

    cv_results = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring)

    precision = cv_results['test_precision'].mean()
    recall = cv_results['test_recall'].mean()
    f1 = cv_results['test_f1_score'].mean()

    print(f"Cross-Validation Results for {model.__class__.__name__}:\n")
    print(f"Average Precision: {precision:.4f}")
    print(f"Average Recall: {recall:.4f}")
    print(f"Average F1-Score: {f1:.4f}")
    print("="*50, "\n")

    return precision, recall, f1

metrics = {}

models = {
    "Random Forest": RandomForestClassifier(**best_rf, class_weight="balanced"),
    "Logistic Regression": LogisticRegression(**best_lr, class_weight="balanced"),
    "SVM": SVC(**best_svm, class_weight="balanced"),

    "KNN": KNeighborsClassifier(**best_knn),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(**best_dt, class_weight="balanced"),
}

for name, model in models.items():
    print(f"Cross-Validating {name}...")
    precision, recall, f1 = cross_validate_model(model, X_train, y_train, cv=5)
    accuracy_sc = accuracy[name]
    metrics[name] = [accuracy_sc, precision, recall, f1]

"""Membuat laporan klasifikasi menggunakan parameter terbaik."""
def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)  # Train the model
    y_pred = model.predict(x_test)  # Predict on test data

    report = classification_report(y_test, y_pred, zero_division=1)  # Get precision, recall, f1-score
    print(f"Evaluation for {model.__class__.__name__}:\n")
    print(report)
    print("="*50, "\n")

# Define models
models = {
    "Random Forest": RandomForestClassifier(**best_rf, class_weight="balanced"),
    "Logistic Regression": LogisticRegression(**best_lr, class_weight="balanced"),
    "SVM": SVC(**best_svm, class_weight="balanced"),

    "KNN": KNeighborsClassifier(**best_knn),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(**best_dt, class_weight="balanced"),
}

# Run evaluation for each model
for name, model in models.items():
    evaluate_model(model, X_train, y_train, X_test, y_test)

"""Membuat bagan perbandingan"""
import matplotlib.pyplot as plt
import numpy as np



labels = list(metrics.keys())
accuracy_vals = [metrics[m][0] for m in labels]
precision_vals = [metrics[m][1] for m in labels]
recall_vals = [metrics[m][2] for m in labels]
f1_vals = [metrics[m][3] for m in labels]

x = np.arange(len(labels))  # label locations
width = 0.2  # width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5*width, accuracy_vals, width, label='Accuracy', color='#b5ead7')
ax.bar(x - 0.5*width, precision_vals, width, label='Precision', color='#a3d9a5')
ax.bar(x + 0.5*width, recall_vals, width, label='Recall', color='#d0f0c0')
ax.bar(x + 1.5*width, f1_vals, width, label='F1-score', color='#c1e1c1')

ax.set_ylabel('Score')
ax.set_title('Model Comparison: Accuracy, Precision, Recall, and F1-score')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()


plt.tight_layout()
plt.show()

"""Kita dapat melihat bahwa Random Forest berkinerja jauh lebih baik daripada model-model lainnya."""
feature_importances = models["Random Forest"].feature_importances_

feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})

importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color='#a3d9a5')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance from Random Forest")
plt.gca().invert_yaxis()
plt.show()

print(importance_df.head(10))

from sklearn.metrics import confusion_matrix, roc_curve, auc

rf_model = RandomForestClassifier(max_depth=20, min_samples_split=2, n_estimators=200, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',  # Changed to green colormap
    xticklabels=['Predicted 0', 'Predicted 1'],
    yticklabels=['Actual 0', 'Actual 1']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(
    fpr,
    tpr,
    color='green',
    lw=2,
    label=f'ROC Curve (AUC = {roc_auc:.2f})'
)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Changed baseline to gray
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc='lower right')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler explicitly

# Re-initialize and fit the scaler on X_train (features only)
scaler = MinMaxScaler()
scaler.fit(X_train) # Fit only on X_train to avoid data leakage

# Scale X_train and X_test using the newly fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inisialisasi model Random Forest terbaik dengan parameter yang tepat
# Using best_rf from grid search results and adding class_weight as used in previous steps
best_model = RandomForestClassifier(**best_rf, class_weight="balanced")

# Melatih model terbaik pada data pelatihan yang telah discaling
best_model.fit(X_train_scaled, y_train)

# Gunakan model ini untuk membuat prediksi
predictions = best_model.predict(X_test_scaled)

"""## Save Model"""

import pickle

# Simpan skaler ke dalam file 'scaler_ds.pkl'
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Simpan model terbaik ke dalam file 'model_ds.pkl'
with open('model_rf.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

"""## Testing"""

import numpy as np
import pickle

Age = 0.5
Gender = 1.0 # Male
Blood_Pressure = 0.3
Cholesterol_Level = 0.6
Exercise_Habits = 0.5 # Medium
Smoking = 1.0 # Yes
Family_Heart_Disease = 1.0 # Yes
Diabetes = 0.0 # No
BMI = 0.4
High_Blood_Pressure = 0.0 # No
Low_HDL_Cholesterol = 0.0 # No
High_LDL_Cholesterol = 1.0 # Yes
Alcohol_Consumption = 0.5 # Medium
Stress_Level = 1.0 # Medium
Sleep_Hours = 0.7
Sugar_Consumption = 0.0 # High
Triglyceride_Level = 0.5
Fasting_Blood_Sugar = 0.4
CRP_Level = 0.3
Homocysteine_Level = 0.6

data = [
    [
        Age, Gender, Blood_Pressure, Cholesterol_Level, Exercise_Habits,
        Smoking, Family_Heart_Disease, Diabetes, BMI, High_Blood_Pressure,
        Low_HDL_Cholesterol, High_LDL_Cholesterol, Alcohol_Consumption,
        Stress_Level, Sleep_Hours, Sugar_Consumption, Triglyceride_Level,
        Fasting_Blood_Sugar, CRP_Level, Homocysteine_Level
    ]
]

# Load the previously saved model and scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))
best_model = pickle.load(open('model_rf.pkl', 'rb'))

# Standardize the data
data_scaled = scaler.transform(data)

# Predict the Status
hasil_prediksi = best_model.predict(data_scaled)
hasil_prediksi = int(hasil_prediksi[0]) # Get the single prediction value

# Map prediction result to appropriate label
if hasil_prediksi == 0:
    status = "No Heart Disease"
elif hasil_prediksi == 1:
    status = "Heart Disease"
else:
    status = "Unknown Status"

# Display the prediction result
print(f"Hasil Prediksi Status: {status}")