# Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from my_config import *
import os
import warnings
from extract_features import extract_features
from sklearn.metrics import classification_report
import joblib

# Load selected folders
def load_training_folders(setup_file):
	setup_df = pd.read_csv(setup_file)

	if "select" in setup_df.columns:
		setup_df = setup_df[setup_df["select"].str.lower().isin(["yes", "1"])]
	
	return setup_df["folder"].tolist()

def get_label(filename):
	if "cpu" in filename.lower() and "memory" in filename.lower():
		return "CPU-Memory-mixed"
	elif "cpu" in filename.lower():
		return "CPU intensive"
	elif "memory" in filename.lower():
		return "Memory intensive"
	elif "disk" in filename.lower():
		return "Disk intensive"
	elif "network" in filename.lower():
		return "Network intensive"
	else:
		return "N/A"  # Skip files that don't match expected labels

warnings.simplefilter("ignore", category=RuntimeWarning)
# Load folders
folders = load_training_folders(TRAINING_DATA_CONFIG)

for group_name, selected_columns in FEATURE_GROUPS.items():
	X, y = [], []
	for folder in folders:
		for file in os.listdir(folder):
			if file.endswith(".csv"):
				full_path = os.path.join(folder, file)
				df = pd.read_csv(full_path)
				# Ensure all required features exist, fill missing with 0
				missing_cols = [col for col in selected_columns if col not in df.columns]
				# Add dummy columns for missing features
				for col in missing_cols:
					df[col] = 0  # or use np.nan if needed

				feats = extract_features(df, selected_columns)
				label = get_label(file)
				if feats:
					X.append(feats)
					y.append(label)
	if X:
		df_X = pd.DataFrame(X)
		# Handle missing values with mean values
		df_X = df_X.fillna(df_X.mean()) 

		# Split the dataset
		X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.2, random_state=42)
		
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)

		# Train the SVM classifier
		svm_model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, random_state=42,probability=True)
		svm_model.fit(X_train_scaled, y_train)

		# Make predictions
		y_pred = svm_model.predict(X_test_scaled)

		# Evaluate the model
		print(group_name, " - Accuracy:", accuracy_score(y_test, y_pred))
		print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
		# print("Classification Report:\n", classification_report(y_test, y_pred))
		
		os.makedirs(MODEL_DIR, exist_ok=True)
		joblib.dump(svm_model, f"{MODEL_DIR}/model_svm_{group_name}_{SVM_KERNEL}_{SVM_GAMMA}_{SVM_C}.pkl")
		joblib.dump(scaler, f"{MODEL_DIR}/scaler_svm_{group_name}_{SVM_KERNEL}_{SVM_GAMMA}_{SVM_C}.pkl")
		joblib.dump(list(df_X.columns), f"{MODEL_DIR}/metrics_{group_name}_{SVM_KERNEL}_{SVM_GAMMA}_{SVM_C}.pkl")
