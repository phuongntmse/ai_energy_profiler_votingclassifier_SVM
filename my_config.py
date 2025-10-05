# Training and prediction configuration
TRAINING_DATA_CONFIG = "./config/training_data_setup.csv"
FEATURE_CONFIG = "./config/feature_config.csv"

# Paths to save/load model artifacts
MODEL_DIR  = "./ai_models"
# SCALER_PATH = "scaler.pkl"
# EXPECTED_FEATURES_PATH = "expected_features.pkl"

# Paths to load test data and save results
RESULT_DIR = "./all-results"
TESTING_DATA_PATH = "./test"
RESULT_PATH = "prediction_results_seenbm_sigmoid_scale_1000.csv"
# TESTING_DATA_PATH = "./test-unseen"
# RESULT_PATH = "prediction_results_unseenbm_poly_scale_10.csv"
# TESTING_DATA_PATH = "./test-grid5000"
# RESULT_PATH = "prediction_results_seenbm_grid5000_sigmoid_scale_1000.csv"

# Simulate anomalies setup
INPUT_DATA_PATH = "./data/network-2"
OUTPUT_DATA_PATH = "./data/network-2-anomalies-injected"
COLUMNS_TO_ALTER = ["CpuALL_usage", "cpu_frequency","Energy_usage_during_time","memory_usage_perce","write_rate_during_time","read_rate_during_time"]
ANOMALY_FRACTION = 0.1
MODIFIERS = [2, 3, 5] 

FEATURE_GROUPS = {
	"CPU": ["CpuALL_usage", "cpu_frequency"],
	"MEMORY": ["memory_usage"],	
	"DISK_IO": ["Writes_on_sda","Reads_on_sda"],
	"ENERGY": ["Energy_usage_during_time"],
	"ALL": ["CpuALL_usage","cpu_frequency","Energy_usage_during_time","memory_usage","Writes_on_sda","Reads_on_sda"]
}

# SVM parameters
SVM_C = 1000
SVM_GAMMA='scale'
SVM_KERNEL='sigmoid'