# Training and prediction configuration
TRAINING_DATA_CONFIG = "./config/training_data_setup.csv"
FEATURE_CONFIG = "./config/feature_config.csv"

# Paths to save/load model artifacts
MODEL_DIR  = "./ai_models"
# SCALER_PATH = "scaler.pkl"
# EXPECTED_FEATURES_PATH = "expected_features.pkl"

# Paths to load test data and save results
TESTING_DATA_PATH = "./test"
RESULT_PATH = "prediction_results.csv"
# TESTING_DATA_PATH = "./test-2"
# RESULT_PATH = "prediction_results_new.csv"

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