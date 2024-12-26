import toml

NOT_PROVIDED = NOT_APPLICABLE = NOT_DEFINED = object()

# Load the configuration from the config.toml file
config = toml.load("config.toml")

# Access the configurations
LINES_COUNT_TO_READ = config["general"]["LINES_COUNT_TO_READ"]
TRAIN_TEST_SPLIT_RATIO = config["general"]["TRAIN_TEST_SPLIT_RATIO"]

ALS_HYPER_N_EPOCH = config["als"]["HYPER_N_EPOCH"]
ALS_HYPER_N_FACTOR = config["als"]["HYPER_N_FACTOR"]
ALS_HYPER_GAMMA = config["als"]["HYPER_GAMMA"]
ALS_HYPER_LAMBDA = config["als"]["HYPER_LAMBDA"]
ALS_HYPER_TAU = config["als"]["HYPER_TAU"]
ALS_CHECKPOINT_FOLDER = config["als"]["CHECKPOINT_FOLDER"]

PLT_FIGURE_FOLDER = config["figures"]["PLT_FIGURE_FOLDER"]
PLT_FIGURE_FORMAT = config["figures"]["PLT_FIGURE_FORMAT"]
