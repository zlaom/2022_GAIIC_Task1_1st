# data preprocess
ORIGIN_DATA_DIR = "/home/haoquan/zlm/GAIIC_2022_1/attr_final/input/train/train"
PREPROCESS_DATA_DIR = (
    "/home/haoquan/zlm/GAIIC_2022_1/attr_final/temp/zlm_tmp_data/unequal_processed_data"
)

# train
MODEL_SAVE_PATH = "/home/haoquan/zlm/GAIIC_2022_final/project/best_model"

# test
ORIGIN_TEST_FILE = f"{ORIGIN_DATA_DIR}/preliminary_testA.txt"
PREPROCESS_TEST_FILE = f"{PREPROCESS_DATA_DIR}/preprocess_test4000.txt"
MODEL_SAVE_DIR = "/home/haoquan/zlm/GAIIC_2022_1/attr_final/project/best_model/final_unequal_attr/cat_s11_e60_b256_drop0.3_pos0.47/best"
RESULT_SAVE_DIR = "/home/haoquan/zlm/GAIIC_2022_1/attr_final/project/submission"
RESULT_SAVE_NAME = "attr_submission.txt"
