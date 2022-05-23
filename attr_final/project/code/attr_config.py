# data preprocess
ORIGIN_DATA_DIR = "/home/haoquan/zlm/GAIIC_2022_1/attr_final/input/train/train"
PREPROCESS_DATA_DIR = (
    "/home/haoquan/zlm/GAIIC_2022_1/attr_final/temp/zlm_tmp_data/unequal_processed_data"
)

# train
ATTR_MODEL_SAVE_DIR = (
    "/home/haoquan/zlm/GAIIC_2022_1/attr_final/project/best_model/final_attr"
)
# test
ORIGIN_TEST_FILE = f"{ORIGIN_DATA_DIR}/preliminary_testA.txt"
PREPROCESS_TEST_FILE = f"{PREPROCESS_DATA_DIR}/preprocess_test4000.txt"
RESULT_SAVE_DIR = "/home/haoquan/zlm/GAIIC_2022_1/attr_final/project/submission"
RESULT_SAVE_NAME = "attr_submission.txt"
