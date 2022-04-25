# 参数传递
GPUS=$1

# 获取conda路径
CONDA_CONFIG_ROOT_PREFIX=$(conda config --show root_prefix)
echo "CONDA_CONFIG_ROOT_PREFIX= ${CONDA_CONFIG_ROOT_PREFIX}"
get_conda_root_prefix() {
  TMP_POS=$(awk -v a="${CONDA_CONFIG_ROOT_PREFIX}" -v b="/" 'BEGIN{print index(a, b)}')
  TMP_POS=$((TMP_POS-1))
  if [ $TMP_POS -ge 0 ]; then
    echo "${CONDA_CONFIG_ROOT_PREFIX:${TMP_POS}}"
  else
    echo ""
  fi
}
CONDA_ROOT=$(get_conda_root_prefix)
echo "CONDA_ROOT= ${CONDA_ROOT}"

# 切换conda环境
source "${CONDA_ROOT}/bin/activate" gaiic2022

# 显示当前环境 *号代表当前环境
conda env list

# 数据预处理
echo "\nPreprocessing......\n"
python code/data_preprocess.py 