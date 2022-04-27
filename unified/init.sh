#!/bin/bash
# conda 换清华源

# cp .condarc ~/
# 从文件创建conda环境
# conda env create -f env.yaml
conda create -n gaiic2022 python==3.9

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

# 安装依赖
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install -r ./requirements.txt 

