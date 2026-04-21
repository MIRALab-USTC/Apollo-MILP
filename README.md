# **Apollo-MILP**

This is the code of paper **Apollo-MILP: An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear Programming**. Haoyang Liu, Jie Wang, Zijie Geng, Xijun Li, Yuxuan Zong, Fangzhou Zhu, Jianye HAO, Feng Wu. ICLR 2025. 

The code is build on top of https://github.com/sribdcn/Predict-and-Search_MILP_method

## News

🚀🚀 Apollo-MILP has now been integrated into MILP-X (https://github.com/happypu326/MILP-X). MILP-X is a unified machine learning framework for MILP that supports data generation, training, evaluation, and benchmarking across 16+ problem types and multiple learning methods. This integration makes it easier for researchers to reproduce Apollo-MILP, compare it with other approaches under a common framework, and conduct more systematic experiments on ML for MILP.

## Dependencies

- Python 3.9
- PyTorch 1.12.1
- torch-geometric 2.5.3
- pyscipopt 4.2.0
- gurobipy 9.5.2 
- ecole 0.8.1

## Environments

```
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       2_gnu    conda-forge
_sysroot_linux-64_curr_repodata_hack 3                   h69a702a_16    conda-forge
absl-py                   2.1.0                    pypi_0    pypi
aiohttp                   3.9.5                    pypi_0    pypi
aiosignal                 1.3.1                    pypi_0    pypi
ampl-mp                   3.1.0             h2cc385e_1006    conda-forge
antlr4-python3-runtime    4.9.3                    pypi_0    pypi
asttokens                 2.4.1                    pypi_0    pypi
async-timeout             4.0.3                    pypi_0    pypi
attrs                     23.2.0                   pypi_0    pypi
binutils_impl_linux-64    2.40                 ha1999f0_7    conda-forge
binutils_linux-64         2.40                 hb3c18ed_0    conda-forge
blinker                   1.8.2                    pypi_0    pypi
boost                     1.85.0               he8689d4_2    conda-forge
bzip2                     1.0.8                h4bc722e_7    conda-forge
ca-certificates           2024.7.4             hbcca054_0    conda-forge
cachetools                5.4.0                    pypi_0    pypi
certifi                   2024.7.4                 pypi_0    pypi
charset-normalizer        3.3.2                    pypi_0    pypi
click                     8.1.7                    pypi_0    pypi
cliquer                   1.22                 hd590300_1    conda-forge
community                 1.0.0b1                  pypi_0    pypi
contourpy                 1.2.1                    pypi_0    pypi
cppad                     20230000.0           h59595ed_2    conda-forge
cycler                    0.12.1                   pypi_0    pypi
cython                    0.29.24                  pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
docker-pycreds            0.4.0                    pypi_0    pypi
ecole                     0.8.1            py39hfbe7800_3    conda-forge
exceptiongroup            1.2.2                    pypi_0    pypi
executing                 2.1.0                    pypi_0    pypi
flask                     3.0.3                    pypi_0    pypi
fmt                       10.2.1               h00ab1b0_0    conda-forge
fonttools                 4.53.1                   pypi_0    pypi
frozenlist                1.4.1                    pypi_0    pypi
fsspec                    2024.6.1                 pypi_0    pypi
gcc                       14.1.0               h6f9ffa1_0    conda-forge
gcc_impl_linux-64         14.1.0               h3c94d91_0    conda-forge
gcc_linux-64              14.1.0               h3f71edc_0    conda-forge
gcg                       3.5.3                hd47b8d6_2    conda-forge
gitdb                     4.0.11                   pypi_0    pypi
gitpython                 3.1.43                   pypi_0    pypi
gmp                       6.3.0                hac33072_2    conda-forge
grpcio                    1.65.1                   pypi_0    pypi
gsl                       2.7                  he838d99_0    conda-forge
gurobipy                  9.5.2                    pypi_0    pypi
gxx                       14.1.0               h6f9ffa1_0    conda-forge
gxx_impl_linux-64         14.1.0               h2879b86_0    conda-forge
gxx_linux-64              14.1.0               hc55ae77_0    conda-forge
hydra-core                1.3.1                    pypi_0    pypi
icu                       73.2                 h59595ed_0    conda-forge
idna                      3.7                      pypi_0    pypi
importlib-metadata        8.2.0                    pypi_0    pypi
importlib-resources       6.4.0                    pypi_0    pypi
intervaltree              3.1.0                    pypi_0    pypi
ipdb                      0.13.13                  pypi_0    pypi
ipopt                     3.14.12              hf9e1ecf_0    conda-forge
ipython                   8.18.1                   pypi_0    pypi
itsdangerous              2.2.0                    pypi_0    pypi
jedi                      0.19.1                   pypi_0    pypi
jinja2                    3.1.4                    pypi_0    pypi
joblib                    1.4.2                    pypi_0    pypi
kernel-headers_linux-64   3.10.0              h4a8ded7_16    conda-forge
kiwisolver                1.4.5                    pypi_0    pypi
ld_impl_linux-64          2.40                 hf3520f5_7    conda-forge
libblas                   3.9.0           23_linux64_openblas    conda-forge
libboost                  1.85.0               hba137d9_2    conda-forge
libboost-devel            1.85.0               h00ab1b0_2    conda-forge
libboost-headers          1.85.0               ha770c72_2    conda-forge
libboost-python           1.85.0           py39h85c637f_2    conda-forge
libboost-python-devel     1.85.0           py39he8689d4_2    conda-forge
libcblas                  3.9.0           23_linux64_openblas    conda-forge
libecole                  0.8.1                hd2e6256_3    conda-forge
libedit                   3.1.20191231         he28a2e2_2    conda-forge
libffi                    3.4.4                h6a678d5_1  
libgcc-devel_linux-64     14.1.0             h5d3d1c9_100    conda-forge
libgcc-ng                 14.1.0               h77fa898_0    conda-forge
libgfortran-ng            14.1.0               h69a702a_0    conda-forge
libgfortran5              14.1.0               hc5f4f2c_0    conda-forge
libgomp                   14.1.0               h77fa898_0    conda-forge
libhwloc                  2.9.3           default_h554bfaf_1009    conda-forge
libiconv                  1.17                 hd590300_2    conda-forge
liblapack                 3.9.0           23_linux64_openblas    conda-forge
libnsl                    2.0.1                hd590300_0    conda-forge
libopenblas               0.3.27          pthreads_hac2b453_1    conda-forge
libsanitizer              14.1.0               hcba0ae0_0    conda-forge
libscotch                 7.0.4                h2fe6a88_5    conda-forge
libspral                  2023.09.07           h6aa6db2_2    conda-forge
libsqlite                 3.45.2               h2797004_0    conda-forge
libstdcxx-devel_linux-64  14.1.0             h5d3d1c9_100    conda-forge
libstdcxx-ng              14.1.0               hc0a3c3a_0    conda-forge
libuuid                   2.38.1               h0b41bf4_0    conda-forge
libxcrypt                 4.4.36               hd590300_1    conda-forge
libxml2                   2.12.7               hc051c1a_1    conda-forge
libzlib                   1.2.13               h4ab18f5_6    conda-forge
markdown                  3.6                      pypi_0    pypi
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.9.1                    pypi_0    pypi
matplotlib-inline         0.1.7                    pypi_0    pypi
metis                     5.1.0             h59595ed_1007    conda-forge
multidict                 6.0.5                    pypi_0    pypi
mumps-include             5.2.1               ha770c72_14    conda-forge
mumps-seq                 5.2.1               h2104b81_11    conda-forge
ncurses                   6.4                  h6a678d5_0  
networkx                  3.2.1                    pypi_0    pypi
numpy                     1.26.4           py39h474f0d3_0    conda-forge
nvidia-ml-py              12.535.161               pypi_0    pypi
nvitop                    1.3.2                    pypi_0    pypi
omegaconf                 2.3.0                    pypi_0    pypi
openssl                   3.3.1                h4bc722e_2    conda-forge
packaging                 24.1                     pypi_0    pypi
pandas                    2.2.2                    pypi_0    pypi
parso                     0.8.4                    pypi_0    pypi
pexpect                   4.9.0                    pypi_0    pypi
pillow                    10.4.0                   pypi_0    pypi
pip                       24.0             py39h06a4308_0  
platformdirs              4.2.2                    pypi_0    pypi
prompt-toolkit            3.0.47                   pypi_0    pypi
protobuf                  4.25.4                   pypi_0    pypi
psutil                    6.0.0                    pypi_0    pypi
ptyprocess                0.7.0                    pypi_0    pypi
pure-eval                 0.2.3                    pypi_0    pypi
pygcgopt                  0.1.4            py39h5a03fae_3    conda-forge
pygments                  2.18.0                   pypi_0    pypi
pyparsing                 3.1.2                    pypi_0    pypi
pyscipopt                 4.2.0            py39h5a03fae_3    conda-forge
python                    3.9.18          h0755675_1_cpython    conda-forge
python-dateutil           2.9.0.post0              pypi_0    pypi
python-louvain            0.16                     pypi_0    pypi
python_abi                3.9                      4_cp39    conda-forge
pytz                      2024.1                   pypi_0    pypi
pyyaml                    6.0.1                    pypi_0    pypi
readline                  8.2                  h5eee18b_0  
requests                  2.32.3                   pypi_0    pypi
scikit-learn              1.5.1                    pypi_0    pypi
scip                      8.0.3                h4a32fe0_2    conda-forge
scipy                     1.13.1                   pypi_0    pypi
scotch                    6.0.9                hb2e6521_2    conda-forge
sentry-sdk                2.11.0                   pypi_0    pypi
setproctitle              1.3.3                    pypi_0    pypi
setuptools                69.5.1           py39h06a4308_0  
six                       1.16.0                   pypi_0    pypi
smmap                     5.0.1                    pypi_0    pypi
sortedcontainers          2.4.0                    pypi_0    pypi
sqlite                    3.45.2               h2c6b66d_0    conda-forge
stack-data                0.6.3                    pypi_0    pypi
sysroot_linux-64          2.17                h4a8ded7_16    conda-forge
tbb                       2021.11.0            h00ab1b0_1    conda-forge
tensorboard               2.17.0                   pypi_0    pypi
tensorboard-data-server   0.7.2                    pypi_0    pypi
tensorboardx              2.6.2.2                  pypi_0    pypi
termcolor                 2.4.0                    pypi_0    pypi
threadpoolctl             3.5.0                    pypi_0    pypi
tk                        8.6.13          noxft_h4845f30_101    conda-forge
tomli                     2.0.1                    pypi_0    pypi
torch                     1.12.1+cu113             pypi_0    pypi
torch-cluster             1.6.0+pt112cu113          pypi_0    pypi
torch-geometric           2.5.3                    pypi_0    pypi
torch-scatter             2.1.0+pt112cu113          pypi_0    pypi
torch-sparse              0.6.16+pt112cu113          pypi_0    pypi
torch-spline-conv         1.2.1+pt112cu113          pypi_0    pypi
torchaudio                0.12.1+cu113             pypi_0    pypi
torchvision               0.13.1+cu113             pypi_0    pypi
tqdm                      4.66.4                   pypi_0    pypi
traitlets                 5.14.3                   pypi_0    pypi
typing-extensions         4.12.2                   pypi_0    pypi
tzdata                    2024.1                   pypi_0    pypi
unixodbc                  2.3.12               h661eb56_0    conda-forge
urllib3                   2.2.2                    pypi_0    pypi
wandb                     0.17.5                   pypi_0    pypi
wcwidth                   0.2.13                   pypi_0    pypi
werkzeug                  3.0.3                    pypi_0    pypi
wheel                     0.43.0           py39h06a4308_0  
xz                        5.4.6                h5eee18b_1  
yarl                      1.9.4                    pypi_0    pypi
zipp                      3.19.2                   pypi_0    pypi
zlib                      1.2.13               h4ab18f5_6    conda-forge
zstd                      1.5.6                ha6fb4c9_0    conda-forge
```

## Running Apollo

```
python Apollo.py -p [problem_name]
```

## Citation

If you find this code useful, please consider citing the following papers.

```
@inproceedings{
liu2025apollomilp,
title={Apollo-{MILP}: An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear Programming},
author={Haoyang Liu and Jie Wang and Zijie Geng and Xijun Li and Yuxuan Zong and Fangzhou Zhu and Jianye HAO and Feng Wu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=mFY0tPDWK8}
}
```

