# Replication package for "Performance Prediction From Source Code Is Task and Domain Specific"
The paper will appear in the proceedings of ICPC'23, RENE track.

All the results are available in `results/`.

All data are in the 'data/'. Those are compressed with 7z. Please unzip before usage.

The replication instructions are the following:

install `requirements.txt` for Python 3.8

## Task a: Prediction of the execution time of a program directly:

- `astnn_train_reg.py`
* To reproduce Table 1 and Figure 6, run:
    * `python astnn_train_reg.py --n_epoch=20 --n_min_samples=100 --split_method=p`
    * `python astnn_train_reg.py --n_epoch=20 --n_min_samples=100 --split_method=r`
* `CodeForces\astnn_test_reg.py` to make plots

## Task b: Prediction of the task solved by a program:

- `astnn_train_class.py`
* To reproduce Figure 7 and 8, run:
    * `python astnn_train_class.py --n_epoch=10 --n_min_samples=100`
    * `astnn_class4reg.py`
    * `astnn_similarity_vs_reg.py`

## Task c: For a pair of programs, prediction of which performs better:

* To reproduce Table 2 and Figure 9, run: 
    * `python astnn_train.py --n_epoch=5 --N=50000 --pairs_name=cf_cpp_pairs_1.5_1.1_False --n_min_samples=100 --split_method=p`
    * `python astnn_train.py --n_epoch=5 --N=50000 --pairs_name=cf_cpp_pairs_1.5_1.1_False --n_min_samples=100 --split_method=rp`

## Domain Adaption

* To reproduce Figure 10, run: 
    * `python astnn_train.py --n_epoch=5 --N=50000 --pairs_name=cf_cpp_pairs_1.5_1.1_False --n_min_samples=100 --split_method=rp --ubi`
* For the plots, check: `astnn_transfer.py` 

© [2023] Ubisoft Entertainment. All Rights Reserved
