# Replication package for "Performance Prediction From Source Code Is Task and Domain Specific"
The paper will appear in the proceedings of ICPC'23, RENE track.

All the results are available in the `results/` folder.

You can find the data in the `data/` folder. Those are compressed with 7z. Please unzip before usage.

The replication instructions are the following:

Install `requirements.txt` for Python 3.8.

## Scenario a: Prediction of the execution time of a program directly:

- `astnn_train_reg.py`
* To reproduce Table 1 and Figure 6, run:
    * `python astnn_train_reg.py --n_epoch=20 --n_min_samples=100 --split_method=p`
    * `python astnn_train_reg.py --n_epoch=20 --n_min_samples=100 --split_method=r`
* `astnn_test_reg.py` to make plots

## Scenario b: Prediction of the task solved by a program:

- `astnn_train_class.py`
* To reproduce Figure 7 and 8, run:
    * `python astnn_train_class.py --n_epoch=10 --n_min_samples=100`
    * `astnn_class4reg.py`
    * `astnn_similarity_vs_reg.py`

## Scenario c: For a pair of programs, prediction of which performs better:

- `astnn_train.py`
* To reproduce Table 2 and Figure 9, run: 
    * `python astnn_train.py --n_epoch=5 --N=50000 --pairs_name=cf_cpp_pairs_1.5_1.1_False --n_min_samples=100 --split_method=p`
    * `python astnn_train.py --n_epoch=5 --N=50000 --pairs_name=cf_cpp_pairs_1.5_1.1_False --n_min_samples=100 --split_method=rp`


© [2023] Ubisoft Entertainment. All Rights Reserved.
