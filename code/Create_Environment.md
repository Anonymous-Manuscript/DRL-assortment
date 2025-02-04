1. run *simulate.py* with specific settings will create ground-truth choice model, generate synthetic data, and fit various choice models on the data;
2. the data and fitted choice models is saved in corresponding log folder, and is used in the consequent training and testing process.

| Section in paper | Figure in paper | Environment Log Folder                                       | Environment Creation Code                                    |
| :--------------- | --------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 5.1.3            | Figure 5        | GT/rlN=10M=4k=0.03L=100T=500                                 | simulate.py                                                  |
| 5.1.4            | Figure 6        | GT/rlN=10M=4k=0.03L=100T=500                                 | simulate.py                                                  |
| EC.3             | Figure EC.1     | GT/rlN=10M=4k=0.03L=100T=500                                 | simulate.py                                                  |
| EC.6.1           | Figure EC.5     | GT/rlN=10M=4k=0.03L=100T=500                                 | simulate.py                                                  |
| EC.6.2           | Figure EC.6     | GT/rlN=20M=4k=0.03L=100T=500                                 | simulate.py --num_products=20                                |
| EC.6.2           | Figure EC.7     | GT/rlN=10M=20k=0.03L=100T=500                                | simulate.py --num_cus_types=20                               |
| EC.6.2           | Figure EC.8     | GT/rlN=10M=4k=0.03L=500T=100                                 | simulate.py --L=500 --T=100                                  |
| EC.6.3           | Figure EC.9     | GT/rlN=10M=4k=0L=100T=500                                    | simulate.py --k=0                                            |
| EC.6.4           | Figure EC.10    | GT/lcmnlN=10M=4k=0.03L=100T=500                              | simulate.py --GT='lcmnl'                                     |
| EC.6.4           | Figure EC.11    | GT/lcmnlN=10M=4k=0.03L=100T=500                              | simulate.py --GT='lcmnl'                                     |
| EC.7             | Figure EC.13    | GT/rlN=10M=4k=0.03L=100T=500<br />GT/rlN=10M=20k=0.03L=100T=500 | simulate.py<br />simulate.py --num_cus_types=20              |
| EC.7             | Figure EC.14    | GT/rlN=10M=4k=0.03L=100T=500                                 | simulate.py                                                  |
| EC.7             | Figure EC.16    | GT/rlN=10M=4k=0.03L=100T=500<br />GT/rlN=20M=4k=0.03L=100T=500 | simulate.py<br />simulate.py --num_products=20               |
| EC.9.1           | Figure EC.18    | GT/rlN=10M=4k=0.03L=100T=500                                 | simulate.py                                                  |
| EC.9.2           | Figure EC.19    | GT/rlN=10M=4k=0.03L=100T=50<br />GT/rlN=10M=4k=0.03L=100T=100<br />GT/rlN=10M=4k=0.03L=100T=500<br />GT/rlN=10M=4k=0.03L=100T=1000 | simulate.py --T=50<br />simulate.py --T=100<br />simulate.py --T=500<br />simulate.py --T=1000 |
|                  |                 |                                                              |                                                              |



