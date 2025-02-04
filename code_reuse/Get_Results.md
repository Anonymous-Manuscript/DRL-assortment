1. The code to plot the corresponding Figures in the following table is in *plot.ipynb*;
2. The Figures are plotted based on previous running results logged in "Log File";
3. To replicate the log results, you can run the "Replication code", which in turn will generate a log file that is exactly the same as the previous log results in the corresponding folder;

| Section in paper | Figure in paper    | Log File                                                     | Replication Code                                             |
| :--------------- | ------------------ | :----------------------------------------------------------- | :----------------------------------------------------------- |
| EC.10.2          | Figure EC.21 case1 | GT/rlN=10M=4k=0.03L=100T=500/Benchmark_INV12prod<br />rlN=10M=4k=0.03L=100T=500INV=12RNNmcprod/DRLTrainLog2024-12-06-21-03-36 | main.py --reuse_type='prod' --only_test_Benchmark=True<br />main.py --reuse_type='prod' |
| EC.10.2          | Figure EC.21 case2 | GT/rlN=10M=4k=0.03L=100T=500/Benchmark_INV12cus<br />rlN=10M=4k=0.03L=100T=500INV=12RNNmccus/DRLTrainLog2024-12-06-21-03-46 | main.py --reuse_type='cus' --only_test_Benchmark=True<br />main.py --reuse_type='cus' |
| EC.10.2          | Figure EC.21 case3 | GT/rlN=10M=4k=0.03L=100T=500/Benchmark_INV12rand<br />rlN=10M=4k=0.03L=100T=500INV=12RNNmcrand/DRLTrainLog2024-12-12-23-28-18 | main.py --reuse_type='rand' --only_test_Benchmark=True<br />main.py --reuse_type='rand' |
|                  |                    |                                                              |                                                              |



