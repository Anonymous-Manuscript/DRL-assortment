1. The code to plot the corresponding Figures in the following table is in *plot.ipynb*;
2. The Figures are plotted based on previous running results logged in "Log File";
3. To replicate the log results, you can run the "Replication code", which in turn will generate a log file that is exactly the same as the previous log results in the corresponding folder;

| Section in paper | Figure in paper | Log File                                                     | Replication Code                                             |
| :--------------- | --------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 5.2.1            | Figure 7        | GT/30/Benchmark_INV5<br />log/N=30INV=5RNNGT/DRLTrainLog2024-12-23-20-00-22<br />log/N=30INV=5RNNGT/DRLTrainLog2025-01-02-22-34-03 | main.py --num_products=30 --ini_inv=5 --only_test_Benchmark=True<br />main.py --num_products=30 --ini_inv=5<br />main.py --num_products=30 --ini_inv=5 --feature=False |
| EC.6.5           | Figure EC.12    | GT/100/Benchmark_INV2<br />log/N=100INV=2RNNGT/DRLTrainLog2024-12-14-10-11-28<br />log/N=100INV=2RNNGT/DRLTrainLog2025-01-02-22-34-24 | main.py --num_products=100 --ini_inv=2 --only_test_Benchmark=True<br />main.py --num_products=100 --ini_inv=2<br />main.py --num_products=100 --ini_inv=2 --feature=False |
|                  |                 |                                                              |                                                              |
