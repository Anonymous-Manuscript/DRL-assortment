## This is the code for paper "Deep Reinforcement Learning for Online Assortment Customization: A Data-Driven Approach"

There are three types of experiments: simulation with ground-truth choice model, simulation with real data, simulation with ground-truth choice model and reusable products. These three types of experiments are recorded in the folder *code, code_realdata, code_reuse*, respectively.

**In each folder, how each Figure/Table in the paper came out is explained, and the instructions to check and replicate the results are given in the correponding Markdown file.**

For replicating the results in the *code* and *code_reuse* folder, you should follow two steps:

1. create an environment using *simulate.py*, and this step is detailed in the Markdown file *Create_Environment.md*;
2. test benchmark, train and test A2C agents using *main.py*, and this step is detailed in the Markdown file *Get_Results.md*.

For replicating the results in the *code_realdata* folder, you should follow three steps:

1. data pre-processing using *DRL_ass.ipynb*, and this is explained in *Process_data.md*;
2. fit choice models based on processed data using *read_realdata.py*, and this step is detailed in the Markdown file *read_realdata.md*;
3. test benchmark, train and test A2C agents using *main.py*, and this step is detailed in the Markdown file *Get_Results.md*.
