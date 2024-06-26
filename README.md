# Public repository for FARC
**Paper Title:** Offline Reinforcement Learning for Bandwidth Estimation in RTC Using a Fast Actor and Not-So-Furious Critic 
([🔗Link](https://dl.acm.org/doi/10.1145/3625468.3652184))

Submission for the [2nd Bandwidth Estimation Challenge at ACM MMSys 2024](https://www.microsoft.com/en-us/research/academic-program/bandwidth-estimation-challenge/)

## 🥈 FARC Ranked 2nd in the final challenge

## Installation
Install the required Python packages

```bash
pip3 install -r requirements.txt
```

Download both `testbed` ([🔗Link](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/download-testbed-dataset.sh)) and `emulated` ([🔗Link](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/download-emulated-dataset.sh)) datasets from and put them into `Data`folder

Split them into `train` and `test`folders. We used `98%` of the data for training and `2%` for testing.

Final structure should look like this:
```
|-- Data
|   |-- emulated_dataset
|   |   |-- train
|   |   |-- test
|   |-- testbed_dataset
|   |   |-- train
|   |   |-- test
```

## Training
To train the model, first run the `train_critic.py` script to train the critic model. 

Once the training for the `critic` model is done, you can run the `train_actor.py` script to train the `actor` model.

## Evaluation
To evaluate the model, run the `evaluate.py` script. 

This will evaluate the model on the test subset of emulated dataset and save the results in the `figs` folder.

If you want to reproduce the figures in the paper, extract the call traces in the `test-traces.zip` and change the line 18 (`data_dir`) in `eval.py` to point to this folder. 

This will evaluate the model on these call traces and save the figures in the `figs` folder.

## ONNX Visualization
You can find the visualization of the ONNX model using the [Netron](https://netron.app/) tool below:

![Network Visualization](./repo/netron.png)

## Citation
If you use this code in your research, please cite our paper:

```
@inproceedings{FARC,
    author = {Çetinkaya, Ekrem and Pehlivanoglu, Ahmet and Ayten, Ihsan U. and Yumakogullari, Basar and Ozgun, Mehmet E. and Erinc, Yigit K. and Deniz, Enes and Begen, Ali C.},
    booktitle = {Proceedings of the 15th ACM Multimedia Systems Conference},
    doi = {10.1145/3625468.3652184},
    publisher = {Association for Computing Machinery},
    series = {MMSys'24},
    title = {{Offline Reinforcement Learning for Bandwidth Estimation in RTC Using a Fast Actor and Not-So-Furious Critic}},
    url = {https://doi.org/10.1145/3625468.3652184},
    year = {2024}
}
```
