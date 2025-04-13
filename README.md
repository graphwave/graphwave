# GraphWave: A Dynamic Context-Adaptive Multimodal Feature Fusion Framework for Threat Detection
![image](https://github.com/graphwave/graphwave/blob/main/fig/framework.png)

## Abstract
This paper introduces GraphWave, a dynamic context-adaptive framework that enhances the recognition rate of malicious behaviors through multimodal feature fusion. Comprehensive evaluations on six real-world datasets reveal that GraphWave attains a 99.13% F1 score in threat detection, outperforming the state-of-the-art baseline methods by an average of 7.62%.

## Environment
We conducted our experiments on a CentOS 7 (Linux 3.10.0) server equipped with an Intel Xeon Gold 6150 CPU. Our program ran on a single NVIDIA GeForce RTX 4090 GPU with a driver version of 535.183.06. The server had 256GB of DDR4 memory, with 32GB being utilized. Python 3.7.12 was chosen as the programming language, and the PyTorch framework (v1.13.1 for CUDA v12.2) was employed to boost the functionality and efficiency of our experimental processes.

## Usage
```pip install -r requirements.txt```

### Step 1: Generate dataset for trainning.
```
python dataset_gen.py --multiple --contextual --pcaps_path="./middleResults/pcap_data/botnet2014/" --session_pcaps_used="./middleResults/temporal_contextual_data/botnet2014/temporal_session_used.json" --wave_name="cgau8" --data_path="./middleResults/temporal_contextual_data/botnet2014/" --base_dir="./middleResults/pcap_data/botnet2014/"
```
### Step 2: Train and evaluate.
```
python train.py
```
