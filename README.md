# Author statement

This repository contains supporting files of data and code for the paper manuscript titled "Prot-ΔΔG: Prediction of protein-protein binding affinity changes upon mutations with pre-training strategies".The focus of this work is to predict changes in protein-protein affinity using large pre-trained protein language models and protein sequence information.

# Resources:

+ README.md: this file.
+  data/binding_affinity: This file stores protein sequence files and ΔΔG-related data for the three protein mutation datasets used for the experiments (SKP1102s,SKP1400m-single,SKP1400m). Among them, SKP1400m-single and SKP1400m share a seq.txt file.
+  model/src:This file holds three model files that correspond to the experimental codes under the three datasets in this project. The corresponding results can be output by running them directly.
+  data/embedding:Used to store the result files for the ProtBert pre-trained protein language model characterizing the sequence information in the three datasets. These files will be used in model training.The pre_embedding.pkl file is the sequence pre-training characterization result file for SKP1102s dataset; the test_pre_1402multiply_embedding.pkl file is the sequence pre-training characterization result file for SKP1400m dataset; the test_pre_1402single_embedding.pkl file is the sequence pre-training characterization result file for SKP1400m-single dataset(Can be found at https://figshare.com/s/47bf20d8156e33e49ca3)

###  Source codes:

+ Transfer_bert_dbrnn.py:Runs were performed to obtain experimental results for the Transfer-ΔΔG model on the single point mutation dataset SKP1102s.
+ Transfer-bert_1400single_dbrnn.py:Runs were performed to obtain experimental results for the Transfer-ΔΔG model on the single point mutation dataset SKP1400m-single.
+ Transfer_bert_1400mutiply_dbrnn.py:Runs were performed to obtain experimental results for the Transfer-ΔΔG model on the mixed mutation dataset SKP1400m.

# Step-by-step running:

## 1.Install Python libraries needed

The deep learning frameworks used in this paper are tensorflow=2.6.2 and keras=2.2.4. This is accomplished by executing the following command.

```
conda create -n transfer python=3
conda activate transfer
conda install -c huggingface transformers
conda install tensorflow-gpu=2.6.2
conda install keras=2.2.4
conda install numpy=1.19.5
conda install tqdm=4.64.1
conda install scipy=1.5.4
pip install scikit-learn=0.24.2
```

## 2. Generating embedding using pre-trained macromodels

The documentation and usage of the ProtBert pre-training model can be found at https://github.com/agemagician/ProtTrans.

## 3. Select dataset and train

To train a model using training data. The model is chosen if it gains the best MSE for testing data.There are training files corresponding to the three datasets under the model/src file, which can be used directly. 

run when trained and tested on the SKP1102s dataset:

```
conda activate transfer
python Transfer_bert_dbrnn.py
```

run when trained and tested on the SKP1400m-single dataset:

```
conda activate transfer
python Transfer_bert_1400single_dbrnn.py
```

run when trained and tested on the SKP1400m dataset:

```
conda activate transfer
python Transfer_bert_1400mutiply_dbrnn.py
```

