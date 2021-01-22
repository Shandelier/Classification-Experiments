# Obrazowanie Biomedyczne

A study of Extraction Methods and SVM hyper-parameters impact on classification problem for 4 class COVID-19 and Pneumonia dataset.



## How to use

To preprocess original dataset (on google colab)
**kaggle.json authorization file is required to be in repo directory**

```bash
!git clone https://github.com/Shandelier/PWr-OB-Metrics
%cd PWr-OB-Metrics

!mkdir /content/PWr-OB-Metrics/curated-chest-xray-image-dataset-for-covid19 -p
!pip install kaggle
import json
import zipfile
import os
auth = {}
with open('/content/PWr-OB-Metrics/kaggle.json', 'r') as file:
  auth = json.load(file)
os.environ['KAGGLE_USERNAME'] = auth["username"]
os.environ['KAGGLE_KEY'] = auth["key"]
# !kaggle config set -n path -v '/content/drive/MyDrive/Colab Notebooks/PWr9'
!kaggle datasets download -d unaissait/curated-chest-xray-image-dataset-for-covid19
os.chdir('/content/PWr-OB-Metrics/curated-chest-xray-image-dataset-for-covid19')
for file in os.listdir():
    zip_ref = zipfile.ZipFile(file, 'r')
    zip_ref.extractall()
    zip_ref.close()
! mv /content/PWr-OB-Metrics/curated-chest-xray-image-dataset-for-covid19/curated-chest-xray-image-dataset-for-covid19.zip /content/PWr-OB-Metrics/curated-chest-xray-image-dataset-for-covid19
%cd ..

# Unzip dataset
!unzip curated-chest-xray-image-dataset-for-covid19.zip -d /content/PWr-OB-Metrics/curated-chest-xray-image-dataset-for-covid19

!python ./preprocess.py --dataset_dir "/content/PWr-OB-Metrics/curated-chest-xray-image-dataset-for-covid19" --results_dir "/content/PWr-OB-Metrics/results" --output_dir "/content/PWr-OB-Metrics/output" --output_dataset_dir "/content/PWr-OB-Metrics/datasets"
```



To analyze prepared dataset

```bash
git clone https://github.com/Shandelier/PWr-OB-Metrics
cd PWr-OB-Metrics
python ./analyze_extraction.py
python ./post_extraction.py
python ./extract.py
python ./analyze_SVM.py
pythom ./post_SVM.py
```

The results are in `"results*"` folders.

## Extraction Methods impact

We have compared PCA and Chi2 extraction methods with basic dataset from [COVID-19 Kaggle ds](https://www.kaggle.com/unaissait/curated-chest-xray-image-dataset-for-covid19). The 48 features were extracted using methods from `preprocess.py` file. Then in the loop we compared Balanced Accuracy score for diffrent number of preserved components.

## SVM parameters

From extraction experiment we choose the best settings (PCA with `n_components=10`), and prepare dataset for this experiment. In here we compared 3 cathegories of parameters:

- Kernels – Linear, Sigmoid, RBF;
- Gammas – 1 – 0.0001
- C parameteres – 1000 – 0.01

## Wilcoxon Test

The results were tested with Wilcoxon test to reveal statisticly different scores. Under every score in table there is a list of scores that were worst compared to this one.



