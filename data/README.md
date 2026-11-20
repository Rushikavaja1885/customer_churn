# Dataset Download

This project uses the IBM Telco Customer Churn dataset published on Kaggle.

## Kaggle Page

Download the dataset from:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Expected File

After downloading and extracting the archive, place this CSV at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The training, evaluation, prediction, notebooks, and Streamlit app use that path by default.

## Kaggle CLI Option

If you use the Kaggle CLI, configure your Kaggle API token first, then run:

```bash
kaggle datasets download -d blastchar/telco-customer-churn -p data --unzip
```

The `data/` directory is ignored by Git except for this README, so raw and processed data files stay local.
