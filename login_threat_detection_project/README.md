# Login Threat Detection ML Project

This project trains a defensive machine learning model to detect suspicious login activity using the
public **Login Data Set for Risk-Based Authentication (RBA)**.

The dataset is privacy-preserving/synthesized from real-world login behavior. It is not raw company data,
so do not claim it is raw production data.

## What the model predicts

Default target:

- `Is Account Takeover`

This means the model tries to predict whether a login attempt is associated with account takeover activity.

You can also change the target to:

- `Is Attack IP`

by passing a command-line argument.

## Project structure

```text
login_threat_detection_project/
  data/
    rba-dataset.zip              # downloaded dataset
    rba-dataset.csv              # extracted dataset
  models/
    login_threat_model.joblib    # saved trained model
  reports/
    evaluation_report.txt        # model metrics
  src/
    download_data.py
    rba_utils.py
    train_model.py
    predict_one.py
  requirements.txt
  README.md
```

## Step 1: Install packages

```bash
pip install -r requirements.txt
```

## Step 2: Download the dataset

The dataset is large: about 1.1 GB compressed and about 9 GB extracted.

```bash
python src/download_data.py
```

If the automatic download is slow, manually download `rba-dataset.zip` from Zenodo and put it in the
`data/` folder, then run:

```bash
python src/download_data.py --skip-download
```

## Step 3: Train the model

Start with a smaller sample first:

```bash
python src/train_model.py --sample-rows 200000 --target "Is Account Takeover"
```

If the model says there are too few positive examples, increase the sample size:

```bash
python src/train_model.py --sample-rows 1000000 --target "Is Account Takeover"
```

Alternative easier target:

```bash
python src/train_model.py --sample-rows 200000 --target "Is Attack IP"
```

## Step 4: Predict one login

```bash
python src/predict_one.py
```

## Notes for your report

Use wording like this:

> This project uses the public Login Data Set for Risk-Based Authentication, a privacy-preserving dataset
> synthesized from real-world login behavior. The model analyzes login context features such as timestamp,
> country, device type, browser, operating system, login success, round-trip time, and attack-IP status to
> classify whether a login attempt is suspicious.

## Important limitations

- The dataset is synthesized from real login behavior, not raw production data.
- The model is for educational research only.
- This should not be deployed in a real security system without professional validation.
- Login behavior data can be highly imbalanced, so recall, precision, F1, and PR-AUC matter more than plain accuracy.
