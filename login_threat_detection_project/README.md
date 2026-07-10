# Login Threat Detection ML Project

This project trains a defensive machine learning model to detect suspicious login activity using the
public **Login Data Set for Risk-Based Authentication (RBA)**.

The dataset is privacy-preserving/synthesized from real-world login behavior. It is not raw company data,
so do not claim it is raw production data.

## What the model predicts

Main target:

- `Is Attack IP`

This means the model tries to predict whether a login attempt came from an IP address associated with
known attacker data.

We originally tested:

- `Is Account Takeover`

However, this label was extremely rare in the first 1,000,000 rows, so the model did not have enough
positive examples to learn from. Because of that, the main version of this project uses `Is Attack IP`,
which gives the model enough suspicious-login examples to train and evaluate properly.

## Project structure

```text
login_threat_detection_project/
  data/
    rba-dataset.zip                         # downloaded dataset
    rba-dataset.csv                         # extracted dataset
  models/
    login_threat_model_balanced.joblib      # saved trained balanced model
  reports/
    balanced_evaluation_report.txt          # balanced model metrics
  src/
    download_data.py                        # downloads and extracts the RBA dataset
    rba_utils.py                            # helper functions for cleaning and preprocessing
    train_model_balanced.py                 # trains the balanced ML model
    predict_one_balanced.py                 # predicts one example login
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

If you already downloaded and extracted the dataset, you do not need to download it again.

## Step 3: Train the model

Recommended command:

```bash
python src/train_model_balanced.py --target "Is Attack IP" --max-rows 1000000 --negatives-per-positive 5
```

This scans the first 1,000,000 rows, collects attack-IP examples, samples normal login examples,
trains a Random Forest model, chooses a better decision threshold, and saves the trained model.

The model will be saved here:

```text
models/login_threat_model_balanced.joblib
```

The evaluation report will be saved here:

```text
reports/balanced_evaluation_report.txt
```

## Step 4: Predict one login

```bash
python src/predict_one_balanced.py
```

This loads the saved model and predicts whether a sample login is normal or suspicious.

## Current model results

Using the command:

```bash
python src/train_model_balanced.py --target "Is Attack IP" --max-rows 1000000 --negatives-per-positive 5
```

The model produced the following test results:

```text
Accuracy: 0.8558
F1-score: 0.6687
PR-AUC / Average Precision: 0.6875
ROC-AUC: 0.9221
```

For the suspicious class, class `1`, the model achieved:

```text
Precision: 0.5418
Recall:    0.8733
F1-score:  0.6687
```

This means the model detected about 87.33% of attack-IP login attempts in the balanced test set.
However, its precision was 54.18%, meaning some normal login attempts were also falsely flagged as
suspicious.

## Confusion matrix

```text
[[84833 14701]
 [ 2522 17385]]
```

This means:

```text
True normal logins correctly marked normal:      84,833
Normal logins incorrectly flagged suspicious:   14,701
Attack-IP logins missed by the model:            2,522
Attack-IP logins correctly detected:            17,385
```



## Important limitations

- The dataset is synthesized from real-world login behavior, not raw production login data.
- The test set is balanced through sampling, so the results should be described as results on a balanced test set.
- The model may produce false positives, meaning some normal logins may be incorrectly flagged as suspicious.
