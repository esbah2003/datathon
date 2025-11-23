# Datathon - Burnout Risk Model

Small project to generate synthetic student burnout data, train a regression model, and quickly test predictions.


## set up environment
mac
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```bat
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Generate training data
This creates a CSV of fake burnout scores.

```bash
python src/mock_data/generate_burnout_data.py
```

The script writes `burnout_data_scores.csv` in your cwd. Copy it into the model folder so the training script can find it:

mac
```bash
cp burnout_data_scores.csv src/model/burnout_data_scores.csv
```

Windows:
```bat
copy burnout_data_scores.csv src\model\burnout_data_scores.csv
```

## Train the model
Trains randomForest regressor and saves the pipeline.

```bash
python src/model/train.py
```

Output: `src/model/model.joblib` plus validation metrics in the console.


## Runs a few sample inputs through the saved pipeline.

```bash
python src/mock_data/test.py
```