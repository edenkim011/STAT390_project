# STAT390_project

## How to reproduce the Baseline
1. Environment setup:
`pip install pandas numpy scikit-learn` if not done already.

2. Data placement:
Place the Kaggle `train.csv` in the `/data` folder.

3. Initialize evaluation:
Run `python prepare.py` to generate deterministic 80/20 split and lock the test set.

4. Run baseline:
Run `python train.py` to see the baseline RMSE and runtime. 


### Current baseline results
- Validation metric: RMSE
- Baseline score: 1.1706
- Measured runtime: 0.0502 seconds