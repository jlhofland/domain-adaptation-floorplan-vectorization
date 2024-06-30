@REM cd "C:\Users\j.a.hofland\Documents\thesis-coding\pytorch-lightning-example\"

python3.10 train.py wandb.experiment_name=TEST_LV000 wandb.weights=model-egtpv2r8 mmd.lambda_constant=0.00
python3.10 train.py wandb.experiment_name=TEST_LV010 wandb.weights=model-t414g2t1 mmd.lambda_constant=0.10
python3.10 train.py wandb.experiment_name=TEST_LV020 wandb.weights=model-mh0lfraa mmd.lambda_constant=0.20
python3.10 train.py wandb.experiment_name=TEST_LV030 wandb.weights=model-fy20md1t mmd.lambda_constant=0.30
python3.10 train.py wandb.experiment_name=TEST_LV040 wandb.weights=model-cxwjavtf mmd.lambda_constant=0.40
python3.10 train.py wandb.experiment_name=TEST_LV050 wandb.weights=model-kgn8g355 mmd.lambda_constant=0.50

pause