@REM cd "C:\Users\j.a.hofland\Documents\thesis-coding\pytorch-lightning-example\"

python3.10 train.py wandb.experiment_name=TEST_LC000 wandb.weights=model-egtpv2r8 mmd.lambda_constant=0.00
python3.10 train.py wandb.experiment_name=TEST_LC001 wandb.weights=model-zazi4vl8 mmd.lambda_constant=0.01
python3.10 train.py wandb.experiment_name=TEST_LC005 wandb.weights=model-wc3td2vc mmd.lambda_constant=0.05
python3.10 train.py wandb.experiment_name=TEST_LC010 wandb.weights=model-w97jlr7u mmd.lambda_constant=0.10
python3.10 train.py wandb.experiment_name=TEST_LC020 wandb.weights=model-ppjlia6a mmd.lambda_constant=0.20
python3.10 train.py wandb.experiment_name=TEST_LC030 wandb.weights=model-mkd81otc mmd.lambda_constant=0.30
python3.10 train.py wandb.experiment_name=TEST_LC040 wandb.weights=model-6j5w4aqk mmd.lambda_constant=0.40
python3.10 train.py wandb.experiment_name=TEST_LC050 wandb.weights=model-nuj5sb6p mmd.lambda_constant=0.50

pause