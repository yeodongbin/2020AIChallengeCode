
 06_face_verification_accessories_limit
cd /tf/notebooks/06_face_verification_accessories_limit


 # train
python main.py --lr=0.002 --cuda=True --num_epochs=100 --print_iter=10 --model_name="model.pth" --prediction_file="prediction.txt" --batch=128 --mode="train"


# test (for submission)
python main.py --batch=8 --model_name="best_72.pth" --prediction_file="prediction_test.txt" --mode="test"

# unzip
unzip 05_face_verification_Accessories.zip -d /tf/notebooks/06_face_verification_accessories_limit/datasets
