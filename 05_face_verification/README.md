# datasets path
/datasets/objstrgzip/05_face_verification_Accessories
<!-- \_data
    \_ train
        \_ ??????_S???_L??_E??_C??_cropped.jpg (images)
        \_ train_meta.csv
        \_ train_label.csv
    \_ validate
        \_ ??????_S???_L??_E??_C??_cropped.jpg (images)
        \_ validate_label.csv
    \_ test
        \_ ??????_S???_L??_E??_C??_cropped.jpg (images)
        \_ test_label.csv (dummy labels) -->

# image file mean
image_name = "Person"_"Accessory"_"Illumination"_"Expression"_"Camera_Angle"_cropped.jpg

# unzip
unzip 05_face_verification_Accessories.zip -d /tf/notebooks/05_face_verification_accessories_여동빈/datasets

# train
python main.py
python main.py --lr=0.002 --cuda=True --num_epochs=100 --print_iter=10 --model_name="model.pth" --prediction_file="prediction.txt" --batch=16 --mode="train"

# test (for submission)
python main.py --batch=16 --model_name="best_90.pth" --prediction_file="prediction_test.txt" --mode="test" 

# caution
1. Read spec
2. unzip

# 학습&용어설명
알고리즘이 iterative 하다는 것: gradient descent와 같이 결과를 내기 위해서 여러 번의 최적화 과정을 거쳐야 되는 알고리즘

epoch : 인공 신경망에서 전체 데이터 셋에 대해 forward pass/backward pass 과정을 거친 것을 말함. 즉, 전체 데이터 셋에 대해 한 번 학습을 완료한 상태

batch Size : 한 번의 batch마다 주는 데이터 샘플의 size. epoch를 나누어서 실행하는 횟수라고 생각하면 된다. 메모리의 한계와 속도 저하 때문에 대부분의 경우에는 한 번의 epoch에서 모든 데이터를 한꺼번에 집어넣을 수는 없습니다. 그래서 데이터를 나누어서 주게 되는데 이때 몇 번 나누어서 주는가를 iteration, 각 iteration마다 주는 데이터 사이즈를 batch size라고 합니다.

iteration(iter) : epoch를 나누어서 실행하는 횟수라고 생각하면 됨.

lr(Learning rate) : 학습률 ( Learning Rate )가 낮을경우  기존에 있던 시작점으로부터  기울기 * Learning Rate만큼 이동하는데 Learning Rate가 작다면 그만큼 천천히 내려가게 되어 학습하는 속도가 매우 오래 걸릴것입니다.
