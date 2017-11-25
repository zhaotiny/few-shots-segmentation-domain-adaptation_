run the program: python trainRegressNet.py --pair=model_pair.txt
model_pair.txt specify the path to the pair of model pairs

CUDA_VISIBLE_DEVICES=0 python trainRegressNet2.py --pair=model_pair_6.txt --regmodel=./models/15000.npy --model=../Models/segnet_inference.prototxt --weights=../Models/Inference/test_weights.caffemodel

28_16_50_1_2000_reg1.npy
28: total number of categories
16: current category
50: number of sample images
1: 1th sample
2000: iterations
reg1: small regularization

28 -> 50 -> 1 -> 2000 -> reg1 -> 16




CUDA_VISIBLE_DEVICES=0 python ./SegNet/Scripts/test_regression.py --model ./SegNet/Models/segnet_inference_6_city_val.prototxt --weights ./SegNet/Models/city_test_models_cityall/segnet1_2_iter_5000.caffemodel --iter 500 --list ./SegNet/cityscape/cityValIdx.txt --save ./SegNet/cityscape/reg_test/ --conv ./SegNet/Scripts/tmp.npy
