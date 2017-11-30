This project is about to apply the learning to learn method to solve the segmentation domain adaptation. Specifically,
we focus on semi-supervised domain adaptation (Few-shot learning). Our work is based on SegNet.


The overall procedure is as followed:
1. Obtain lots of pairs of classifiers from models learnt from small samples and those learnt from large samples in
    GTA5 dataset.
2. Train the regression network that maps the small sampled classifier to large sampled classifier.
3. Apply the regression network to the small sampled models learnt from Cityscape dataset to get the target models.

4. Learn a model to predict the overall layout distribution in GTA5 dataset.
5. Apply the "distribution prediction model" to cityscape dataset to obtain predicted distributions.

6. Train the final model using the constraints from Step 3 and Step 5.
7. Evaluate at Cityscape testing set.

Here are some details about how to run the program.

For Step 1, first train a lot of model pairs and then use "extract_classifier_from_caffemodel.py" to extract the
classifiers from the models.

For Step 2, it takes two stages to train the regression network. The first stage only minimise
the regression loss, while the second stage combining with the segmentation loss.
To train stage 1, run python train_regression_net_stage1.py --pair=model_pair.txt.
To train stage 2, run python train_regression_net_stage2.py --pair=model_pair_6.txt --regmodel=stage1.npy
                        --model=segnet_inference.prototxt --weights=test_weights.caffemodel

For step 3, train a lots of small sampled models in cityscape dataset. And then run
            "get_target_models_from_regression_net.py" to get the target models.
            python get_transformed_model.py --regmodel=stage2.npy --model=segnet_inference.prototxt
            --weights=segnet.caffemodel --folder=./target_weights/ --name=target.npy

For step 4, obtain the GT distribution first before running "train_layout_distribution_prediction.py"
            to train the distribution prediction model
For step 6, to make the data to fit into the gpu and increase the training speed, you might need to first extract
            features before the last classifier from caffemodel and store it using "extract_conv_features.py".
            After that, run python learning_to_learn_ds.py --f2gt=feature2GT.txt --conv=target.npy
            --name=final.npy --normal
For step 7, run python test_regression_simple.py --conv=final.npy --list=city_val_f2gt.txt --save=results/







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

extract the transformed model:
CUDA_VISIBLE_DEVICES=2 python get_transformed_model.py --regmodel=./models/reg1_20000.npy --model=../Models/segnet_inference_train_6_city.prototxt --weights=../Models/city_test_models_cityall/segnet1_1_iter_5000.caffemodel --folder=../test_results/model_reg1_50000/ --name=20000.npy

test learning_to_learn
