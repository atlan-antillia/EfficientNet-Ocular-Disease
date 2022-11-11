<h2>EfficientNet-Ocular-Disease (Updated: 2022/09/17)</h2>
<a href="#1">1 EfficientNetV2 Ocular Disease Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Prepare Ocular Disease dataset</a><br>
<a href="#1.3">1.3 Install Python packages</a><br>
<a href="#2">2 Python classes for Ocular Disease Classification</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>

<h2>
<a id="1">1 EfficientNetV2 Ocular Disease Classification</a>
</h2>

 This is an experimental EfficientNetV2 Ocular Disease Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>.
<br>
This image dataset has been taken from the following website:<br>
<br>
<b>Ocular Disease Intelligent Recognition ODIR-5K</b><br>
https://academictorrents.com/details/cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72
<br>
<br>
<b>Ocular Disease Recognition</b><br>
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
<br>
<br>
About this Data<br>
See also: https://academictorrents.com/details/cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72<br>
<pre>
Ocular Disease Intelligent Recognition (ODIR) is a structured ophthalmic database of 5,000 patients with age, color fundus photographs from left and right eyes and doctors' diagnostic keywords from doctors.

This dataset is meant to represent ‘‘real-life’’ set of patient information collected by Shanggong Medical Technology Co., Ltd. from different hospitals/medical centers in China. In these institutions, fundus images are captured by various cameras in the market, such as Canon, Zeiss and Kowa, resulting into varied image resolutions.
Annotations were labeled by trained human readers with quality control management. They classify patient into eight labels including:

Normal (N),
Diabetes (D),
Glaucoma (G),
Cataract (C),
Age related Macular Degeneration (A),
Hypertension (H),
Pathological Myopia (M),
Other diseases/abnormalities (O)

License
License was not specified on source
</pre>
Splash Image
Image from <a href="https://pixabay.com/pt/users/matryx-15948447/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=5061291">Omni Matryx </a>
by Pixabay<br>
<br>We use python 3.8 and tensorflow 2.8.0 environment on Windows 11.<br>
<br>

<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/atlan-antillia/EfficientNet-Ocular-Disease.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
└─projects
    └─Ocular-Disease
        ├─eval
        ├─evaluation
        ├─inference
        ├─models
        ├─Resampled_Ocular_Disease_Images
        └─test
</pre>
<h3>
<a id="1.2">1.2 Prepare Ocular_Disease dataset</a>
</h3>

Please download the dataset <b>Ocular_Disease dataset dataset</b> from the following web site:<br>
<a href="https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k">Ocular Disease Recognition</a>
<br>
<br>

As a working master dataset, we have created <b>Ocular_Disease_master</b> ]
dataset from the original <b>preprocessed_images</b> above
 by using python script <a href="./projects/Ocular-Disease/create_master_dataset.py">create_master_dataset.py</a>.<br>
<pre>
Ocular_Disease_master
  ├─A
  ├─C
  ├─D
  ├─G
  ├─H
  ├─M
  ├─N
  └─O
</pre>
The number of images in this dataset is the following:<br>
<img src="./projects/Ocular-Disease/_Ocular_Disease_master_.png" width="740" height="auto"><br>
<br>
This is a typical imbalanced dataset, because the number of images of class N is extremely large compared with other classes.
<br>

Futhermore, we have created a balanced dataset <b>Resampled_Ocular_Disease_Images</b> 
 by using <a href="https://github.com/martian-antillia/ImageDatasetResampler">ImageDatasetResampler</a><br>
<br>
<pre>
Resampled_Ocular_Disease_Images
├─test
│  ├─A
│  ├─C
│  ├─D
│  ├─G
│  ├─H
│  ├─M
│  ├─N
│  └─O
└─train
    ├─A
    ├─C
    ├─D
    ├─G
    ├─H
    ├─M
    ├─N
    └─O
</pre>
The number of images in this Resampled_Ocular_Disease_Images is the following:<br>
<img src="./projects/Ocular-Disease/_Resampled_Ocular_Disease_Images_.png" width="740" height="auto"><br>
<br>
<br>
Resampled_Ocular_Disease_Images/train/A (Age related Macular Degeneration) :<br>
<img src="./asset/Ocular_Disease_train_A.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/C (Cataract):<br>
<img src="./asset/Ocular_Disease_train_C.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/D (Diabetes):<br>
<img src="./asset/Ocular_Disease_train_D.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/G (Glaucoma):<br>
<img src="./asset/Ocular_Disease_train_G.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/H (Hypertension):<br>
<img src="./asset/Ocular_Disease_train_H.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/M (Pathological Myopia):<br>
<img src="./asset/Ocular_Disease_train_M.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/N (Normal):<br>
<img src="./asset/Ocular_Disease_train_N.png" width="840" height="auto">
<br>
<br>
Resampled_Ocular_Disease_Images/train/O (Other diseases/abnormalities):<br>
<img src="./asset/Ocular_Disease_train_O.png" width="840" height="auto">
<br>
<br>


<h3>
<a id="#1.3">1.3 Install Python packages</a>
</h3>
Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for Ocular Disease Classification</a>
</h2>
We have defined the following python classes to implement our Ocular Disease Classification.<br>
<li>
<a href="./ClassificationReportWriter.py">ClassificationReportWriter</a>
</li>
<li>
<a href="./ConfusionMatrix.py">ConfusionMatrix</a>
</li>
<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>
<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>
<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>

<li>
<a href="./TestDataset.py">TestDataset</a>
</li>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-m</b> to train Ocular Disease Classification FineTuning Model.
Please download the pretrained checkpoint file from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz">efficientnetv2-m.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-m
└─projects
    └─Ocular-Disease
  ...
</pre>

<h2>
<a id="4">4 Train</a>

</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our Ocular Disease Classification efficientnetv2 model by using
<b>Resampled_Ocular_Disease_Images/train</b>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-m  ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-m/model ^
  --optimizer=rmsprop ^
  --image_size=384 ^
  --eval_image_size=480 ^
  --data_dir=./Resampled_Ocular_Disease_Images/train ^
  --data_augmentation=True ^
  --valid_data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.3 ^
  --num_epochs=50 ^
  --batch_size=4 ^
  --patience=10 ^
  --debug=True  
</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config

[training]
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 180
horizontal_flip    = True
vertical_flip      = True 
width_shift_range  = 0.2
height_shift_range = 0.2
shear_range        = 0.01
zoom_range         = [0.1, 2.0]
data_format        = "channels_last"

[validation]8
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 180
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.2
height_shift_range = 0.2
shear_range        = 0.01
zoom_range         = [0.1, 2.0]
data_format        = "channels_last"
</pre>

<h3>
<a id="4.2">4.2 Training result</a>
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/Ocular-Disease/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/Ocular-Disease/eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/Ocular_Disease_train_at_epoch_24_0916.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<img src="./projects/Ocular-Disease/eval/train_accuracies.png" width="640" height="auto"><br>

<br>
Train_losses:<br>
<img src="./projects/Ocular-Disease/eval/train_losses.png" width="640" height="auto"><br>

<br>
<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the Ocular-Disease in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.3 ^
  --image_path=./test/*.jpg ^
  --eval_image_size=480 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
A
C
D
G
H
M
N
O
</pre>
<br>
<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/Ocular-Disease/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/Ocular-Disease/Resampled_Ocular_Disease_Images/test">Resampled_Ocular_Disease_Imagess/test</a>.
<br>
<img src="./asset/Ocular_Disease_test.png" width="840" height="auto"><br>

<h3>
<a id="5.3">5.3 Inference result</a>
</h3>
This inference command will generate <a href="./projects/Ocular-Disease/inference/inference.csv">inference result file</a>.
<br>
<br>
Inference console output:<br>
<img src="./asset/Ocular_Disease_infer_at_epoch_24_0916.png" width="740" height="auto"><br>
<br>

Inference result (inference.csv):<br>
<img src="./asset/Ocular_Disease_inference_at_epoch_24_0916.png" width="740" height="auto"><br>
<br>
<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Ocular-Disease/Resampled_Ocular_Disease_Images/test">
Resampled_Ocular_Disease_Images/test</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --data_dir=./Resampled_Ocular_Disease_Images/test ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.3 ^
  --eval_image_size=480 ^
  --mixed_precision=True ^
  --debug=False 
</pre>
<br>

<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/Ocular-Disease/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/Ocular-Disease/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/Ocular_Disease_evaluate_at_epoch_24_0916.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/Ocular_Disease_classificaiton_report_at_epoch_24_0916.png" width="740" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./projects/Ocular-Disease/evaluation/confusion_matrix.png" width="740" height="auto"><br>


<br>
<h3>
References
</h3>
<b>1. Ocular Disease Intelligent Recognition ODIR-5K</b><br>
<pre>
https://academictorrents.com/details/cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72
</pre>

<b>2. Ocular Disease Recognition</b><br>
<pre>
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
</pre>
