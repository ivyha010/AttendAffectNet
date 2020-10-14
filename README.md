# AttendAffectNet: Self-Attention based Networks for Predicting Affective Responses from Movies. 

These code snippets are for predicting affective responses of viewers from movies, in which both video and audio are used as the inputs of models. 

## Prerequisites

Code snippets were implemented in Ubuntu 18, Python 3.6 and the experiments were run on a NVIDIA GTX 1070.

## Data

We use the folowing datatsets: 
 - The [extended Cognimuse dataset](http://cognimuse.cs.ntua.gr/database), in which each movie clip is split into 5-second segments 
 - The dataset for the Global Annotation subtask in the [MediaEval2016 Emotional Impact of Movies Task](https://liris-accede.ec-lyon.fr/database.php)

## Feature Extration 

* Features extracted from video:  We extract a 2048-feature vector using pre-trained ResNet-50, a 1024-feature vector using pre-trained RGB-stream I3D model and a 1024-feature vector using the contracting part of the pre-trained FlowNetS 

 - Appearance features:  Using the [pre-trained ResNet-50](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html) without its last fully connected layer, we extract a 2048-feature vector from each RGB frame, then we compute the element-wise averaging of all the extracted 2048-feature vectors to get a 2048-feature vector for each clip.

   We  also  use  the  [RGB-stream  I3D  model](https://github.com/piergiaj/pytorch-i3d)  based  on Inception-v1  with  pre-trained  weights  on  the  Kinetics human  action  video  dataset  to  extract  spatio-temporal features  from  video  frames  carrying  information  about  the appearance  and  temporal  relation.   The stack  of frames is  passed  through  the  pre-trained Inception-v1  based  RGB-stream  I3D  network  after  removing all layers following its “mixed-5c” layer before applying  the  average  pooling  to  get  a 1024-feature  vector  from  each  movie excerpt. 

 - Motion features: We use the contracting part of the [FlowNet Simple (FlowNetS)](https://github.com/ClementPinard/FlowNetPytorch) as a feature extractor  to  obtain  motion  features,  in  which  the  FlowNetS  was  pre-trained  on  the  Flying  Chairs  dataset.  A 1024-feature vector is obtained by passing each pair of consecutive frames through the  contracting part of the FlowNetS. We compute the element-wise average of the extracted motion features over all pairs  of  frames  to  get  a 1024-feature  vector  for  each  movie excerpt. 

* Features extracted from audio:

  - Audio features are extracted using the OpenSMILE toolkit with an audio frame size of 320 ms and a hop size of 40 ms. We use a configuration file named “emobase2010”, which is based on INTERSPEECH 2010 paralinguistics challenge. 
  
  - We  use  the  [VGGish  network](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) pre-trained for sound classification on the AudioSet dataset as an audio feature extractor. From each 0.96-second audio segment, we obtain 128 audio features. We compute the element-wise averaging of the extracted features over all segments to finally obtain a 128-feature vector for each movie excerpt. 


## Training and testing

  - For the extended COGNIMUSE dataset, we do leave-one-out cross-validation. This dataset includes 12 movie clips, therefore, we have 12 pre-trained models. The uploaded pre-trained models saved in *.pth* files are the best ones.

  - For the dataset for the Global Annotation subtask in the MediaEval2016 Emotional Impact of Movies Task, we train the proposed models on its training set and test them on its test set. 
 
## Paper & Citation

If you use this code, please cite the following paper: 

@article{,
  author={Ha Thi Phuong Thao, Balamurali B.T., Dorien Herremans, Gemma Roig},
  title={AttendAffectNet: Self-Attention based Networks forPredicting Affective Responses from Movies},
  journal={},
  year={2020}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


