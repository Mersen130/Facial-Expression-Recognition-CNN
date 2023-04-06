# Run Guide

FER2013 database should be downloaded and saved in the current directory. https://www.kaggle.com/datasets/msambare/fer2013

## To run the notebook in docker:

`docker run --gpus all -it tensorflow/tensorflow:latest-gpu -p 8888:8888 -v ${PWD}:/app bash`

In docker container, install jupyterlab and run it:

`jupyter-lab --allow-root --ip 0.0.0.0 --port 8888`

You also need to install all the dependencies imported on the top of the notebook.


## To run in google colab:
it will be slower for free user, but you don't need a gpu on host machine, and saves a lot of time configuring environment.

Upload the notebook to colab, and also upload FER2013 dataset to google drive, (this step will take very long time as the dataset contains thousands of files).

Connect your colab session with google drive, you need to load the dataset from google drive to colab by yourself. Many documentations available on internet.

# Model Explanation

1. Attention-based Emotion Region CNN (AER-CNN):
AER-CNN focuses on emotion-relevant facial regions (e.g., eyes, mouth) using an attention mechanism.

Implement an attention mechanism (e.g., self-attention or spatial attention) to weigh the importance of different facial regions.
Apply this attention mechanism to the feature maps at different layers of the CNN.
Fuse the attention-weighted feature maps before passing them through the fully connected layers and the softmax layer for classification.
The novelty in this architecture lies in the attention mechanism's ability to focus on emotion-relevant regions, which may help in learning more discriminative features for FER.

2. Multi-Scale Feature Fusion CNN (MSFF-CNN):
The MSFF-CNN model can fuse features from different scales in a hierarchical fashion. This architecture addresses the issue that facial expressions can appear at various scales due to different distances from the camera, varying face sizes, and diverse expressions.

Start with a typical CNN architecture consisting of convolutional, activation, and pooling layers.
After each pooling layer, add a parallel branch that extracts features at different scales.
Fuse the outputs of these branches using concatenation or summation.
Follow this with fully connected layers and a softmax layer for classification.
The novelty in this architecture lies in the fusion of multi-scale features, allowing the model to learn features from different scales in a single pass.

3. Novel FER Model
Composed of 2 main layers

Dynamic Receptive Field (DRF) - A novel layer that adjusts the receptive field size based on the input data.

Multi-Scale Attention Module (MSAM) - A novel module that enhances the feature maps using attention mechanisms at different scale

# Conclusion
Multi-Scale Feature Fusion CNN (MSFF-CNN) may noy be an ideal architecture to this problem as it is easily overfitted in a few epochs.
However, the problem may not caused by architecture, it could be caused by FER2013 dataset as this dataset is not large, furthur improvement can be done on fine tuning the hyperparameters and input augmentation.

As a comparison, Attention-based Emotion Region CNN (AER-CNN) is slightly more robust than the other one.

The third model is the best among the three models.

Third model has a peak performance of 0.59 accuracy on validation set.

# Furthur work
The third model can have a better result if training on a dataset that has better image quality, for example, a classicle image dimension for CNN would be 224 * 224, but FER2013 only has 48 * 48. The proposed three deep neural networks have too many parameters for this dataset.
