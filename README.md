# MAS
Humans can learn in a continuous manner. Old rarely utilized knowledge can be overwritten by new incoming information while important, frequently used knowledge is prevented from being erased. In artificial learning systems,
lifelong learning so far has focused mainly on accumulating knowledge over tasks and overcoming catastrophic forgetting. In this paper, we argue that, given the limited model capacity and the unlimited new information to be learned, knowl-
edge has to be preserved or erased selectively. Inspired by neuroplasticity, we propose a novel approach for lifelong learning, coined Memory Aware Synapses(MAS). It computes the importance of the parameters of a neural network in an
unsupervised and online manner. Given a new sample which is fed to the network,MAS accumulates an importance measure for each parameter of the network,  based  on  how  sensitive  the  predicted  output  function  is  to  a  change  in
this parameter. When learning a new task, changes to important parameters can then be penalized, effectively preventing important knowledge related to previous tasks from being overwritten. Further, we show an interesting connection between
a local version of our method and Hebb’s rule, which is a model for the learning process  in  the  brain.  We  test  our  method  on  a  sequence  of  object  recognition tasks and on the challenging problem of learning an embedding for predicting
<subject, predicate, object> triplets. We show state-of-the-art performance and, for the first time, the ability to adapt the importance of the parameters based on unlabeled data towards what the network needs (not) to forget, which may vary
depending on test conditions.

![Global Model](https://raw.githubusercontent.com/rahafaljundi/MAS-Memory-Aware-Synapses/master/teaser_fig.png)

This directory contains a pytorch implementation of Memory Aware Synapses: Learning what not to forget method. A demo file that shows a learning scenario in mnist split set of tasks is included.

## Authors

Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach and Tinne Tuytelaars


For questions about the code, please contact me, Rahaf Aljundi (rahaf.aljundi@esat.kuleuven.be)
## Requirements
The code was built using pytorch version 0.3, python 3.5 and cuda 9.1
## Citation
Aljundi R., Babiloni F., Elhoseiny M., Rohrbach M., Tuytelaars T. (2018) Memory Aware Synapses: Learning What (not) to Forget. In:  Computer Vision – ECCV 2018. ECCV 2018. Lecture Notes in Computer Science, vol 11207. Springer, Cham

## License

This software package is freely available for research purposes.

## Our Work

# Introduction

Millions of images with new captions and tags and a huge amount of streaming data  such  as  videos  are  generated  and  uploaded  to  the  internet  daily. These new data may contain new topics, trends, and patterns. The classic supervised learning algorithms have been designed to reach a high performance on a specific task of image classification, object detection or speech recognition. These algorithms do not have the ability of continuous learning from different tasks and datasets during the time. Performing these algorithms to the new datasets may lead to forgetting the learned model on the previous tasks issue which is known as catastrophic forgetting. Lifelong learning has emerged to address the issue of catastrophic forgetting and enabling learning models to have an acceptable performance on both current and previous tasks. In this project we try to implement a model-based lifelong learning approach ”Memory Aware Synapses”. We also extend the idea of the paper, and with a change in the problem formulation, we reduce the average forgetting value over different tasks.

Memory-aware Synapses enjoys several properties such as constant memory size, Problem agnostic, supporting unlabeled data, adaptive, and also can be established on top of a pre-trained network on the previous tasks. Problem Agnostic means this method can be generalized to any dataset and it is not limited to specific tasks or datasets. By adaptive we mean the capability of a method to adapt the learned model continually to a new task from the same or different environment. Thus it means the samples from different tasks should not necessarily follow a unique distribution and can have different ground truth distributions. The Memory-aware Synapses (MAS) satisfies all the properties mentioned above; However, it might not work equally well on all tasks. If the tasks which should be learned come in a different order, the trained model can be different showing it puts more weights to the parameters of the most recent tasks. To cope with this issue, we introduce new parameters to equalize the importance of the tasks regardless of their training order. The experimental results show the same performance when we change the order of tasks and demonstrate a better average forgetting comparing to MAS. We call the independence of the model from the order of tasks the consistency property. 

# Existing Approaches

A common application of Lifelong learning is in the context of image processing where the goal is to learn several image classification tasks with only one convolutional neural networks. To deal with the problem of catastrophic forgetting, several methods have been proposed.[1] Defines two set of parameters: $\theta_s$ are the shared parameters among all different tasks. For each given task, $\theta_n$ represents the set of parameters defined specifically for this task and $\theta_0$ denotes all the parameters from the previous tasks. During the first 

## Model Based Approaches
Model base approaches focus on the parameters of the network instead of depending on the task data. Most similar to our work are. Like them, we estimate an importance weight for each model parameter and add a regularizer when training a new task that penalizes any changes to important parameters.
The difference lies in the way the importance weights are computed. In the Elastic Weight Consolidation work this is done based on an approximation of the diagonal of the Fisher information matrix. In the Synaptic Intelligence work importance weights are computed during training in an online manner. To this end, they record how much the loss would change due to a change in a specific parameter and accumulate this information over the training trajectory. However, also this method has some drawbacks:

1. Relying on the weight changes in a batch gradient descent might overestimate the importance of the weights, as noted by the authors.
2. When starting from a pretrained network, as in most practical computer vision applications, some weights might be used without big changes. As a result, their importance will be underestimated.
3. The computation of the importance is done during training and fixed later.


**Method** | **Type** | **Constant Memory** | **Problem Agnostic** | **On Pre-trained** | **Unlabeled Data** | **Adaptive** | **Consistency**
--- | --- | --- | --- | --- | --- | --- | ---
LwF | Data | :heavy_check_mark: | :x: | :heavy_check_mark: | :x: | :x: | :x:
EBLL | Data | :x: | :x: | :x: | :x: | :x: | :x:
EWC | Model | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x:
IMM | Model | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: | :x:
SI | Model | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: | :x:
MAS | Model | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
Alpha | Model | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:

