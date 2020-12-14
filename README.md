# Introduction
Learning algorithms especially deep learning methods achieved a reliable performance on supervised learning tasks such as image classification, object detection, and speech recognition.
The general objective of classic supervised learning is to minimize the following risk function based on a finite sample from the unknown, but fixed distribution P*.

![](eq1.jpg)

However, the assumption that the training data follow a fixed distribution during time has several limitations and many problems cannot be formulated in this framework. For instance, if the learning problem consists of several tasks with different distributions (e.g. different image datasets), updating the parameters of the model
upon the arrival of new task data, leads to poor performance on the previously learned tasks. 
The procedure of continuous learning for different tasks with a single model (e.g. a deep neural network) is known as lifelong learning. Lifelong learning enables us to learn 
continually from tones of images, videos, and other media contents generated daily on the internet, without forgetting the learned model based on the previous data. 
The following figure illustrates the lifelong learning high-level idea. Figure (a) shows for a cetrian task (2nd task) some parameters of the network would be involved. In classic machine learning approaches when the next task comes in consideration using the same of parameters as the previous task, parameters get adjusted for the new task. This leads drastic forgetting and reduction of validation accuracy if we test previous tasks on the network. As shown in figure (c) a lifelong based machine learning approach trys to change the value of the shared parameters (the intersection in the figure) such that it is still beneficial for previous tasks depending on their importance for different tasks.  

![](Concept.png)


A question might arise is "whether the classic formulation of learning can lead to reliable performance on different tasks\datasets?". The experimental results show that by updating the parameters of a deep neural network upon the arrival of the new tasks, the performance of the network on previous tasks decreases profoundly. This phenomenon is known as catastrophic forgetting. 
As such, deep lifelong learning has emerged to address the issue of catastrophic forgetting and enable learning models to have an acceptable performance on both current and previous tasks.  
We can summarize the assumptions of lifelong learning as follows: 

1. In contrast to the conventional (supervised) learning, Lifelong learning is not task-specific, and its goal is to learn different tasks over time. This makes lifelong learning as a new method for timeseries learning also.
2. When a new task arrives, the model does not have access to the data of the previous tasks. 
3. It is similar to online learning in a sense that the entire data from different tasks are not available and they arrive during the time; However, despite the online learning, we cannot assume that data samples are from a single data distribution. Each task can have a unique and different distribution. Thus, lifelong learning is a more general problem, and therefore harder to solve.

![](Fig1.png)


# Existing Approaches

In this section, we introduce several well-known lifelong learning methods proposed in the recent years. Generally speaking, these methods can be divided into two groups: (1) data-based approaches and (2) model based. In the following, we elaborate on these two approaches.

## Data-based approaches
**Data-based approaches** use data from the new task to approximate the performance of the previous tasks. This works best if the *distribution mismatch\distance* between tasks is limited. 
These approaches are mainly designed for a classification scenario. They also need to have a preprocessing step before each task, to record the targets for the previous tasks, which is an additional limitation for them. Some examples of this approach are Encoder-based Lifelong Learning (EBLL) and Learning Without Forgetting (LwF).

### Encoder-based Lifelong Learning (EBLL) [2]
In this approach, for each new task, an autoencoder which projects the dataset to a lower dimensional space is learned. Also, a fully connected layer is added to the network
per task. Combining these two ideas, the network can learn different tasks without completely forgetting the previous tasks. The following figure briefly shows this concept.

![](Fig3.jpg)

### Learning Without Forgetting (LwF) [3]
This approach defines three sets of parameters: <img src="https://latex.codecogs.com/gif.latex?\theta_s" style="margin-top: 3px"/> are the shared parameters among all different tasks, <img src="https://latex.codecogs.com/gif.latex?\theta_n" style="margin-top: 3px"/> represents the set of parameters for every given task,
defined specifically for task <img src="https://latex.codecogs.com/gif.latex?n" style="margin-top: 3px"/>, and <img src="https://latex.codecogs.com/gif.latex?\theta_0" style="margin-top: 3px"/> denotes all the parameters from the previous tasks. <img src="https://latex.codecogs.com/gif.latex?\theta_n" style="margin-top: 3px"/> parameters are added to the last layer of the network 
(Typically a fully-connected layer) upon the arrival of new task data. To train this network for the new task, first <img src="https://latex.codecogs.com/gif.latex?\theta_S" style="margin-top: 3px"/> and <img src="https://latex.codecogs.com/gif.latex?\theta_0" style="margin-top: 3px"/>  are  freezed and the network trains
until the convergence of <img src="https://latex.codecogs.com/gif.latex?\theta_n" style="margin-top: 3px"/>. Then these parameters are used as an initilization for a joint training of all parameters in the network. Since this method is based on the 
optimization of the network for the new task data, it can be considered as a data-based approach. The following figure briefly demonstrates this concept.

![](Fig4.jpg)

## Model Based Approaches
Model-based approaches focus on the parameters of the network instead of depending on the task data. They estimate an importance weight for each model parameter and add a regularizer when training a new task that penalizes changes in the important parameters.
The difference between methods in this approach lies in the way that they compute the importance weights. Examples of this approches are Elastic Weight Consolidation, Synaptic Intelligence and Memory Aware Synapses (MAS), which we introduce them in the following.

###  Elastic Weight Consolidation [4]
This method is based on the intution that in an over-parametrized regime (i.e. when the number of nodes in the network is of order of size of dataset) there exists a <img src="https://latex.codecogs.com/gif.latex?\theta_B^*" style="margin-top: 3px"/> solution for task B which is very close to 
<img src="https://latex.codecogs.com/gif.latex?\theta_A^*" style="margin-top: 3px"/>, the solution to task A. Based on this idea, they choose an optimal set of parameters for task B which is within a low-radius ball around <img src="https://latex.codecogs.com/gif.latex?\theta_A^*" style="margin-top: 3px"/>. Below, we illustrate this idea in terms of the optimization path towards it 

![](Fig2.jpg)

###  Synaptic Intelligence (SI) [5]
In the Synaptic Intelligence literature, importance weights are computed during training in an online manner. To this end, they record how much the loss would change due to a change in a specific parameter and accumulate this information over the training trajectory. This method has the following drawbacks:
1. Relying on the weight changes in a batch gradient descent might overestimate the importance of the weights, as noted by the authors of paper [1].
2. When starting from a pre-trained network, as in most practical computer vision applications, some weights might be used without big changes. As a result, their importance will be underestimated.
3. The computation of the importance is done during training and fixed later which can be problematic for testing.

# Memory Aware Synapses (MAS) [1]
The memory-aware synapses is a model-based approach which computes the importance of all network parameters with the gradient of output (e.g. logits) with respect to each weight parameter. 
Then it penalizes the learning objective function based on how much it changes the weights. If a weight parameter has more importance, the objective function is penalized more. By adding this regularizer, 
results have shown MAS outperforms other model-based approaches discussed above.

Below figure shows how MAS is different from other penalty-based approaches. Other penalty-based approaches in the literature estimate the importance of the parameters based on the loss gradient, comparing the network output (light blue) with the ground truth labels (green) using training data (in yellow) (a). In contrast, MAS estimates
the importance of the parameters, after convergence, based on the learned function's sensitivity to their changes (b). This allows using additional unlabeled data points (in orange). This empowers MAS to be used in semisupervised or unsupervised lifelong learning as well as supervised learnings. When learning a new task, changes to important parameters are penalized, the function is preserved over the domain densely sampled in (b) while adjusting not important parameters 
to ensure good performance on the new task (c).

![](MAS1.png)

MAS enjoys several properties such as constant memory size, problem agnostic, supporting unlabeled data, adaptability, and also it can be established on top of a pre-trained network on the previous tasks. Problem agnostic property means this method can be generalized to any dataset and it is not limited to specific tasks or datasets. By adaptability, we mean the capability of a method to adapt the learned model continually to a new task from the same or different environment. Thus this means the samples from different tasks should not necessarily follow a unique distribution and can have different ground truth distributions. Note that, MAS satisfies all the properties mentioned above; However, it might not work equally well on all tasks. 

## MAS can be biased towards one or more tasks! 
As we stated above the MAS approach, computes the importance (gradient of the output function) of parameters in the network with respect to each task. Then it aggregates the absolute values of these gradients and regularizes the objective function. Due to the difference in the distributions of tasks, the scale of the gradients can be very different among the tasks. Thus, it can lead to overfitting one or more tasks, and neglecting the rest. Moreover, these differences in scales can affect the importance estimation compared to the other weights. To clarify this problem, suppose we have a simple neural network with two weight parameters v and w, and two tasks T1 and T2. Let the following table demonstrate the importance of v and w with respect to each task.

![](table.png)

As we can observe, the importance of **v** is 5 times more than **w** in task 1. But since the scales of importances are different among two tasks, at the end, both **v** and **w** nearly have the same importance in the regularization term. Thus, this can lead to poor performance in task 1. To cope with this issue,
we introduce new scaling parameters per task to equalize the importance of different tasks and remove the bias towards one or more tasks. On top of MAS, our proposed MAS approach considers
the same importance for all tasks which makes the final performance independent of tasks order. We refer to this feature as **consistency**. 

Finally, The following table depicts the properties satisfied by different models discussed above.


**Method** | **Type** | **Constant Memory** | **Problem Agnostic** | **On Pre-trained** | **Unlabeled Data** | **Adaptive** | **Consistency**
--- | --- | --- | --- | --- | --- | --- | ---
LwF | Data | :heavy_check_mark: | :x: | :heavy_check_mark: | :x: | :x: | :x:
EBLL | Data | :x: | :x: | :x: | :x: | :x: | :x:
EWC | Model | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x:
IMM | Model | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: | :x:
SI | Model | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: | :x:
MAS | Model | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:
Alpha | Model | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:

# Problem Formulation
We use the model introduced in the paper, to start with and improve it as follows. For a given data point x<sub>k</sub>, the output of the network is  F(x<sub>k</sub>;&theta;). We approximate the gradient as F(x<sub>k</sub>;&theta;+&delta;) - F(x<sub>k</sub>;&theta;) &cong; &sum;<sub>i,j</sub> g<sub>ij</sub>(x<sub>k</sub>)&delta;<sub>ij</sub>
where g<sub>ij</sub>(x<sub>k</sub>) = dF(x<sub>k</sub>;&theta;)/d&theta;<sub>ij</sub> and &delta; = {&delta;<sub>ij</sub>} is a small perturbation, in the parameters &theta; = {&theta;<sub>ij</sub>}. So we consider a few last epochs of the learning to be able to have better estimation of the parameters importance. Our goal is to preserve the prediction of the network (the learned function) at each observed data point and prevent changes to parameters that are important for this prediction. We then accumulate the gradients over the given data points to obtain importance weight &Omega;<sub>tij</sub> in task t for parameter &theta;<sub>ij</sub>, &Omega;<sub>tij</sub>= 1/M &sum;<sub>k</sub> ||g<sub>ij</sub>(x<sub>k</sub>)||,
in which M is the size of training set. When a new task t;
is fetching into the network, we have in addition to the new task prediction error loss L<sub>t</sub>(&theta;), a regularizer that penalizes changes to parameters that are deemed important for previous tasks:

![](loss.png)

With &lambda; a hyperparameter for the regularizer and &theta;<sub>tij</sub>* is the ij parameter learned in task t. We add &alpha;<sub>t</sub> to make sure that we impose a consistency among tasks and so increase the accuracy, i.e. &sum;<sub>ij</sub> &alpha;<sub>t</sub>&Omega;<sub>tij</sub> = &sum;<sub>ij</sub> &alpha;<sub>t'</sub>&Omega;<sub>t'ij</sub>   &forall; t, t'. 
Note that this equation has infinitely many solutions; so, we should add an arbitrary constraint like &sum;<sub>t</sub> &alpha;<sub>t</sub>= &lambda;. Later on, we demonstrate that how this arbitrary constraint can be utilized as a hyperparameter to improve the results.

# Implementation Details
Using the flexibility of *Pytorch* on manipulating the gradients in back-propagation and changing the loss function, we examined our method on the MNIST and CIFAR10 datasets. [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains 10 classes each of which corresponds to a digit. We break 
it into 5 tasks, each including two digits, as follow: task 1: (1,2), task 2: (3,4), task 3: (5,6), task 4: (7,8), task 5: (9,0). In the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets we also divide the dataset
into five tasks: [Task 1: Bird, Automobile], [Task 2: Cat, Deer], [Task 3: Dog, Frog], [Task 4: Horse, Ship], [Task 5: Truck, Airplane]. The implementation is divided into 
three steps: (i) First, we learn the weights of our convolutional neural network by only considering the task 1. Thus, our first task to learn is the learning of the first two 
classes (ii) Then, we train on tasks 2 to 5, consecutively, by avoiding catastrophic forgetting according to the approach described earlier. (iii) Finally, we 
compute the forgetting coefficient for each task which is the difference between *the accuracy of the network trained only for that specific task* and *the accuracy of the network
trained on all tasks together*. The average forgetting will be the average of forgetting values for all tasks. 

# Results
In this section, we provide the results MAS and our variation of this method on the MNIST, and CIFAR10 data-sets. We compare our approach with the baseline introduced by [1] where there is no optimization over alpha parameters. As we can observe, our approach improves the forgetting value more than twice on average for the MNIST dataset. By forgetting the value of a task, we mean the difference between the accuracy of the model, trained jointly on all tasks and the model trained only on that specific task. Since all the MNIST classes have very close distributions (white digits on a black background), the alpha coefficients are very close to each other. For instance, when the sum of alphas equals to 5, the maximum alpha is 1.03, while the minimum one is 0.97. This motivated us to implement our method on CIFAR10 dataset. Actually, since CIFAR10 dataset samples have a diversified class distributions, the vulnerability to the forgetting is higher. 

## Results on the MNIST dataset
Figure below shows the forgetting of each task for three different scenarios. Baseline means the original MAS formulation [1] in which the &alpha;<sub>t</sub> is not optimized and it is equal for all tasks. Other methods are based on optimizing the &alpha;<sub>t</sub>  value for all tasks by solving the earlier explained equations in the problem formulation part. Here, we experimented with hyperparameter &lambda; in &sum;<sub>t</sub> &alpha;<sub>t</sub>= &lambda; and change it to &lambda; == N, 2N, &dots; (N indicates the number of tasks) . By comparing the forgetting value of each task for difference scenarios, it is evident that there is more fluctuation on Baseline compared to our approach. This declares the consistency of our technique on forgetting per task.

![](MNIST1.png)

To have a sense of overall performance of our approach compared to the baseline, we calculate the average and maximum forgetting for all five tasks. Figure below shows the result for average and maximum forgetting. Figure (a) shows that average forgetting is almost 50% lower than the baseline. Also, the maximum forgetting is also more than 3 times lower in our approach compared to the baseline. This plot shows our approach, has the worst-case performance of the neural network on different tasks decreased compared to MAS

![](MNIST2.png)

A naive way of avoiding forgetting is stop learning new tasks! Here, we investigate the validation accuracy of these three scenarios to demonstrate that although we reduce the forgetting, our approach keeps learning. As it can be seen from figure below, at the first epochs our approach with &sum;<sub>t</sub> &alpha;<sub>t</sub>= N has slightly lower validation accuracy. However, it gets better after 13th epoch and it has even higher accuracy than the baseline at some points.

![](MNIST3.png)

Figure below shows the independence of our method considering the order of tasks compared to the baseline. In this experiment, we change the learning order of task [1, 2] from first to fifth. MAS works better if this task is the last one (fifth). But if the task is the first one in the queue of tasks, MAS performance is different and it is less when this task is the last one. However, in our method, the variation in the accuracy by changing the order of task in the task queue is less than MAS, due to the fact that we put equal weights on all tasks.

![](MNIST4.png)

## Results on the CIFAR10 dataset
Figure below includes the results for CIFAR10 with a similar setup that is mentioned for MNIST. Figure (a) shows the validation accuracy of three different settings for 5 epochs. Since CIFAR10 is much larger dataset than MNIST, we confine the number epochs to 5 just to limit the running time. This is also evident from the accuracy percentage which is started from ~50% whereas it is started from ~98% on MNIST. This is due to the fact that we used pretairaned MNIST_NET for MNIST but vgg_16 not pretrained for CIFAR10. Although the accuracy of the baseline is better for these five epochs, the accuracy for our approach is still acceptable and the system keep learning.
Figure (b) shows the forgetting value per task. Figure (c) and (d) illustrate the average and maximum forgetting, respectively. These results indicate significant performance improvement that average forgetting is 30 times less than the baseline when &sum;<sub>t</sub> &alpha;<sub>t</sub>= 2N. We believe this outstanding improvement is because of the fact that the oversimplification of importance update in MAS is clearer in a more complex dataset such as CIFAR10 compared to MNIST dataset.  

![](Res3.png)  

# References
[1]  Aljundi, Rahaf, et al. "Memory aware synapses: Learning what (not) to forget." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

[2] Rannen, Amal, et al. "Encoder based lifelong learning." Proceedings of the IEEE International Conference on Computer Vision. 2017.

[3] Li, Zhizhong, and Derek Hoiem. "Learning without forgetting." IEEE transactions on pattern analysis and machine intelligence 40.12 (2018): 2935-2947.

[4] Kirkpatrick, James, et al. "Overcoming catastrophic forgetting in neural networks." Proceedings of the national academy of sciences 114.13 (2017): 3521-3526.

[5] Zenke, Friedemann, Ben Poole, and Surya Ganguli. "Continual learning through synaptic intelligence." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.
