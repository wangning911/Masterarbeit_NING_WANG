Hello everyone, today is my master thesis final presentation, my topic is Retina net: which relationships  humans pay more attention to.



## Introduction 

*let's begin with the introduction*

**scene graph generation or visual relation detection** is to understand the relationship between any two objects. It plays an important role in scene understanding. It can also connect the computer vision and natural language. 

*Because we need three evaluations in our work, I introduce them here first, then you can understand my work better.*

First one is Predicate Classification, given object classes and bounding boxes to predict the predicates. Second one is Scene graph classification , it is  given the bounding boxes to classify the object and predict the predicates. the last one is the Scene Graph Detection, which is given only an image to detect the objects and the relationship between them.

---

### Background

*At first, I want to introduce the Transformer structure to you, because our work is based on it.*

**Transformer** is a sequence to sequence architecture based on attention mechanism, It consists of an Encoder and a Decoder. Its main module is the Multi-Head Attention. The inputs are Query, Key and Value. Then through a scaled dot-product Attention to get the outputs.  The function is here: $Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_K}})V $

The Transformer can be not only applied to the NLP task, but also to the computer vision area.

---

### Related works

*Now we come to some related works, our model is inspired by them*

**Motifs Net** is a framework for scene graph generation, they use LSTM to encode the global context , that can directly inform the local predictors.

**RelDN** is also a framework for scene graph generation, they propose a set of Graphical Contrastive losses to solve entity instance confusion and proximal relationship ambiguity problems. There are spatial module, semantic module and visual module.

---

 **DETR** is based on a transformer encoder-decoder structure to solve object detection problem.  Its experimental result is as well as traditional Faster R-CNN, and even better. It used a fixed learnable query in each image, and a bipartite matching to match the classes and bounding boxes. In the attention map , each object can be visualised, the model knows where should be paid more attention to. 

---

### Motivation 

*After reading these related works, we propose our model because of the following motivations.*

1. Firstly, because of the success of detr in object detection, we want to use the transformer to do the visual relation detection.
2. secondly, The object query in Detr is not suitable for the predicate classification and scene graph classification.  That query is fixed and obtained from training, it has no physical meaning, so it need a macher. We want to use the known information such as bounding boxes to design a new object query,  which can achieve a one-to-one correspondence with objects. 
3. Thirdly, Attention mechanism is the fundamental principle of transformer, and we hope to obtain some information which is conducive to solving visual relation detection problems . For example, through the attention between objects to explore whether they have a relationship.
4. At last, The most challenging is to predict the correct predicate, which describes the relation between two objects. To realise our goals, we use multihead attention to encode the glabal context.

---

## Our Method 

we propose two methods : Pixel-based attention and Retina Net.

### Pixel-based Attention

#### Idea: 

The idea of Pixel-based attention is to generate attention map to represent the relationship between objects. For example there are three objects in an image, and only one realtionship between $object_1$ and $object_2$, Now we use a mulithead self attention to compute the attention of the whole image, and then we get an attention map between all pixels of the image, like figure three. the purple lines shows the attention of the first pixel in $object_1$  to the whole image. Same ,the green area means the attention of the $object_1$ to $object_2$ . We hope to through the highlights in the attention map to find which pair has realtionship.

---

#### Implementation 

*For the implementaition,* we add a multihead attention model after the VGG backbone to obtain the image feature and an attention map. We design a pixel attention loss to adjust the attenion map.

---

#### Attention Loss

Then we talk about the attention loss funciton in detail. It is a ranking loss.  This ($\frac{1}{m}\sum_{m}Att_j^{no\_rel}$) is the attention of no relation pairs , and this ($\frac{1}{n}\sum_{n}Att_j^{rel}$) is the attention of the relation pair.  we calculate their average. In this attention map the green area is the attention of relation pair, the red areas are the attention of no relation pairs. In the training process, the attention loss can make the green area higher and the red area lower. So that in the evaluation, in the green area should be a highlight, we can know there is a relationship between the $object_1$ and $object_2$.



---

#### Visualised result

*This is the visualised result of our Pixel-based attention methode.*

There is a ground thruth relation pair \<dog sit on beach>, the subject is red box, the object is yellow box. we randomly choice eight pixles from the subject 'dog'. And draw their attention map.  The result shows that the subject pay more attention to itself, or others, like the mountain. but not the object 'beach'.

---

#### Result analysis

In Order to find the reason, we draw the position of the attention, where our loss function work on.  There are a ground truth realtionship \<man ride bike>, and two no relation pairs <man, wheel1> and \<man wheel2>. Their position  in attention map is shown in these three figures. And then we draw the postion of all relation pairs, and all no relation pairs.  The attention loss make this part lower and this part higher. But there is a serious overlap between them. In real image the object wheel1 and object wheel2 are in the object bike, it results in a oerlap bettween relation pairs. so our loss function will not work. and we also find the overlap is not a special case, is very common. Therefore we propose the retina net based on the transformer structure.

---

### Retina Net

#### Implementation

There are four commponents in our retina net:

1. We generate the image feature trough VGG and multihead self attentoin.
2. We design an object decoder to obtain the object feature and object context, and we redesign the object query to suit the visual realtion detection task.
3. We design a relation decoder to get the relation context.
4. Finally is a predicate classifier.

---

#### Generation of Image Feature Maps

The generation of image Feature is same as detr. There are two visual image features, F1 is obtained from VGG backbone and F2 is obtained from Encoder, it has pixel-wise interaction.

---

#### Object decoder

##### Architechture 

*This is the Architechture of object decoder for the retina net.*

It is similar to the original decoder structure, we add a multihead attention module at the end to calculate the object context. This is the setting of the object decoder, we designed a new object query for predicate classification and scene graph classification , and in scene graph detection we use the learnable query same as detr. And it need a bipartite matcher to match the classes and bounding boxes.

Next we will introduce the object query , object context and the attention loss.

---

##### Object query

we obtain the query from the bounding box. Firstly we project the boxes into a matrix, the inside of the box is 1, the outside of the box is 0. Then it minus 0.5 , to get the box mask.  Then through a convolution layer to extract the features, flatten it into a queuence to obtain the object query.

 Our query is one-to-one recorespondence to the object, because it's obtained from bounding boxes. But the query in Detr is fixed in every image, it need a matcher.

---

For ablation study,  we also design other object queries.  We add semantic feature to object query2 base on the obejct query1, and the object query 3 has only simple design.

---

##### Object context

*now I'll show you how to get the object context.*

Our obejct context is obtained from mulithead attention module. The figure show the first calculation of the context, we initial the object context as zero. so the q, k,v are object feature. After softmax we get an attention map between objects. For example, the $A_{02}$ means the attention bettwen $object_0$ and $object_2$. Then the attention map multiply the object feature to obtain the object context. It has interaction between all objects.

---

##### Attention loss

In order to make our obeject feature better, we designed the attention loss.  We can get the attention map from the second multihead attention module in obeject decoder.From each row of it we can extract the attenion map of object.  

We wanna introduce an attention loss to adjust the attention map, make each highlight in attention map can represent a certain object in the real image. Like the right side, the first highlight part represents a jacket, the second represent a bus, and so on.

---

we design  a ranking loss, this  $Att_j^{no\_obj}$ is the attention of the background, and this is the attention of the object. In the attention map , the green area is the object, the blue area is the background, we want to use the attention loss to make the attention weighs of the green area higher and make the attention of the blue area lower.

---

#### Relation decoder

The third component is relation decoder, it is the orignal decoder structure.  The input is the object feature. which consists of visual feature, spatial feature and semantic feature. The visual feature is obtained from object decoder, the spatial feature is extracted from bounding boxes, like RelDN. The semantic feature is extracted by encoding the object classes through word embedding.

If there are three objects, it should has six relation pairs. so the N should be six.

---

##### Relation query

The generation of relation query is similar as the object query, the input is two-channcel box mask, one is the subject boxes, the other is the object boxes.

---

#### Predicate classifier

It is our predicate classifier, we use subject feature, relation context, object context and object feature as inputs, they are obtained from object decoder and relation decoder. 

---

## VG Dataset

now we come to the dataset

We use Visual genome as our dataset. In our works we use about 60 thousand images for training and obout 26 thousand for evaluation. We use Recall@K as an evaluation indicator, which measures the scores of the true relationship triples in the top K. And we have three evalution settings: Predicate classification , scene graph classification and scene graph detection. 

---

## Experimental result

Now I'll show you the experimental result of our retina net. 

#### Object decoder

---

We use a MLP net to do regression test of the object queries and their object features.  The table one shows that the IoU of object queries can reach 0.75 and the IoU of object features can reach 0.6. It prove that there is one-to-one corespondece between the three object queries and objects.

Next we fed the object queries into our model to test the performance. Table 2 shows that their performance are similar. That means the object query don't need a complex design.

---

This is ablation study for object context, we can see from the table, the recall with object context is obvious higher than without object context, which means our object context is very helpful to the predicate classification.

---

#### Attention loss

This is the visualised result of our attention loss. Before using our loss fucntion , the attention map is at a mess. but after using our loss function, the attention map can clearly show the position and the shape of the object.

---

This is the ablation study for the attenten loss, our model with attention loss has better result, it means the loss funciton make the object feature better.

---

#### Relation decoder

This is the visualised result of relation context, in the attention between relation and objects we can see the relation pair pay more attention to their objects. for example the relationship \<food on plate> , in attention map this pair has high attention score on 'food' and 'plate'. The color green represent high attention, the dunkel blue represent low attention.

---

This is the ablation study for the relation decoder, The Table one shows that the relation context is very helpful to the predicate classification. 

The table two shows that the combination of visual feature ,spatial feature and semantic feature can get the best performance.

---

#### Setting of transformer

The multihead attention has two importent parameters: layer and head. But through our experiments we find the layers and the heads don't have a great influence on predicate classification as shown in table 1.

The table two shows that the PredCLS is more suitable for without encoder, and the SGCLS is more suitable for with encoder.

---

#### Result of comparison 

We also compare our model with others. Although our model can't achieve the best result, but it still can complete the visual relation detection task well.

The good things is that the model can avoid some complex calulations and has less parameters. but i think it may be the reason for lack of key information, the result is not very good.

---

### Qualitative Result

#### Predicate classification

This is an instance of predicate classification. There are five relationships in ground truth, our model predicts 3 relationships correctly. The worng realtionship are \<windows on bike>, \<seat on bike> and \<bike has seat>. The possible reason is that the predicate such as 'on' and 'has' are the majority in our dateset, so our model predict them into these.

---

#### Scene graph classification

This is an instance of scene graph classification. There are three relationships in ground truth. but only one relationship is detected. Because our model didn't predict 'building' and 'coat'. 

---

#### scene graph detection 

This is an instance of scene graph detection. There are three relationships in ground truth. but only one relationship is detected. The detected box of 'lamp' is so small, that our model can't detect the relationship.  And our model didn't detect the 'leg'. The relationship \<sceen on desk> is correct, but it is not in ground truth.

---

### Conclusion

*Finally we get some conclusions.*

Frist, Our model Retina net is based on transformer structure. it can solve the visual relation detection task well.

second, our object query is very suitable for predicate classification and scene graph detection.

Third, the attention loss can complete the task very well, which makes our object feature more recognisable and the object can be visualised through the corresponding attention map.

The last, we propose the global object context and the relation context, which significantly improved the final result.

---

## Outline

P4: Introduction

P5: Transformer

P6: Motifs net, RelDN

P7: Detr

P10-14: Pixel-based Attention

P15: Retina-Implementation

P16: Encoder

P17: Object decoder

P20: Object query

P21: Object Context

P22-23: Attention loss

P24-25: Relation decoder

P26: predicate classifier

P28: VG dataset

P30: Ablation study for object decoder

P31: Ablation study for object context

P32: Visualised results of attention loss

P33: Ablation study for attention loss

P34: Visualised result of relation context

P35: Ablation study for Relation decoder

P36: Setting of Transformer

P37: Results comparison

P38-40: Qualitative result

P42: Conclusions



### Question

1. How to set the margin?

   First they are the average of attention weights, attention of all relation pairs and all no relation pairs, then we calculate the average of all attention weighs, it is easy to calculate,becase of softmax operation, the sum of each row in attention map is one, so if the attention map is size of N times N, the average of all attention weighs is 1 device N, so we set the margin as 1 device 4N, this is, the Quarter of average of all attention weights.

2. Object decoder

   The first multihead attention module calculate the interaction between objects, the second multihead attention mudule extract the object feature from image feature, the third multihead attention mudule calculate the object context.

3. Relation decoder

   We obtain  the attention between relation and object. like page34. we can see one relationship pay attention to two objects. 

4. why 64 object features, N relation queries

   We count objects in an image, and we find maximal 62 objects in an image, so we set 64 object queries, then get 64 object features in one image. 

   If there are 3 objects in a image, the rest 61 object queries should be paded as zero. and there should be 6 relation pairs. so the N should be 6.

5. why don't use fasterrcnn?

   Transofomer has same perfermance as fasterrcnn in object detection. Our model is completely based on the transformer structure,  it can avoids some complex calculations and has less parameters.  

6. why your result is low?

   Maybe our object feature is not perfect or our model structure lacks some key information ,which leads to the recall not high.
