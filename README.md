# Unsupervised Class-Agnostic Instance Segmentation of 3D LiDAR Data for Autonomous Vehicles

**Abstract -** Fine-grained scene understanding is essential for autonomous driving. The context around a vehicle can change drastically while navigating, making it hard to identify and understand the different objects that may appear. Although recent efforts on semantic and panoptic segmentation pushed the field of scene understanding forward, it is still a challenging task. Current methods depend on annotations provided before deployment and are bound by the labeled classes, ignoring long-tailed classes not annotated in the training data due to the scarcity of examples. However, those long-tailed classes, such as baby strollers or unknown animals, can be crucial when interpreting the vehicle surroundings, \eg, for safe interaction. We address the problem of class-agnostic instance segmentation in this paper that also tackles the long-tailed classes. We propose a novel approach and a benchmark for class-agnostic instance segmentation and a thorough evaluation of our method on real-world data. Our method relies on a self-supervised trained network to extract point-wise features to build a graph representation of the point cloud. Then, we use GraphCut to perform foreground and background separation, achieving instance segmentation without requiring any label. Our results show that our approach is able to achieve instance segmentation and a competitive performance compared to state-of-the-art supervised methods.

Source code for our work soon to be published at RA-L:

```
@article{nunes2022ral-3duis,
  author = {Lucas Nunes and Xieyuanli Chen and Rodrigo Marcuzzi and Aljosa Osep and Laura Leal-Taixé and Cyrill Stachniss and Jens Behley},
  title = {{Unsupervised Class-Agnostic Instance Segmentation of 3D LiDAR Data for Autonomous Vehicles}},
  journal = {IEEE Robotics and Automation Letters (RA-L)},
  year = 2022
}
```

More information on the article and code will be published soon.
