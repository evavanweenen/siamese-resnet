# Siamese ResNet for food taste similarity
Given a private dataset of 10,000 images of food, the task is to predict food taste similarity between triplets of images.
Given are triplets of images (`anchor`, `positive`, `negative`) (a subset of the 10,000 images), where `anchor` and `positive` are foods similar in taste and `anchor` and `negative` are foods dissimilar in taste, according to human judgement.
For an unseen set of triplets, the task is to forecast whether the second image is closer in taste to the first image than the third image.

I solved the task with a siamese resnet. For each image in the triplet, I use a pretrained ResNet to obtain a lower-dimensional representation of the image. 
The network is siamese in the sense that each image in the triplet is forwarded through the same network, and thus weights are shared. The final layer of the ResNet is reduced to 100 nodes.
I used a triplet loss function [`TripletMarginWithDistanceLoss`](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html), which uses triplets (`anchor`, `positive` and `negative` samples as inputs) and aims to minimze the difference between the distance from `anchor` to `positive` and maximize the distance from `anchor` to `negative`.
As a distance function I used the cosine distance, with a margin of 1. Using this loss function, the lower dimensional representations of the anchor and positive image should be close to each other, and the lower-dimesional representation of the anchor and negative image should be further away from each other. This lower-dimensional space should therefore give a good impression of similarity of images. 
