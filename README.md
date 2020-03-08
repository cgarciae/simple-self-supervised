# Simple Self Supervised

Self Supervised Learning (SSL) has become a hot topic due to advances such as Contrastive Predictive Coding (CPC) and the recent SimCLR paper. However, most of the papers / blogs show applications with giant datasets + models, due to this, this projects tries to explore the subject on a simpler 2D dataset of synthetic spirals to make it more accessible. The code is developed using pytorch + skorch.

## Setup

Setup the `poetry` package manager, then run

```bash
poetry install
poetry shell
```

to install the dependencies and activate the environment. To run the experiment execute:

```bash
python training/experiment.py --viz
```

## Method
The repo currently implements a Siamese Network with the embedding space in 1D so that the model is forced to learn a compact representation of the spirals. The most difficult thing is to define similarity in this space since some points can be close in euclidean distance may be in very distant parts of the spiral. For this I defined a graph using a KNN of 6 neighbors, in this graph 2 nodes are positive if they are at a distance <= 5 in graph space, else they are negative. The network is trained in batches by sampling from this graph to predict whether a pair of nodes are positive or negative.

## Results
TODO: Add the images posted on twitter.

[Embeddings Image]
Scatter plot of the spiral, the color is the embedding value, the change is expected to be monotonically increasing / decreasing as you travel through the spiral.

[Graph Image]
Visualization of the expanded graph to nodes of distance <= 5 in the original KNN graph.

[Triplets Image]
Visualization of the anchors, positives, and negatives in a random batch, the anchor-positive pairs are connected by blue lines and the anchor-negative pairs by red lines.

## Additional Resources
**Pytorch Metric Learning**: https://github.com/KevinMusgrave/pytorch-metric-learning
**Andrew Ng's lectute on Siamese Networks**: https://www.youtube.com/watch?v=6jfw8MuKwpI