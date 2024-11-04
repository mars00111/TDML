# TDML - A Trustworthy Distributed Machine Learning Framework

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This demonstration showcases the research in the paper of 'TDML - A Trustworthy Distributed Machine Learning Framework'. While we aim to share the framework's core functionality, the stakeholders have decided to pursue a patent. Consequently, only essential demonstration code related to the paper can be made publicly available at this time. We hope this provides valuable context and understanding of TDML's capabilities.

## Performance

The following diagrams show our framework performance based on the ResNet-50 with the CIFAR-10 dataset. Since our framework includes data parallelism (DP) and pipeline model parallelism (MP) for large model training, the report schema has two parts: (i) [m] DP and (ii) MP client[N]. The first part, similar to FedAvg, divides the dataset into M pieces with independent model training and aggregates the local models into a global model after each epoch. The second part, MP client[$N$], represents the pipeline model parallelism training where the entire model is split into $N$ shards to ease memory requirements. For example, "2 DP + MP, client[2]" means the training set is divided into two with independent models, and each model is split into two shards across two computing nodes. 

<div style="display: flex; justify-content: space-between;">
    <figure style="margin-right: 10px;">
        <img src="imgs/our_test_acc.jpg" alt="Figure 1" width="300">
        <figcaption style="text-align: center;">Figure 1: Our framework's testing accuracy</figcaption>
    </figure>
    <figure>
        <img src="imgs/baseline_test_acc.jpg" alt="Figure 2" width="300">
        <figcaption style="text-align: center;">Figure 2: Baseline testing accuracy</figcaption>
    </figure>
</div>
