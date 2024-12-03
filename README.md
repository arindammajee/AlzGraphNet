# AlzGraphNet: Alzheimer Detection from MRI scans with Graph Neural Networks

This repository contains a project that leverages advanced deep learning techniques, including graph-based neural networks, for the detection of Alzheimer's disease using MRI scans. The work focuses on constructing graphs from patches of 3D medical images and analyzing them using graph convolutional networks (GCNs), graph attention networks (GATs), and graph isomorphism networks (GINs).

The pipeline involves preprocessing MRI data, handling shape inconsistencies, creating graphs with K-Nearest Neighbors (KNN), and training graph-based models for classification between Alzheimer's Disease (AD) and Healthy Control (HC) groups.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Approach

## Data Preprocessing:
- The pipeline begins by processing 3D MRI scans from the MIRIAD dataset.
- It resolves shape inconsistencies in the scans to ensure uniformity.
- The dataset is split into training, validation, and test sets, ensuring a balanced distribution of AD and HC samples.

## Patch-Based Graph Construction:

MRI scans are divided into patches using a sliding window technique with specified kernel sizes and strides.
Each patch is treated as a node in the graph, and its features are derived from its pixel/voxel data.
Graph adjacency matrices are constructed using the KNN algorithm, ensuring each node is connected to its nearest neighbors.

| MRI Image | Constructed Graph from the MRI Scan |
|---|---|
| ![Image](/images/MRI%20Scan%20Sagital%20View.png) | ![Image](images/Constructed%20Brain%20Graph.png) |

## Graph Neural Networks:

Initially we have only used  Graph Convolutional Network (GCN) but we will expand it with other type of networks also -
- Graph Attention Network (GAT)
- Graph Isomorphism Network (GIN)
The models operate on graph representations of MRI scans, leveraging the spatial relationships between patches.

# Evaluation:

Performance is measured using metrics like accuracy, confusion matrices, and classification reports.
The results demonstrate the efficacy of graph-based analysis in medical imaging.

# Features
- Fully customizable configuration for kernel size, stride, and number of neighbors.
- Graph-based models to capture spatial relationships within 3D MRI scans.
- Handles large-scale medical imaging data with preprocessing steps for consistency.

# Dataset Information
This project uses the MIRIAD dataset, which contains MRI scans categorized as AD and HC. After preprocessing, the dataset contains:
- 707 total scans (after filtering inconsistent images).
- Separate training, validation, and test splits with balanced class distributions.

# Requirements
```
Python 3.10+
PyTorch
PyTorch Geometric
scikit-learn
prettytable
```

# How to Use

1. Clone the repository:
   ```
   git clone https://github.com/<your-username>/GraphMRI-Alzheimer-Detection.git  
   cd GraphMRI-Alzheimer-Detection  
   ```
2. Install dependencies:
  ```
  pip install -r requirements.txt  
  ```

3. Update the dataset path in the script or notebook:
  ```
  DATA_PATH = '/path/to/mri/data'
  ``` 
4. Run the project - Execute the notebook step by step to process data, create graphs, and train the models.

# Outputs
- Trained models for Alzheimer's classification.
- Graph visualizations and adjacency matrices.
- Metrics like accuracy, confusion matrix, and precision-recall scores.

# Future Improvements
- Automating configuration for kernel size and dataset path.
- Extending support for additional datasets and graph neural network architectures.
- Enhancing evaluation with cross-dataset validation.

# Acknowledgments
1. Han, Kai, Yunhe Wang, Jianyuan Guo, Yehui Tang, and Enhua Wu. "Vision gnn: An image is worth graph of nodes." Advances in neural information processing systems 35 (2022): 8291-8303.
