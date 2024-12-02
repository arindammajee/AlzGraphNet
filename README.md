# Graph-MRI-Alzheimer-Detection

This repository contains a project that leverages advanced deep learning techniques, including graph-based neural networks, for the detection of Alzheimer's disease using MRI scans. The work focuses on constructing graphs from patches of 3D medical images and analyzing them using graph convolutional networks (GCNs), graph attention networks (GATs), and graph isomorphism networks (GINs).

The pipeline involves preprocessing MRI data, handling shape inconsistencies, creating graphs with K-Nearest Neighbors (KNN), and training graph-based models for classification between Alzheimer's Disease (AD) and Healthy Control (HC) groups.

# Approach

## Data Preprocessing:
- The pipeline begins by processing 3D MRI scans from the MIRIAD dataset.
- It resolves shape inconsistencies in the scans to ensure uniformity.
- The dataset is split into training, validation, and test sets, ensuring a balanced distribution of AD and HC samples.

## Patch-Based Graph Construction:

MRI scans are divided into patches using a sliding window technique with specified kernel sizes and strides.
Each patch is treated as a node in the graph, and its features are derived from its pixel/voxel data.
Graph adjacency matrices are constructed using the KNN algorithm, ensuring each node is connected to its nearest neighbors.
Graph Neural Networks:

Three types of graph-based models are implemented:
- Graph Convolutional Network (GCN)
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
This project draws inspiration from state-of-the-art research in graph neural networks and medical imaging. Special thanks to the contributors of the MIRIAD dataset and the developers of PyTorch Geometric.

# License
The repository is licensed under the MIT License.
