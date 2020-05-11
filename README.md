# Message passing neural networks for partialcharge assignment in metal-organic frameworks

Jupyter Notebook to produce data for:

> A. Raza, A. Sturluson, C. M. Simon, and X. Fern. Message passing neural networks for partialcharge assignment in metal-organic frameworks
predicting charges on MOFs via a message passing network.

# Structure of repository
```
.
├── ..
├── build_graphs                                        # Directory containing notebooks and information regarding the graph creation process
│   ├── 'Construct graphs representing the MOF.ipynb'   # Notebook where MOFs are used to create corresponding graph and graph features
│   ├── ddec_xtals                                      # Crystal structures with DDEC charges assigned to them{1}
│   ├── graphs                                          # Directory with graph information (edges, node features and node labels) for the DDEC set
│   ├── CoRE_v2_ASR                                     # Crystal structure from CoRE v2 (All Solvents Removed){2}
│   ├── CoRE_v2_FSR                                     # Crystal structure from CoRE v2 (Free Solvents Removed){2}
│   ├── deployment_graphs_ASR                           # Directory with graph information for the CoRE_v2 (All Solvents Removed) set
│   ├── deployment_graphs_FSR                           # Directory with graph information for the CoRE_v2 (Free Solvents Removed) set
│   └── Bonds.jl                                        # Bond generation code. Utilized in graph creation notebook
│
├── MPNN                                                # Directory containing the Message Passing Neural Network (MPNN) code and results
│   ├── model.py                                        # Contains the network and GNN layer
│   ├── charge_prediction_system.py                     # Evaluates different models
│   ├── data_handling.py                                # Reads in graph information from [../build_graphs/graphs] and generates a data list
│   ├── main.py                                         # Includes module for training and testing. Loads in datalist from [./data_handling.py]
│   ├── main.ipynb                                      # Notebook for main.py
│   └── results                                         # Contains results from the MPNN
│       ├── embedding                                   # Element embedding from the MPNN
│       └── results                                     # Results from the MPNN
│
├── embedding_visualization                             # Element Embedding visualizations
│   └── Embedding_Visualization.ipynb                   # Notebook for element embedding visualization. Utilizes UMAP, t-SNE and PCA
│
├── deployment                                          # Code for deployment dataset, where MPNN charges are assigned to the CoRE v2{2} dataset
│   ├── data_handling.py                                # Reads in graph information from [../build_graphs/deployment_graphs[A/F]SR] and generates a data list
│   ├── deployment_main.py                              # Main file for charge predictions for deployment sets.
│   ├── deployment_main.ipynb                           # Notebook for deployment_main.py
│   ├── model_embedding.py                              # ?
│   └── results                                         # Results of charge predictions for the deployment sets
│       └── predictions                                 # Charge predictions
│           ├── deployment_graphs_ASR                   # - for CoRE_v2_ASR
│           └── deployment_graphs_FSR                   # - for CoRE_v2_FSR
│
└── Charge_Assigned_CoRE_MOFs                           # CoRE v2 structures with MPNN charges assigned to them
    ├── MPNN_CoRE-ASR.tar.gz                            # - CoRE v2 ASR (All Solvents Removed) structures with MPNN charges
    └── MPNN_CoRE-FSR.tar.gz                            # - CoRE v2 FSR (Free Solvents Removed) structures with MPNN charges
```

{1}
> J. Chem. Eng. Data 2019, 64, 12, 5985-5998

{2}
> Chem. Mater. 2016, 28, 3, 785-793

