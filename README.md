# Message Passing Neural Networks for Partial Charge Assignment to Metal-Organic Frameworks

Jupyter Notebook to produce data for:

[A. Raza, A. Sturluson, C. M. Simon, and X. Fern. Message passing neural networks for partial charge assignment to metal-organic frameworks.](https://chemrxiv.org/articles/Message_Passing_Neural_Networks_for_Partial_Charge_Assignment_to_Metal-Organic_Frameworks/12298487)

## Assigning MPNN charges to a new MOF 
We have created a [Docker](https://www.docker.com/why-docker) image to facilitate assigning MPNN charges to a new MOF. Docker provides an image-based deployment model that makes it easy to share applications with all of their dependencies across multiple environments. Docker image can fully encapsulate not just the code, but the entire dependency stack down to the hardware libraries. 
To assign charges using our docker image, you need to first install docker:  
[https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)  
and download our docker image by typing this command in the terminal:  
`docker pull razaa/mpnn_charge_prediction_image:version1`  

To assign charges to a single MOF (mof_name.cif), make sure `./mpnn_charge_assignment/mpnn_charge_prediction.bash` and the **.cif** file are in the same directory. Use this command to assign charges:  
`./mpnn_charge_prediction.bash mof_name.cif`  
Assigned charges are written to **mof_name_mpnn_charges.cif** file in the current directory.  

To assign charges to multipe MOFs, make sure `./mpnn_charge_assignment/mpnn_charge_prediction.bash` and the directory containing the MOFs are in the same place. Use this command to assign charges:  
`./mpnn_charge_prediction.bash mof_dir`  
MOFs with MPNN charges are stored in **mof_dir**

### Troubleshooting
- If you encounter the following error  
```permission denied: ./mpnn_charge_prediction.bash```  
You need to fix permissions of the script:  
`chmod +x ./mpnn_charge_prediction.bash`  
- Some users reported errors while installing Docker on Windows.If your windows is not updated to the latest build, try older versions of Docker.




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
│   ├── model.py                                        # Contains the neural network model 
│   ├── charge_prediction_system.py                     # For training the model
│   ├── data_handling.py                                # Reads in graphs from [../build_graphs/graphs] and  returns a datalist
│   ├── main.py                                         # Loads in datalist from [./data_handling.py]. Split it into training, validation and testing dataset. Uses [./charge_prediction_system] for training the model [./model.py] and tests it
│   ├── main.ipynb                                      # Notebook for main.py
│   └── results                                         # Contains results from the MPNN
│       ├── embedding                                   # Element embedding from the MPNN
│       └── graphs                                     # Different graphs related to training and testing
│
├── embedding_visualization                             # Element Embedding visualizations
│   └── Embedding_Visualization.ipynb                   # Notebook for element embedding visualization. Utilizes UMAP, t-SNE and PCA
│
├── deployment                                          # Code for deployment dataset, where MPNN charges are assigned to the CoRE v2{2} dataset
│   ├── data_handling.py                                # Reads in graph information from [../build_graphs/deployment_graphs[A/F]SR] and generates a data list
│   ├── deployment_main.py                              # Main file for reading the graphs, loading the model and generating charge predictions for deployment sets
│   ├── deployment_main.ipynb                           # Notebook for deployment_main.py
│   ├── model.py                                        # Required by [./deployment_main.py/ipynb] to load the trained model [./models_deployment.pt]  
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

