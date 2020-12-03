# Using RAPIDS and Jupyter to Accelerate Visualization Workflows

Welcome to the repository for Accelerated Visualization Workflows, using RAPIDS. This was a tutorial originally presented for JupyterCon 2020. To start, make sure you have the required installs and hardware, then open the `00 Index and Introduction` notebook to get started. The series is comprised of 5 notebooks, an open source bike share dataset, and embeded walkthrough videos.


## Install Requirements
**Requirements can be met with the Anaconda install below. Make sure to update your cuda_toolkit version to match host cuda version, which currently supports cuda 10.1, 10.2, 11.0**
```
conda env create --name JC_RAPIDSViz --file environment.yml
conda activate JC_RAPIDSViz 
```

## Hardware Requirements
**NVIDIA GPU with at least 16GB of memory that also meets the prerequisites (here)[https://rapids.ai/start.html].