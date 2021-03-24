# Building Credit Risk Scorecards on GPU

#### Code by: Stephen Denton, Scotiabank (stephen.denton@scotiabank.com)

This folder contains a (work-in-progress) notebook and tools accelerating the above work on GPU, using libraries like CuPy, cuDF, and cuML. As this is a work in progress, not all functions work on GPU yet.

`Dockerfile` contains a docker recipe that can be used to execute both the CPU and GPU code:

```
$ docker build . -t rapids_container
$ docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
         -v /path/to/host/data:/rapids/my_data \
         rapids_container
```

You can then attach to the Jupyter server at `http://localhost:8888/lab?`







