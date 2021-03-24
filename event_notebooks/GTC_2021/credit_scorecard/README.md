# Building Credit Risk Scorecards with RAPIDS

This repo contains code referenced from the GTC 2021 talk, "Machine Learning in Retail Credit Risk: Algorithms, Infrastructure, and Alternative Data — Past, Present, and Future [S31327]" by Paul Edwards, Director, Data Science and Model Innovation at Scotiabank.

`/cpu` contains a notebook and tools demonstrating how to build scorecards using weight-of-evidence logistic regression (WOELR) on CPU, using libraries like NumPy, Pandas, and Scikit-learn.

`/gpu` contains a (work-in-progress) notebook and tools accelerating the above work on GPU, using libraries like CuPy, cuDF, and cuML.

This work uses vehicle loan default prediction data from L&T Company, accessible through Kaggle: https://www.kaggle.com/sneharshinde/ltfs-av-data.


