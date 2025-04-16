# RecomMIND
# MIND
Pytorch implementation of paper: [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030.pdf)

# Dataset
* [Amazon appliances ratings]
* [Amazon appliances metadata]

# How to run
1. Create a folder called `data`. Download and unzip the data files into the `data` folder, such that the `data` folder contains two files: `meta_Appliances.json` and `Appliances.csv`.
2. Run with default config: `python main.py`
3. Then you can see the output like this:
```
preparing data
start training
Epoch 00 | Step 00000 | Loss 25.838057
Epoch 00 | Step 00200 | Loss 2.774709
Epoch 00 | Step 00400 | Loss 1.765235
Epoch 00 | Step 00600 | Loss 1.416502
...
Epoch 02 | Step 01332 | Loss 0.217909
Epoch 02 | Step 01333 | Loss 0.217899
Epoch 02 | Step 01334 | Loss 0.217884

```
4. This output of training will be stored in a file model.pth
5. After storing, run python recommend.py
6. This asks for Enter a user ID (encoded, 0 to 30251):
7. Enter any user ID, and the top 10 recommended items are printed.
8. Brand and Category are not available in the dataset and so it is unknown.
9. For few users, the Item ID is N/A because, it is not available in the metadata file.
10. The heatmap for a particular user is also displayed for visualisation.
