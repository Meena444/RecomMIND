import random
import pandas as pd
import numpy as np
import torch as th
from torch.utils.data import DataLoader
import json

from utils import parseArgs, genUserTrainSamples, genUserTestSamples, Dataset, load_metadata, encode_profiles
from model import MIND


def train(args, model, trainData):
    BCELoss = th.nn.BCELoss()
    for epoch in range(args.epochs):
        epochTotalLoss = 0
        for step, (his, tar, userProfile) in enumerate(trainData):
            bs = his.shape[0]
            caps, _ = model.B2IRouting_with_coupling(his, bs)
            logits, labels = model.sampledSoftmax(caps, tar, bs, userProfile)

            loss = BCELoss(logits, labels)
            loss.backward()
            model.opt.step()
            model.opt.zero_grad()
            epochTotalLoss += loss
            if step % args.print_steps == 0:
                print(f"Epoch {epoch:02d} | Step {step:05d} | Loss {epochTotalLoss / (step + 1):.6f}")


if __name__ == '__main__':
    args = parseArgs()
    print("Preparing data...")

    # Load ratings
    ratings = pd.read_csv("data/Appliances.csv", header=None, names=['userId', 'itemId', 'rate', 'timestamp'])

    # Load metadata
    with open("data/meta_Appliances.json") as f:
        metadata = [json.loads(line) for line in f]
    meta_df = load_metadata(metadata)

    # Filter items/users
    itemFreq = ratings.groupby('itemId')['itemId'].count()
    ratings = ratings[ratings['itemId'].isin(itemFreq[itemFreq >= args.min_item_freq].index)]
    userFreq = ratings.groupby('userId')['userId'].count()
    ratings = ratings[ratings['userId'].isin(userFreq[userFreq >= args.min_user_freq].index)]

    # Encode user/item IDs
    user_ids = ratings['userId'].unique()
    userEncId = {uid: i for i, uid in enumerate(user_ids)}
    ratings['userId'] = ratings['userId'].map(userEncId)

    item_ids = ratings['itemId'].unique()
    itemEncId = {iid: i + 1 for i, iid in enumerate(item_ids)}  # item id starts from 1
    ratings['itemId'] = ratings['itemId'].map(itemEncId)

    meta_df = meta_df[meta_df['asin'].isin(itemEncId.keys())]
    meta_df['itemId'] = meta_df['asin'].map(itemEncId)

    ratings.sort_values(by=['userId', 'timestamp'], inplace=True, ignore_index=True)

    # Train/test split
    unique_user_ids = ratings['userId'].unique()
    trainUsers = set(random.sample(list(unique_user_ids), int(len(unique_user_ids) * args.train_user_frac)))
    trainRatings = ratings[ratings['userId'].isin(trainUsers)]
    testRatings = ratings[~ratings['userId'].isin(trainUsers)]

    # Encode user/item profiles (not used yet but can be extended)
    itemProfileEnc, itemProfileDim = encode_profiles(meta_df, itemEncId)
    userProfileEnc, userProfileDim = encode_profiles(pd.DataFrame({'userId': list(userEncId.values())}), userEncId)

    # Prepare training samples
    trainSamples = trainRatings.groupby('userId').apply(lambda x: genUserTrainSamples(args, x)).reset_index(drop=True)
    trainHis = np.concatenate(trainSamples.apply(lambda x: x[0]))
    trainTar = np.concatenate(trainSamples.apply(lambda x: x[1]))
    trainUsers = np.repeat(trainSamples.index.values, trainSamples.apply(lambda x: len(x[1])))

    trainData = DataLoader(
        Dataset(trainHis, trainTar, trainUsers, userProfileEnc),
        batch_size=args.train_batch_size,
        shuffle=True
    )

    # Prepare test samples (used in original version)
    testSamples = testRatings.groupby('userId').apply(lambda x: genUserTestSamples(args, x)).reset_index(drop=True)
    testHis = np.stack(testSamples.apply(lambda x: x[0]))
    _testTar = testSamples.apply(lambda x: x[1])
    testTar = np.arange(len(_testTar))

    print("Start training...")
    model = MIND(args, embedNum=len(itemEncId) + 1, userProfileDim=userProfileDim, itemProfileDim=itemProfileDim)
    train(args, model, trainData)

    # ✅ Save model for use in recommend.py
    th.save(model.state_dict(), "model.pth")
    print("Training complete. Model saved to model.pth")
