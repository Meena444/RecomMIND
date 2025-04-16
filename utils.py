import argparse
import numpy as np
import torch as th
import pandas as pd


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_user_freq', type=int, default=5)
    parser.add_argument('--min_item_freq', type=int, default=5)
    parser.add_argument('--train_user_frac', type=float, default=0.8)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--n_neg', type=int, default=2)
    parser.add_argument('--D', type=int, default=8)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--R', type=int, default=3)
    parser.add_argument('--print_steps', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    return parser.parse_args()


def padOrCut(seq, L):
    if len(seq) < L:
        return np.concatenate([seq, [0] * (L - len(seq))])
    elif len(seq) > L:
        return seq[-L:]
    return seq


def genUserTrainSamples(args, userDf):
    userDf = userDf.reset_index(drop=True)
    his, tar = [], []
    for i in range(1, userDf.shape[0]):
        his.append(padOrCut(userDf.iloc[max(0, i - args.seq_len):i]['itemId'].values, args.seq_len))
        tar.append(userDf.iloc[i]['itemId'])
    return np.stack(his), np.stack(tar)


def genUserTestSamples(args, userDf):
    userDf = userDf.reset_index(drop=True)
    idx = int(0.8 * userDf.shape[0])
    his = padOrCut(userDf['itemId'].iloc[:idx].values, args.seq_len)
    tar = userDf['itemId'].iloc[idx:].values
    return his, tar


class Dataset(th.utils.data.Dataset):
    def __init__(self, his, tar, userIds, userProfileEnc):
        self.his = his
        self.tar = tar
        self.userIds = userIds
        self.userProfiles = th.tensor([userProfileEnc.get(uid, [0]) for uid in userIds], dtype=th.long)

    def __getitem__(self, index):
        return th.tensor(self.his[index]), th.tensor(self.tar[index]), self.userProfiles[index]

    def __len__(self):
        return len(self.tar)


def load_metadata(meta):
    df = pd.json_normalize(meta)
    df = df[['asin', 'brand', 'category']]
    df['brand'] = df['brand'].fillna('unknown')
    df['category'] = df['category'].apply(lambda x: x[0] if isinstance(x, list) and x else 'unknown')
    return df


def encode_profiles(df, id_map):
    if 'brand' in df.columns and 'category' in df.columns:
        cat_col = pd.Categorical(df['brand'].astype(str) + '_' + df['category'].astype(str))
        df['profileId'] = cat_col.codes
        profile_map = {id_map.get(row['itemId'], row.get('userId')): [row['profileId']] for _, row in df.iterrows()}
        return profile_map, len(cat_col.categories)
    else:
        # Fallback for users (no metadata)
        return {id_map.get(row.get('userId')): [0] for _, row in df.iterrows()}, 1

