import torch as th


class MIND(th.nn.Module):
    def __init__(self, args, embedNum, userProfileDim=0, itemProfileDim=0):
        super(MIND, self).__init__()
        self.D = args.D
        self.K = args.K
        self.R = args.R
        self.L = args.seq_len
        self.nNeg = args.n_neg

        self.itemEmbeds = th.nn.Embedding(embedNum, self.D, padding_idx=0)
        self.userProfileEmbeds = th.nn.EmbeddingBag(userProfileDim, self.D, mode='mean') if userProfileDim else None
        self.dense1 = th.nn.Linear(self.D, 4 * self.D)
        self.dense2 = th.nn.Linear(4 * self.D, self.D)

        self.S = th.nn.Parameter(th.randn(self.D, self.D))
        self.B = th.nn.init.normal_(th.empty(self.K, self.L), mean=0.0, std=1.0)

        self.opt = th.optim.Adam(self.parameters(), lr=args.lr)

    def squash(self, caps, bs):
        n = th.norm(caps, dim=2, keepdim=True)
        nSquare = n ** 2
        return (nSquare / ((1 + nSquare) * n + 1e-9)) * caps

    def B2IRouting_with_coupling(self, his, bs):
        B = self.B.detach().expand(bs, self.K, self.L).clone()
        mask = (his != 0).unsqueeze(1).expand(-1, self.K, -1)
        drop = (th.ones_like(mask) * -(1 << 31)).float()

        his_emb = self.itemEmbeds(his)
        his_trans = th.matmul(his_emb, self.S)

        for i in range(self.R):
            BMasked = th.where(mask, B, drop)
            W = th.softmax(BMasked, dim=2)
            caps = th.matmul(W, his_trans)
            caps = self.squash(caps, bs)
            if i < self.R - 1:
                B += th.matmul(caps, his_trans.transpose(1, 2))

        caps = self.dense2(th.relu(self.dense1(caps)))
        return caps, W


    def labelAwareAttation(self, caps, tar, p=2):
        tar = tar.transpose(1, 2)
        w = th.softmax(th.pow(th.matmul(caps, tar).transpose(1, 2), p), dim=2)
        w = w.unsqueeze(2)
        caps = th.matmul(w, caps.unsqueeze(1)).squeeze(2)
        return caps

    def sampledSoftmax(self, caps, tar, bs, userProfile=None, tmp=0.01):
        tarPos = self.itemEmbeds(tar.long())  # <- Fixed here
        capsPos = self.labelAwareAttation(caps, tarPos.unsqueeze(1)).squeeze(1)
        posLogits = th.sigmoid(th.sum(capsPos * tarPos, dim=1) / tmp)

        tarNeg = tarPos[th.multinomial(th.ones(bs), self.nNeg * bs, replacement=True)].view(bs, self.nNeg, self.D)
        capsNeg = self.labelAwareAttation(caps, tarNeg)
        negLogits = th.sigmoid(th.sum(capsNeg * tarNeg, dim=2).reshape(-1) / tmp)

        logits = th.concat([posLogits, negLogits])
        labels = th.concat([th.ones(bs), th.zeros(bs * self.nNeg)])
        return logits, labels
