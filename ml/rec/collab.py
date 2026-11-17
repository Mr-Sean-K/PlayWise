# [FAST]   Howard, J. (2022). Practical Deep Learning for Coders [Online course], Lesson 7: Collaborative filtering. fast.ai. https://course.fast.ai/
# [FASTAI] Howard, J. & Gugger, S. (2020). fastai: A Layered API for Deep Learning. Information, 11(2), 108.
# [KOREN]  Koren, Y., Bell, R. & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. IEEE Computer, 42(8), 30-37.

import torch
from torch.nn import Module, Embedding
from fastai.layers import Embedding, sigmoid_range


N_FACTORS = 50 # embedding size used in the course [FAST]
Y_RANGE = (0, 5.5) # top slightly above the max score, since a sigmoid never reaches its upper limit [FAST]

# matrix factorisation model with user and game embeddings and biases.
class dotProductBias(Module):

    # prediction = user bias + item bias + user factors · item factors which follows the "biased matrix factorisation" model in [KOREN]

    # Embedding = lookup table of learned vectors, one row per user/game [FAST].
    # fastai's Embedding adds its own weight initialisation [FASTAI].
    def __init__(self, numUsers, numItems, numFactors=N_FACTORS, yRange=Y_RANGE):
        self.userFactors = Embedding(numUsers, numFactors)
        self.userBias = Embedding(numUsers, 1)
        self.gameFactors = Embedding(numItems, numFactors)
        self.gameBias = Embedding(numItems, 1)
        self.yRange = yRange

    def forward(self, x):
        # x is a 2D array of user-game pairs where each row contains a user ID and a game ID [FAST]
        users = self.user_factors(x[:, 0])
        games = self.game_factors(x[:, 1])
        result = (users * games).sum(dim=1, keepdim=True) # dot product = taste profile from multiplying vector elements together and summing the result [FAST]
        result += self.user_bias(x[:, 0]) + self.game_bias(x[:, 1]) # add biases - each bias is one known number per user/game [FAST]
        return sigmoid_range(result, *self.y_range)

