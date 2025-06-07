import pandas as pd
from sklearn.model_selection import train_test_split
import collections
import random
import numpy as np
import re
from collections import defaultdict
import math


def get_train_val_sets():
    all_data = pd.read_csv("words_250000_train.txt", header=None, names=["word"])
    all_data = all_data.dropna(subset=["word"]).reset_index(drop=True)

    train_set, val_set = train_test_split(all_data, test_size=2000, random_state=42)

    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    return train_set, val_set


if __name__ == "__main__":
    # train, val = get_train_val_sets()
    # num_games = 100
    # player = player_agent(train)

    # games_results = play_games(player, val["word"].to_list(), num_games=num_games)

    # print(
    #     f"Played a total of: {num_games} and won {np.sum(games_results)}\nThis represents a {100*np.sum(games_results)/num_games}% win rate."
    # )
