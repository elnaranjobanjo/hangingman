import pandas as pd
from sklearn.model_selection import train_test_split
import collections
import random
import numpy as np
import re
from collections import defaultdict
import math
import torch

import player_agent


def get_train_val_sets():
    all_data = pd.read_csv("words_250000_train.txt", header=None, names=["word"])
    all_data = all_data.dropna(subset=["word"]).reset_index(drop=True)

    train_set, val_set = train_test_split(all_data, test_size=2000, random_state=42)

    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    return train_set, val_set


class referee_agent:
    def __init__(self, secret_word, num_strikes):
        self.secret_word = secret_word
        self.current_clue = ["_ " for letter in secret_word]
        self.num_strikes = num_strikes
        self.word_letters = set([x for x in secret_word])
        self.game_finished = False
        self.game_won = False

    def provide_word_clue(self):
        return "".join(self.current_clue)

    def update_clue(self, guess):

        for i, real_letter in enumerate(self.secret_word):
            if real_letter == guess:
                self.current_clue[i] = f"{guess} "

    def get_player_guess(self, guess):
        if guess in self.word_letters:
            self.word_letters.remove(guess)
            self.update_clue(guess)
            correct = True
            if len(self.word_letters) == 0:
                self.game_finished = True
                self.game_won = True
        else:
            correct = False
            self.num_strikes += -int(1)
            if self.num_strikes == 0:
                self.game_finished = True
                self.game_won == False

        return correct


def play_games(player, word_bank, num_games=10, num_strikes=6, random_seed=137):
    random.seed(random_seed)
    results = []
    start_with_vowel = 0
    for i, secret_word in enumerate(random.sample(word_bank, k=num_games)):
        print(f"game_number = {i}")
        print(f"{secret_word = }")
        print("Game has started\n")

        referee = referee_agent(secret_word, num_strikes)

        while not referee.game_finished:
            clue = referee.provide_word_clue()
            print(f"{clue = }")
            guess = player.guess(clue)
            round_result = referee.get_player_guess(guess)

            print(f"{guess = } ")
            print(f"Guess correct? {round_result}")
            print(f"Player trials left = {referee.num_strikes}\n")

            if guess in ["a", "e", "o", "i", "u"]:
                start_with_vowel += 1
            # break

        print(f"Game won? {referee.game_won}\n\n")
        results.append(referee.game_won)

        player.reset()

        if i % 100 == 0:
            print(f"\nplayed {i} games.\n")
    # print(f"{start_with_vowel = }")
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.double)
    num_strikes = np.inf
    num_games = 1
    player = player_agent.player_agent(device)

    train, val = get_train_val_sets()

    games_results = play_games(
        player, val["word"].to_list(), num_games=num_games, num_strikes=num_strikes
    )

    print(
        f"Played a total of: {num_games} and won {np.sum(games_results)}\nThis represents a {100*np.sum(games_results)/num_games}% win rate."
    )
