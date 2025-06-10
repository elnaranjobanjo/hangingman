import pandas as pd
from sklearn.model_selection import train_test_split
import collections
import random
import numpy as np
import re
from collections import defaultdict
import math
import torch
import torch.nn.functional as F
import os
import string
import time
from torch.utils.data import TensorDataset, DataLoader
import ast
import matplotlib.pyplot as plt


import player_agent

from gen_utils import get_train_val_sets  # referee_agent, guessed_container, game_board


class referee_agent:
    def __init__(self, secret_word, num_strikes):
        self.secret_word = secret_word
        self.current_clue = ["_ " for letter in secret_word]
        self.num_strikes = num_strikes
        self.word_letters = set([x for x in secret_word])
        self.game_finished = False
        self.game_won = False
        self.letter_prob_dist = {x: 0 for x in string.ascii_lowercase}
        for letter in secret_word:
            self.letter_prob_dist[letter] += 1 / len(secret_word)

    def provide_word_clue(self):
        return "".join(self.current_clue)

    def update_clue(self, guess):

        for i, real_letter in enumerate(self.secret_word):
            if real_letter == guess:
                self.current_clue[i] = f"{guess} "

    def update_prob_dist(self, guessed_container):
        # print("entered")
        # print(f"{self.letter_prob_dist = }")
        self.letter_prob_dist = {x: 0 for x in string.ascii_lowercase}
        num_unguessed = len([x for x in self.secret_word if x not in guessed_container])
        for letter in self.secret_word:
            if letter not in guessed_container:
                self.letter_prob_dist[letter] += 1 / num_unguessed

        # print(f"{self.letter_prob_dist = }")

    def get_player_guess(self, guess, guessed_container=None):
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
        if guessed_container:
            self.update_prob_dist(guessed_container)
        return correct


class guessed_container:
    def __init__(self):
        self.reset()
        self.letters_to_dim = {x: int(i) for i, x in enumerate(string.ascii_lowercase)}

    def reset(self):
        self.guessed_letters = []
        self.guessed_tensor = torch.zeros(26)

    def update(self, letter):
        self.guessed_letters.append(letter)
        self.guessed_tensor[self.letters_to_dim[letter]] = torch.tensor(1.0)


class game_board:
    def __init__(self, device):
        self.characters_to_dim = {
            x: int(i) for i, x in enumerate(string.ascii_lowercase)
        }
        self.characters_to_dim["_"] = 26.0
        self.characters_to_dim["X"] = 27.0

        self.letters_to_dim = {x: int(i) for i, x in enumerate(string.ascii_lowercase)}
        self.device = device

    def get_single_clue_tensor(self, clean_clue):
        clue_tensor = torch.zeros(30, dtype=torch.long)
        for position, letter in enumerate(clean_clue):
            clue_tensor[position] = self.characters_to_dim[letter]

        # padding
        for position in range(len(clean_clue), 29):
            clue_tensor[position] = self.characters_to_dim["X"]

        return clue_tensor

    def get_batch_clue_tensor(self, clue_batch):

        clean_clues = [x[::2] for x in clue_batch]
        lengths = [torch.tensor(len(x) / 29.0) for x in clean_clues]

        position_encodings = list(
            map(lambda x: self.get_single_clue_tensor(x), clean_clues)
        )
        return torch.stack(position_encodings).to(self.device), torch.stack(lengths).to(
            self.device
        )

    def get_single_guess_tensor(self, guess_letters):
        guess_tensor = torch.zeros(26)
        for guess_letter in guess_letters:
            guess_tensor[self.letters_to_dim[guess_letter]] = torch.tensor(1.0)
        return guess_tensor

    def get_batch_guess_tensor(self, guess_batch):
        guess_batch_decoded = [ast.literal_eval(g) for g in guess_batch]
        guess_encodings = list(
            map(lambda x: self.get_single_guess_tensor(x), guess_batch_decoded)
        )

        return torch.stack(guess_encodings).to(self.device)

    def get_batch_true_letter_prob_dist(self, dist_batch):
        return torch.stack(
            [torch.tensor(ast.literal_eval(dist)) for dist in dist_batch]
        ).to(self.device)


def play_games(
    player,
    word_bank,
    device,
    num_games=10,
    num_strikes=6,
    working_add=None,
):

    board = game_board(device)
    if working_add:
        os.makedirs(working_add, exist_ok=True)
        game_records = []
    results = []
    t_0 = time.time()
    for i, secret_word in enumerate(random.sample(word_bank, k=num_games)):
        # print(f"game_number = {i}")
        # print(f"{secret_word = }")
        # print("Game has started\n")

        referee = referee_agent(secret_word, num_strikes)
        guess_cont = guessed_container()

        if working_add:
            game_records.append(
                [
                    referee.provide_word_clue(),
                    list(player.guessed_letters),
                    list(referee.letter_prob_dist.values()),
                ]
            )

        while not referee.game_finished:
            clue = referee.provide_word_clue()
            # print(f"{clue = }")

            guess = player.guess(clue)

            guess_cont.update(guess)
            if working_add:
                round_result = referee.get_player_guess(
                    guess, guessed_container=player.guessed_letters
                )
            else:
                round_result = referee.get_player_guess(guess)
            # results.append(round_result)
            # print(f"{guess = } ")
            # print(f"Guess correct? {round_result}")
            # print(f"{player.guessed_letters = }")
            # print(f"{referee.letter_prob_dist = }")
            # print(f"Player trials left = {referee.num_strikes}\n")

            if working_add:
                game_records.append(
                    [
                        referee.provide_word_clue(),
                        list(player.guessed_letters),
                        [v for v in referee.letter_prob_dist.values()],
                    ]
                )

        player.reset()

        if i % 100 == 0:
            t_1 = time.time()
            print(f"time for 100 games = {t_1-t_0}")
            t_0 = t_1
        # print(f"Game won? {referee.game_won}\n\n")
        results.append(referee.game_won)

    # print(f"{game_records["clue"].head() = }")
    # print(f"{game_records["guesses"].head() = }")
    # print(f"{game_records["true_prob_distribution"].head() = }")

    if working_add:
        pd.DataFrame(
            game_records, columns=["clue", "guesses", "true_prob_distribution"]
        ).to_csv(os.path.join(working_add, "training.csv"), index=False)
    return results


if __name__ == "__main__":
    random_seed = 137
    random.seed(random_seed)

    train, val = get_train_val_sets()
    num_games = train.shape[0]

    player = player_agent.noob_player(train)
    games_results = play_games(
        player,
        train["word"].to_list(),
        "cpu",
        num_games=num_games,
        num_strikes=np.inf,
        working_add="./",
    )
