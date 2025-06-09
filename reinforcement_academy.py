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

from gen_utils import get_train_val_sets, referee_agent, guessed_container, game_board


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
        clue_tensor = torch.zeros(29, dtype=torch.long)
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
    word_bank,
    working_dir,
    device="cpu",
    make_new=False,
    num_games=10,
    num_strikes=6,
):
    if make_new:
        brain = player_agent.player_brain()
        brain.to(device)
    else:
        brain = player_agent.player_brain_v2()
        brain.load(working_dir)
        brain.to(device)

    board = game_board(device)
    optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)
    player = player_agent.player_agent(device)
    player.implant_brain(brain)

    results = []
    t_0 = time.time()
    i = 0
    while True:
        secret_word = random.choice(word_bank)
        # print(f"game_number = {i}")
        # print(f"{secret_word = }")
        # print("Game has started\n")

        referee = referee_agent(secret_word, num_strikes)
        guess_cont = guessed_container()
        round_scores = []
        guess_log_prob_record = []

        optimizer.zero_grad()

        while not referee.game_finished:
            clue = referee.provide_word_clue()
            # print(f"{clue = }")
            clue_tensor, length = board.get_single_clue_tensor(clue[::2]).to(
                device
            ), torch.tensor(len(clue[::2]) / 29.0).to(device)

            guess_letter, guess_log_prob = player.guess(
                clue_tensor,
                length,
                guess_cont.guessed_tensor.to(device),
                guess_cont.guessed_letters,
            )
            guess_cont.update(guess_letter)

            round_result = referee.get_player_guess(guess_letter)
            if round_result:
                round_scores.append(1)
            else:
                round_scores.append(-1)

            guess_log_prob_record.append(guess_log_prob)

            # print(f"{clue = }")
            # print(f"{guess = } ")
            # print(f"Guess correct? {round_result}")
            # print(f"{guess_cont.guessed_letters = }")
            # print(f"{referee.letter_prob_dist = }")
            # print(f"Player trials left = {referee.num_strikes}\n")

        if referee.game_won:
            round_scores = [x + 10 / len(round_scores) for x in round_scores]
        # print(f"{guess_log_prob_record = }\n")
        # print(f"{torch.stack(guess_log_prob_record).to(device) = }")
        round_scores = torch.tensor(round_scores)
        loss = -guess_log_prob_record[0] * round_scores[0]
        # torch.autograd.set_detect_anomaly(True)
        for i in range(1, len(round_scores)):
            print(f"{guess_log_prob_record[i] = }")
            print(f"{round_scores[i] = }")
            loss += -guess_log_prob_record[i] * round_scores[i]

        loss.backward()
        optimizer.step()

        # print(f"Game won? {referee.game_won}\n\n")
        results.append(referee.game_won)
        if i % 100 == 0 and i > 0:
            t_1 = time.time()
            print(f"time for 100 games = {t_1-t_0}")
            t_0 = t_1
            brain.save(working_dir, reinforcement=True)
            win_rate = np.sum(results[-100:])
            print(f"Current win rate over the last 100 games =  {win_rate}% win rate.")
            if win_rate >= 60:
                break

        i += 1


if __name__ == "__main__":
    random_seed = 137
    random.seed(random_seed)
    torch.set_default_dtype(torch.float32)

    working_dir = "./reinforcement_results"

    train, val = get_train_val_sets()

    play_games(
        train["word"].to_list(),
        working_dir,
    )
