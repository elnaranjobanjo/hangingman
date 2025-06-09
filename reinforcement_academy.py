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


def play_games(brain, word_bank, working_dir, device="cpu", num_strikes=6, train=False):
    board = game_board(device)
    if train:
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
        if train:
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
        if train:
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
            if train:
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

    brain = player_agent.player_brain_v2()
    brain.load(working_dir)
    play_games(brain, train["word"].to_list(), working_dir, train=True)
