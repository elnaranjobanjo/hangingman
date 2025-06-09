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

outer_ind = int(10000)


def make_win_rates_diagnostics(i, win_rates, win_stats, losses_in_time, working_dir):
    i = int(i)
    plt.scatter(range(10 * i // outer_ind), win_rates, color="red", zorder=5)
    plt.xlabel(f"Numbers of games ({outer_ind//10})x games")
    plt.ylabel(f"Win percent over the last {outer_ind} games")
    plt.title("Evolution of win rates over time")
    plt.grid(True)
    plt.savefig(os.path.join(working_dir, "win_rates_in_time.png"))
    plt.close()

    plt.scatter(
        range(i // outer_ind), [mean for mean, _ in win_stats], color="red", zorder=5
    )
    plt.xlabel(f"Numbers of games ({outer_ind}x games)")
    plt.ylabel(f"mean of the win rates over the last {outer_ind} games")
    plt.title("Evolution of mean win rate in time")
    plt.grid(True)
    plt.savefig(os.path.join(working_dir, "win_means.png"))
    plt.close()

    plt.scatter(
        range(i // outer_ind), [std for _, std in win_stats], color="red", zorder=5
    )
    plt.xlabel(f"Numbers of games ({outer_ind}x games)")
    plt.ylabel(f"std of the win rates over the last {outer_ind} games")
    plt.title("Evolution of std win rate in time")
    plt.grid(True)
    plt.savefig(os.path.join(working_dir, "win_stds.png"))
    plt.close()

    plt.scatter(range(i // outer_ind), losses_in_time, color="red", zorder=5)
    plt.xlabel(f"Numbers of games ({outer_ind}x games)")
    plt.ylabel(f"loss collected over the last {outer_ind} games")
    plt.title("Evolution of loss in time")
    plt.grid(True)
    plt.savefig(os.path.join(working_dir, "loss.png"))
    plt.close()


def play_games(brain, word_bank, working_dir, device="cpu", num_strikes=6, train=False):
    board = game_board(device)
    if train:
        print("Training a player\n\n")
        optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)
    player = player_agent.player_agent(device)
    player.implant_brain(brain)

    results = []
    t_0 = time.time()
    i = 1
    win_rates = []
    win_stats = []
    loss = 0
    losses_in_time = []
    while True:

        if 10 * i % outer_ind == 0:
            t_1 = time.time()
            print(f"time for {outer_ind//10} games = {t_1-t_0}")
            t_0 = t_1
            win_rate = 10 * np.sum(results[-outer_ind // 10 :]) / outer_ind
            win_rates.append(win_rate)
            print(
                f"Current win rate over the last {outer_ind//10} games =  {win_rate}% win rate.\n\n"
            )
            if i % outer_ind == 0:
                print(f"{i = }")
                losses_in_time.append(loss.item())
                if train:
                    print(f"Updating AI...")
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"loss collected over last {outer_ind} games = {loss.item()}")

                    print("saved_net")
                    brain.save(working_dir, reinforcement=True)
                loss = 0
                mean = np.mean(win_rates[-10:])
                std = np.std(win_rates[-10:])

                print(
                    f"During the last 10 batces of {outer_ind//10} games:\nmean win rate = {mean}\nstd win rate = {std}\n"
                )
                if i > 2 * outer_ind:
                    print(
                        f"Overall improvement in means and std {mean - win_stats[-1][0]} and std {std - win_stats[-1][1]}\n\n"
                    )
                win_stats.append((mean, std))
                make_win_rates_diagnostics(
                    i, win_rates, win_stats, losses_in_time, working_dir
                )

        secret_word = random.choice(word_bank)
        # print(f"game_number = {i}")
        # print(f"{secret_word = }")
        # print("Game has started\n")

        referee = referee_agent(secret_word, num_strikes)
        guess_cont = guessed_container()

        while not referee.game_finished:
            clue = referee.provide_word_clue()
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
                loss += -guess_log_prob
            else:
                loss += guess_log_prob

        #     print(f"{clue = }")
        #     print(f"{guess_letter = } ")
        #     print(f"Guess correct? {round_result}")
        #     print(f"{guess_cont.guessed_letters = }")
        #     print(f"Player trials left = {referee.num_strikes}\n")

        # print(f"Game won? {referee.game_won}\n\n")
        results.append(referee.game_won)

        if referee.game_won:
            loss += -guess_log_prob
        else:
            loss += guess_log_prob

        i += 1
        if i > 220000:
            break

    print("Finished")


if __name__ == "__main__":
    random_seed = 137
    random.seed(random_seed)
    torch.set_default_dtype(torch.float32)

    working_dir = "./reinforcement_results"

    train, val = get_train_val_sets()

    # brain = player_agent.player_brain_v4()
    brain = player_agent.player_brain_v2()
    brain.load(working_dir, reinforcement=False)

    play_games(brain, train["word"].to_list(), working_dir, train=True)
