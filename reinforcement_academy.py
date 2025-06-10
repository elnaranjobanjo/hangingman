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

outer_ind = int(1000)


def make_win_rates_diagnostics(
    i, win_rates, win_stats, losses_in_time, working_dir, train
):
    i = int(i)
    x_vals = range(10 * i // outer_ind)
    m_win_rates, b_win_rates = np.polyfit(x_vals, win_rates, 1)

    plt.scatter(x_vals, win_rates, color="red", zorder=5)
    plt.plot(
        x_vals,
        m_win_rates * x_vals + b_win_rates,
        color="blue",
        linestyle="--",
        label="Trend line",
    )
    plt.xlabel(f"Numbers of games ({outer_ind//10})x games")
    plt.ylabel(f"Win percent over the last {outer_ind} games")
    plt.title(f"Short-term rates over time train = {train}")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"win_rates_in_time_train={train}.png"))
    plt.close()

    x_vals = range(i // outer_ind)
    means = [mean for mean, _ in win_stats]
    stds = [std for _, std in win_stats]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    m_mean, b_mean = np.polyfit(x_vals, means, 1)
    axes[0].scatter(x_vals, means, color="red", zorder=5)
    axes[0].plot(
        x_vals,
        m_mean * x_vals + b_mean,
        color="blue",
        linestyle="--",
        label="Trend line",
    )
    axes[0].set_xlabel(f"Number of games ({outer_ind}x games)")
    axes[0].set_ylabel("Mean")
    axes[0].set_title(f"Mean of win rate over time train={train}")
    axes[0].grid(True)
    axes[0].legend()

    m_std, b_std = np.polyfit(x_vals, stds, 1)
    axes[1].scatter(x_vals, stds, color="red", zorder=5)
    axes[1].plot(
        x_vals,
        m_std * x_vals + b_std,
        color="blue",
        linestyle="--",
        label="Trend line",
    )
    axes[1].set_xlabel(f"Number of games ({outer_ind}x games)")
    axes[1].set_ylabel("Std win rate")
    axes[1].set_title(f"Std of win rate over time train={train}")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"win_stats_summary_train={train}.png"))
    plt.close()

    m_losses_in_time, b_losses_in_time = np.polyfit(x_vals, losses_in_time, 1)
    plt.scatter(x_vals, losses_in_time, color="red", zorder=5)
    plt.plot(
        x_vals,
        m_losses_in_time * x_vals + b_losses_in_time,
        color="blue",
        linestyle="--",
        label="Trend line",
    )
    plt.xlabel(f"Numbers of games ({outer_ind}x games)")
    plt.ylabel(f"loss collected over the last {outer_ind} games")
    plt.title(f"Evolution of loss in time train={train}")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"wloss={train}.png"))
    plt.close()

    # m_longer_period_win_rate, b_longer_period_win_rate = np.polyfit(
    #     x_vals, longer_period_win_rate, 1
    # )
    # plt.scatter(x_vals, longer_period_win_rate, color="red", zorder=5)
    # plt.plot(
    #     x_vals,
    #     m_longer_period_win_rate * x_vals + b_longer_period_win_rate,
    #     color="blue",
    #     linestyle="--",
    #     label="Trend line",
    # )
    # plt.xlabel(f"Numbers of games ({outer_ind}x games)")
    # plt.ylabel(f"Win rate over the last {outer_ind} games")
    # plt.title(f"Win rate over {outer_ind} games train={train}")
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(os.path.join(working_dir, f"wlonger_period_win_rate={train}.png"))
    # plt.close()


def play_games(
    brain,
    word_bank,
    working_dir,
    device="cpu",
    num_strikes=6,
    train=False,
    num_games=np.inf,
):
    print(f"Playing {num_games = }")
    board = game_board(device)
    if train:
        print("Training a player\n\n")
        optimizer = torch.optim.Adam(brain.parameters(), lr=1e-4)
    player = player_agent.player_agent(device)
    player.implant_brain(brain.to("cpu"))

    results = []
    t_0 = time.time()
    i = 0
    longer_period_win_rate = []
    win_rates = []
    win_stats = []
    loss = 0
    losses_in_time = []

    T = 0
    while True:
        i += 1
        if 10 * i % outer_ind == 0:

            t_1 = time.time()
            T += t_1 - t_0
            print(f"time for {outer_ind//10} games = {t_1-t_0}")

            t_0 = t_1

            win_rate = 10 * np.sum(results[-outer_ind // 10 :]) / outer_ind
            win_rates.append(win_rate)
            # time.sleep(3)
            print(
                f"Current win rate over the last {outer_ind//10} games =  {win_rate} win rate.\n\n"
            )
            if i % outer_ind == 0:

                print(f"{i = }")
                # longer_rate = np.sum(results[-outer_ind:]) / outer_ind
                # longer_period_win_rate.append(longer_rate)
                losses_in_time.append(loss.item())
                # print(f"Win rate over the last {outer_ind} games = {longer_rate}")
                mean = np.mean(win_rates[-10:])
                std = np.std(win_rates[-10:])
                print(
                    f"During the last 10 batces of {outer_ind//10} games:\nmean win rate = {mean}\nstd win rate = {std}\n"
                )
                l1, l2 = 0, 0
                if train:
                    l1 = time.time()
                    print(f"Updating AI...")
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"loss collected over last {outer_ind} games = {loss.item()}")

                    print("saved_net")
                    brain.save(working_dir, reinforcement=True)
                    l2 = time.time()
                # print(
                #     f"Time until finished = {(T+(l2-l1)/10)*(len(word_bank)-i)/3600} hrs"
                # )
                T = 0
                loss = 0

                win_stats.append((mean, std))

                if i > 2 * outer_ind:
                    # print(
                    #     f"Overall improvement in means and std {mean - win_stats[-1][0]} and std {std - win_stats[-1][1]}\n\n"
                    # )
                    make_win_rates_diagnostics(
                        i,
                        win_rates,
                        win_stats,
                        losses_in_time,
                        working_dir,
                        train,
                    )
                # time.sleep(3)

        secret_word = random.choice(word_bank)
        # print(f"game_number = {i}")
        # print(f"{secret_word = }")
        # print("Game has started\n")

        referee = referee_agent(secret_word, num_strikes)
        guess_cont = guessed_container()

        log_probs = []
        rewards = []

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

            rewards.append(5.0 if round_result else 1.0)
            log_probs.append(guess_log_prob)

        #     print(f"{clue = }")
        #     print(f"{guess_letter = } ")
        #     print(f"Guess correct? {round_result}")
        #     print(f"{guess_cont.guessed_letters = }")
        #     print(f"Player trials left = {referee.num_strikes}\n")

        # print(f"Game won? {referee.game_won}\n\n")
        game_reward = 15.0 if referee.game_won else 0.5
        rewards = [x + game_reward / len(rewards) for x in rewards]
        loss += sum([-lp * r for lp, r in zip(log_probs, rewards)]) / len(rewards)
        results.append(referee.game_won)

        if i == num_games:
            break

        # if i % 10 == 0:
        #     print(f"{i = }")

    print("Finished")


if __name__ == "__main__":
    random_seed = 137
    random.seed(random_seed)
    torch.set_default_dtype(torch.float32)

    working_dir = "./reinforcement_results"

    train, val = get_train_val_sets()

    brain = player_agent.player_brain_v4()
    # brain = player_agent.player_brain_v2()
    brain.load(working_dir, reinforcement=False)
    play_games(
        brain, train["word"].to_list(), working_dir, train=True, num_games=10000000
    )

    brain.load(working_dir, reinforcement=False)
    play_games(brain, val["word"].to_list(), working_dir, train=False, num_games=25000)
