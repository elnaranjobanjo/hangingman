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

from reinforcement_academy import play_games


class hangingman_academy:
    def __init__(self, device):
        self.device = device
        self.batch_size = int(1024)
        self.board = game_board(device)
        self.supervised_epochs = 150
        # self.loss_func = torch.nn.CrossEntropyLoss()

    def play_round():
        pass

    def supervised_penalty(self, pred, true):
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(true * log_probs).sum(dim=-1)
        return loss

    def get_training_assingments_supervised(self, working_dir):
        # raw = pd.read_csv(os.path.join(working_dir, "training.csv"))
        # print(f"size training set = {raw.shape}")
        # clue, length = self.board.get_batch_clue_tensor(raw["clue"].to_list())
        # guesses = self.board.get_batch_guess_tensor(raw["guesses"].to_list())
        # true_prob_dist = self.board.get_batch_true_letter_prob_dist(
        #     raw["true_prob_distribution"].to_list()
        # )
        # training_dataset = TensorDataset(clue, length, guesses, true_prob_dist)
        # torch.save(training_dataset, os.path.join(working_dir, "training_data.pt"))
        training_dataset = torch.load(os.path.join(working_dir, "training_data.pt"))

        return DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def make_supervised_diagnostic(self, losses_in_time, working_dir):

        plt.plot(
            range(1, len(losses_in_time) + 1),
            losses_in_time,
            linestyle="-",
            color="blue",
        )
        plt.scatter(
            range(1, len(losses_in_time) + 1), losses_in_time, color="red", zorder=5
        )
        plt.xlabel("Epochs")
        plt.ylabel("Sum of KL divergence")
        plt.title("Training Loss Over Epochs")
        plt.grid(True)
        plt.savefig(os.path.join(working_dir, "loss_over_epochs.png"))
        plt.close()

    def train_player_supervised(self, working_dir):
        # print("loading training data")
        t_0 = time.time()
        training_assignments = self.get_training_assingments_supervised(working_dir)
        t_1 = time.time()
        print(f"unpacking training data took: {t_1-t_0}")
        brain = player_agent.player_brain_v4().to(self.device)
        optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)

        total_training_time = 0
        losses_in_time = []
        print("training")
        max_min_relative_improvement = 0

        min_relative_improvement = max_min_relative_improvement
        strikes_allowed = int(3)
        strikes_left = strikes_allowed
        for epoch in range(self.supervised_epochs):
            print(f"\n{epoch = }")
            t_0 = time.time()
            total_loss = 0
            for assignment in training_assignments:
                optimizer.zero_grad()
                clue, length, guess, true_dist = assignment
                pred_dist = brain.forward(clue, length, guess)

                # loss = self.supervised_penalty(pred_dist, true_dist).mean()
                # loss = self.loss_func(pred_dist, true_dist).mean()
                loss = F.kl_div(
                    F.log_softmax(pred_dist, dim=-1), true_dist, reduction="batchmean"
                )
                # print(f"{true_dist = }")
                # print(f"{pred_dist = }")
                # print(f"{F.softmax(pred_dist, dim=-1) =}")
                # print(f"{F.log_softmax(pred_dist, dim=-1) = }")
                # print(f"{loss.item() = }\n")
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            epoch_time = time.time() - t_0
            total_training_time += epoch_time
            print(
                f"Approx time to end = {(self.supervised_epochs-epoch)*epoch_time/3600} hrs"
            )
            print(f"time for this epoch is {epoch_time}, epoch loss: {total_loss}")
            losses_in_time.append(total_loss)

            self.make_supervised_diagnostic(losses_in_time, working_dir)
            brain.save(working_dir)

            if epoch > 0:
                prev_loss = losses_in_time[-2]
                curr_loss = losses_in_time[-1]
                loss_improvement = prev_loss - curr_loss
                rel_improvement = loss_improvement / (abs(prev_loss) + 1e-8)
                print(f"{rel_improvement = }")
                if rel_improvement < 0:
                    min_relative_improvement = (
                        rel_improvement if rel_improvement > 0 else 0
                    )
                    strikes_left -= 1
                    print(
                        f"Low improvement {rel_improvement:.4f} — strikes left: {strikes_left}"
                    )
                    if strikes_left == 0:
                        print("Early stopping triggered.")
                        break
                else:
                    strikes_left = strikes_allowed
                    min_relative_improvement = max_min_relative_improvement

        self.make_supervised_diagnostic(losses_in_time, working_dir)
        brain.save(working_dir)
        print("Done training...")
        return brain


if __name__ == "__main__":
    random_seed = 137
    random.seed(random_seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # working_dir = "./toy_train"
    working_dir = "./supervised_results"
    academy = hangingman_academy(device)
    brain = academy.train_player_supervised(working_dir)

    train, val = get_train_val_sets()

    # brains = player_agent.player_brain_v4().to("cpu")
    # brains.load(working_dir)
    play_games(brain, val["word"].to_list(), working_dir)

    # games_results = play_games(
    #     player,
    #     val["word"].to_list(),
    #     device,
    #     num_games=num_games,
    #     num_strikes=6,
    #     working_add=None,
    # )

    # print(
    #     f"Played a total of: {num_games} and won {np.sum(games_results)}\nThis represents a {100*np.sum(games_results)/num_games}% win rate."
    # )


# We'll extract the necessary data for our supervised learning algorithm.
# def get_player_assignments(word_bank):
#     player = player_agent.noob_player()
#     for word in word_bank:
#         pass


# if __name__ == "__main__":
#     random_seed = 137
#     random.seed(random_seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.set_default_dtype(torch.float32)
#     num_strikes = np.inf

#     train, val = get_train_val_sets()
#     player = player_agent.noob_player(train)
#     num_games = train.shape[0]
#     print(f"{train.shape[0] = }")

#     games_results = play_games(
#         player,
#         train["word"].to_list(),
#         num_games=num_games,
#         num_strikes=num_strikes,
#         record_for_training_address="./using_137_seed",
#     )

#     print(
#         f"Played a total of: {num_games} and won {np.sum(games_results)}\nThis represents a {100*np.sum(games_results)/num_games}% win rate."
#     )
