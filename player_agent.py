import torch.nn as nn
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import collections
import random
import numpy as np
import re
from collections import defaultdict
import math
from dataclasses import dataclass
import string
import os
import torch.nn.functional as F


# Player brain v1
# class player_brain(nn.Module):
#     def __init__(
#         self,
#     ):
#         super().__init__()
#         max_word_len = 29

#         self.vocab_size = 28
#         self.embedding_dim = 32

#         self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

#         self.hidden_dim = 64
#         self.lstm = nn.LSTM(
#             input_size=self.embedding_dim, hidden_size=self.hidden_dim, batch_first=True
#         )

#         guess_dim = 26  # 26 letters

#         self.classifier_input_size = self.hidden_dim + 1 + guess_dim

#         self.classifier = nn.Sequential(
#             nn.Linear(self.classifier_input_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 26),
#         )

#     def forward(self, clue_tensor, length, guessed_tensor):

#         embedded_clue = self.embedding(clue_tensor)  # shape: [1, clue_len, emb_dim]
#         lstm_out, (h_n, c_n) = self.lstm(embedded_clue)
#         x = torch.cat([lstm_out[:, -1, :], length.view(-1, 1), guessed_tensor], dim=1)
#         return self.classifier(x)

#     def save(
#         self,
#         working_dir: str,
#     ) -> None:
#         torch.save(self.state_dict(), os.path.join(working_dir, "player_brain.pt"))

#     def load(self, working_dir):
#         self.load_state_dict(torch.load(os.path.join(working_dir, "player_brain.pt")))


# player brain v2
class player_brain_v2(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        max_word_len = 29

        self.vocab_size = 28
        self.embedding_dim = 32

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.hidden_dim = 128
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=self.hidden_dim, batch_first=True
        )

        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        guess_dim = 26  # 26 letters

        self.guessed_embedding = nn.Sequential(
            nn.Linear(26, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 26),
        )

        self.classifier_input_size = self.hidden_dim + 1 + guess_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 26),
        )

    def forward(self, clue_tensor, length, guessed_tensor):
        # print(f"{clue_tensor.shape = }")
        embedded_clue = self.embedding(clue_tensor)
        lstm_out, (h_n, c_n) = self.lstm(embedded_clue)
        # clue_out = torch.nn.AdaptiveAvgPool1d(1)
        # print(f"{lstm_out.shape = }")
        # print(f"{lstm_out[:,-1,:].shape = }")
        # print(f"{self.pooling(lstm_out).shape = }")
        # print(f"{lstm_out.transpose(1, 2).shape = }")
        # print(f"{self.pooling(lstm_out.transpose(1, 2)).squeeze(-1).shape = }")
        guess_embedded = self.guessed_embedding(guessed_tensor)

        x = torch.cat(
            [
                self.pooling(lstm_out.transpose(1, 2)).squeeze(-1),
                length.view(-1, 1),
                guess_embedded,
            ],
            dim=1,
        )
        return self.classifier(x)

    def save(self, working_dir: str, reinforcement=False) -> None:
        if reinforcement:
            torch.save(
                self.state_dict(),
                os.path.join(working_dir, "player_brain_reinforced_v2.pt"),
            )
        else:
            torch.save(
                self.state_dict(),
                os.path.join(working_dir, "player_brain_supervised_v2.pt"),
            )

    def load(self, working_dir, reinforcement=False):
        if reinforcement:
            self.load_state_dict(
                torch.load(os.path.join(working_dir, "player_brain_reinforced_v2.pt"))
            )
        else:
            self.load_state_dict(
                torch.load(os.path.join(working_dir, "player_brain_supervised_v2.pt"))
            )


# v3
class player_brain_v3(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.max_word_len = 29

        self.vocab_size = 28
        self.embedding_dim = 64

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.hidden_dim = 128

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.max_word_len, self.embedding_dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=4,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        guess_dim = 64  # 26 letters

        self.guessed_embedding = self.guessed_embedding = nn.Sequential(
            nn.Linear(26, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh()
        )

        self.classifier_input_size = self.embedding_dim + 1 + guess_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 26),
        )

    def forward(self, clue_tensor, length, guessed_tensor):

        x = self.embedding(clue_tensor)
        pos = self.pos_embedding[:, : x.size(1), :]
        x = x + pos
        x = self.transformer_encoder(x)
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)

        guessed = self.guessed_embedding(guessed_tensor)

        out = torch.cat([x, length.view(-1, 1), guessed], dim=1)

        return self.classifier(out)

    def save(self, working_dir: str, reinforcement=False) -> None:
        if reinforcement:
            torch.save(
                self.state_dict(),
                os.path.join(working_dir, "player_brain_reinforced_v3.pt"),
            )
        else:
            torch.save(
                self.state_dict(),
                os.path.join(working_dir, "player_brain_supervised_v3.pt"),
            )

    def load(self, working_dir, reinforcement=False):
        if reinforcement:
            self.load_state_dict(
                torch.load(os.path.join(working_dir, "player_brain_reinforced_v3.pt"))
            )
        else:
            self.load_state_dict(
                torch.load(os.path.join(working_dir, "player_brain_supervised_v3.pt"))
            )


class player_agent:
    def __init__(self, device):
        self.device = device
        self.brain = None
        self.dim_to_letter = {int(i): x for i, x in enumerate(string.ascii_lowercase)}

    def implant_brain(self, brain):
        self.brain = brain
        return self

    def guess(self, clue, length, guess_tensor, guess_letters):
        # clue, length = self.encode_clue(clue)
        guess_letter_distribution = self.brain.forward(
            clue.unsqueeze(0),
            length.view(1),
            guess_tensor.unsqueeze(0),
        ).squeeze()
        guess_letter_distribution = F.log_softmax(guess_letter_distribution, dim=-1)
        # print(f"Player letter guess distribution {guess_letter_distribution}")
        letter_priority = guess_letter_distribution.argsort(descending=True)
        # print(f"{letter_priority = }")
        letter_selection = [
            (self.dim_to_letter[int(pos.item())], guess_letter_distribution[pos])
            for pos in letter_priority
        ]
        # print(f"{letter_selection = }")
        for letter, log_prob in letter_selection:
            if letter not in guess_letters:
                guess_letter = letter
                guess_log_prob = log_prob
                break

        return guess_letter, guess_log_prob


# We use the noob in order to obtain training data
class noob_player:
    def __init__(self, training_set: list):
        self.full_dictionary = training_set
        self.full_dictionary["length"] = training_set["word"].str.len().astype(int)
        # self.full_dictionary["length"] = self.build_dictionary(training_set)
        self.guessed_letters = []

        self.full_dictionary_common_letter_sorted = collections.Counter(
            "".join(self.full_dictionary["word"].to_list())
        ).most_common()

        self.current_dictionary = self.full_dictionary["word"].to_list()

    # def build_dictionary(self, training_set):
    #     with open("words_250000_train.txt", "r") as text_file:
    #         full_dictionary = text_file.read().splitlines()
    #     print(f"{full_dictionary = }")
    #     return full_dictionary

    # def start_game(self, clue):
    #     clean_word = clue[::2].replace("_", ".")

    #     self.relevant_words = self.full_dictionary["word"][
    #         self.full_dictionary["length"] == len(clean_word)
    #     ].to_list()

    def reset(self):
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary["word"].to_list()

    def guess(self, word):

        clean_word = word[::2].replace("_", ".")

        len_word = len(clean_word)

        current_dictionary = self.current_dictionary

        new_dictionary = []

        for dict_word in current_dictionary:

            if len(dict_word) != len_word:
                continue

            if any(
                letter in dict_word
                for letter in self.guessed_letters
                if letter not in clean_word
            ):
                continue

            if re.fullmatch(clean_word, dict_word):
                new_dictionary.append(dict_word)

        self.current_dictionary = new_dictionary

        full_dict_string = "".join(new_dictionary)

        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()

        letter_scores = defaultdict(int)

        for word in new_dictionary:
            for i, letter in enumerate(word):
                if clean_word[i] == "." and letter not in self.guessed_letters:
                    letter_scores[letter] += 1

        sorted_letter_count = sorted(letter_scores.items(), key=lambda x: -x[1])

        # candidate_letters = set("abcdefghijklmnopqrstuvwxyz") - set(
        #     self.guessed_letters
        # )

        # entropy_scores = {}

        # for letter in candidate_letters:
        #     count_in = sum(1 for w in new_dictionary if letter in w)
        #     count_out = len(new_dictionary) - count_in

        #     if count_in == 0 or count_out == 0:
        #         entropy_scores[letter] = 0  # no information gained
        #         continue

        #     p_in = count_in / len(new_dictionary)
        #     p_out = count_out / len(new_dictionary)

        #     entropy = -p_in * math.log2(p_in) - p_out * math.log2(p_out)
        #     entropy_scores[letter] = entropy

        # # Sort by descending entropy
        # sorted_letter_count = sorted(entropy_scores.items(), key=lambda x: -x[1])

        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        guess_letter = "!"
        for letter, instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break

        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == "!":
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter, instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break

        self.guessed_letters.append(guess_letter)
        return guess_letter
