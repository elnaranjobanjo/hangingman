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


class player_brain(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        max_word_len = 29
        clue_dim = max_word_len + 1  # 29 letter spots + actual_length_of_word
        guess_dim = 26  # 26 letters
        self.input_size = clue_dim + guess_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 26),  # one score per letter
        )

    def forward(self, clue_tensor, guessed_tensor):
        x = torch.cat([clue_tensor, guessed_tensor])
        return self.net(x)


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


class player_agent:
    def __init__(self, device):
        self.device = device
        self.brain = player_brain().to(device)
        self.already_guessed = []
        self.guessed = guessed_container()
        self.characters_to_dim = self.guessed.letters_to_dim
        self.characters_to_dim["_"] = 26.0
        self.characters_to_dim["X"] = 27.0
        self.dim_to_letter = {int(i): x for i, x in enumerate(string.ascii_lowercase)}

    def encode_clue(self, clue):
        clean_clue = clue[::2]
        length_word = len(clean_clue)
        clue_tensor = torch.zeros(30)
        for position, letter in enumerate(clean_clue):
            clue_tensor[position] = torch.tensor(self.characters_to_dim[letter])

        # padding
        for position in range(length_word, 29):
            clue_tensor[position] = torch.tensor(self.characters_to_dim["X"])

        clue_tensor[29] = torch.tensor(length_word)
        # print(f"{clue_tensor = }")
        return clue_tensor

    def guess(self, clue):
        # print(f"player saw clue: {clue}")

        guess_letter_distribution = self.brain.forward(
            self.encode_clue(clue).to(self.device),
            self.guessed.guessed_tensor.to(self.device),
        )
        letter_priority = guess_letter_distribution.argsort(descending=True)
        letter_selection = [
            self.dim_to_letter[int(pos.item())] for pos in letter_priority
        ]
        for letter in letter_selection:
            if letter not in self.guessed.guessed_letters:
                guess_letter = letter
                break
        self.guessed.update(guess_letter)

        return guess_letter

    def reset(self):
        self.guessed = guessed_container()

        # print(f"{a = }")

    #     self.full_dictionary = training_set
    #     self.full_dictionary["length"] = training_set["word"].str.len().astype(int)
    #     # self.full_dictionary["length"] = self.build_dictionary(training_set)
    #     self.guessed_letters = []

    #     self.full_dictionary_common_letter_sorted = collections.Counter(
    #         "".join(self.full_dictionary["word"].to_list())
    #     ).most_common()

    #     self.current_dictionary = self.full_dictionary["word"].to_list()

    # # def build_dictionary(self, training_set):
    # #     with open("words_250000_train.txt", "r") as text_file:
    # #         full_dictionary = text_file.read().splitlines()
    # #     print(f"{full_dictionary = }")
    # #     return full_dictionary

    # # def start_game(self, clue):
    # #     clean_word = clue[::2].replace("_", ".")

    # #     self.relevant_words = self.full_dictionary["word"][
    # #         self.full_dictionary["length"] == len(clean_word)
    # #     ].to_list()

    # def reset(self):
    #     self.guessed_letters = []
    #     self.current_dictionary = self.full_dictionary["word"].to_list()

    # def guess(self, word):

    #     clean_word = word[::2].replace("_", ".")

    #     len_word = len(clean_word)

    #     current_dictionary = self.current_dictionary

    #     new_dictionary = []

    #     for dict_word in current_dictionary:

    #         if len(dict_word) != len_word:
    #             continue

    #         if any(
    #             letter in dict_word
    #             for letter in self.guessed_letters
    #             if letter not in clean_word
    #         ):
    #             continue

    #         if re.fullmatch(clean_word, dict_word):
    #             new_dictionary.append(dict_word)

    #     self.current_dictionary = new_dictionary

    #     full_dict_string = "".join(new_dictionary)

    #     c = collections.Counter(full_dict_string)
    #     sorted_letter_count = c.most_common()

    #     letter_scores = defaultdict(int)

    #     for word in new_dictionary:
    #         for i, letter in enumerate(word):
    #             if clean_word[i] == "." and letter not in self.guessed_letters:
    #                 letter_scores[letter] += 1

    #     sorted_letter_count = sorted(letter_scores.items(), key=lambda x: -x[1])

    #     # candidate_letters = set("abcdefghijklmnopqrstuvwxyz") - set(
    #     #     self.guessed_letters
    #     # )

    #     # entropy_scores = {}

    #     # for letter in candidate_letters:
    #     #     count_in = sum(1 for w in new_dictionary if letter in w)
    #     #     count_out = len(new_dictionary) - count_in

    #     #     if count_in == 0 or count_out == 0:
    #     #         entropy_scores[letter] = 0  # no information gained
    #     #         continue

    #     #     p_in = count_in / len(new_dictionary)
    #     #     p_out = count_out / len(new_dictionary)

    #     #     entropy = -p_in * math.log2(p_in) - p_out * math.log2(p_out)
    #     #     entropy_scores[letter] = entropy

    #     # # Sort by descending entropy
    #     # sorted_letter_count = sorted(entropy_scores.items(), key=lambda x: -x[1])

    #     # return most frequently occurring letter in all possible words that hasn't been guessed yet
    #     guess_letter = "!"
    #     for letter, instance_count in sorted_letter_count:
    #         if letter not in self.guessed_letters:
    #             guess_letter = letter
    #             break

    #     # if no word matches in training dictionary, default back to ordering of full dictionary
    #     if guess_letter == "!":
    #         sorted_letter_count = self.full_dictionary_common_letter_sorted
    #         for letter, instance_count in sorted_letter_count:
    #             if letter not in self.guessed_letters:
    #                 guess_letter = letter
    #                 break

    #     self.guessed_letters.append(guess_letter)
    #     return guess_letter
