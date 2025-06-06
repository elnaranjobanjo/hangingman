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


class referee_agent:
    def __init__(self, secret_word):
        self.secret_word = secret_word
        self.current_clue = ["_ " for letter in secret_word]
        self.num_strikes = int(6)
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


class player_agent:
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


def play_games(player, word_bank, num_games=10, random_seed=137):
    random.seed(random_seed)
    results = []
    start_with_vowel = 0
    for i, secret_word in enumerate(random.sample(word_bank, k=num_games)):
        print(f"game_number = {i}")
        print(f"{secret_word = }")
        print("Game has started\n")

        referee = referee_agent(secret_word)

        while not referee.game_finished:
            clue = referee.provide_word_clue()
            guess = player.guess(clue)
            round_result = referee.get_player_guess(guess)

            print(f"{clue = }")
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
    train, val = get_train_val_sets()
    num_games = 100
    player = player_agent(train)

    games_results = play_games(player, val["word"].to_list(), num_games=num_games)

    print(
        f"Played a total of: {num_games} and won {np.sum(games_results)}\nThis represents a {100*np.sum(games_results)/num_games}% win rate."
    )
