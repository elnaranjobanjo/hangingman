import dspy

import numpy as np

# from sentence_transformers import SentenceTransformer
import time
import torch
import random

from gen_utils import get_train_val_sets, referee_agent, guessed_container, game_board

outer_ind = int(1000)


def play_games(
    player,
    word_bank,
    working_dir,
    device="cpu",
    num_strikes=6,
    train=False,
    num_games=np.inf,
):
    print(f"Playing {num_games = }")
    board = game_board(device)

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
        if i == num_games:
            break

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
                # print(f"Win rate over the last {outer_ind} games = {longer_rate}")
                mean = np.mean(win_rates[-10:])
                std = np.std(win_rates[-10:])
                print(
                    f"During the last 10 batces of {outer_ind//10} games:\nmean win rate = {mean}\nstd win rate = {std}\n"
                )

                # print(
                #     f"Time until finished = {(T+(l2-l1)/10)*(len(word_bank)-i)/3600} hrs"
                # )
                T = 0

                win_stats.append((mean, std))

        secret_word = random.choice(word_bank)
        print(f"game_number = {i}")
        print(f"{secret_word = }")
        print("Game has started\n")

        referee = referee_agent(secret_word, num_strikes)
        guess_cont = guessed_container()

        while not referee.game_finished:
            clue = referee.provide_word_clue()

            guess_letter = player.guess(
                clue,
                guess_cont.guessed_letters,
            )
            guess_cont.update(guess_letter)

            round_result = referee.get_player_guess(guess_letter)

            print(f"{clue = }")
            print(f"{guess_letter = } ")
            print(f"Guess correct? {round_result}")
            print(f"{guess_cont.guessed_letters = }")
            print(f"Player trials left = {referee.num_strikes}\n")

        print(f"Game won? {referee.game_won}\n\n")
        results.append(referee.game_won)

        # if i % 10 == 0:
        #     print(f"{i = }")

    print("Finished")


class mistral_player(dspy.Module):
    def __init__(self):
        pass

    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     dspy.configure(
    #         lm=dspy.LM(
    #             "ollama_chat/mistral", api_base="http://localhost:11434", api_key=""
    #         )
    #     )
    #     embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    #     embedding = embedding_model.encode(
    #         "What is retrieval augmented generation?",
    #         device=device,
    #         convert_to_tensor=True,
    #     )
    #     self.respond = dspy.ChainOfThought("context, question -> response")

    # def forward(self, question):
    #     return self.respond(question=question)

    def guess(self, clue, guessed_letters):
        return "e"

    # def forward(self, question):
    #     context = self.search(question).passages
    #     print(f"{context = }")
    #     return self.respond(context=self.search(question).passages, question=question)


if __name__ == "__main__":
    working_dir = "./mistral_results"
    player = mistral_player()
    train, val = get_train_val_sets()

    play_games(player, val["word"].to_list(), working_dir, train=False, num_games=2)
