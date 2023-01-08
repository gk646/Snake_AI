
import statistics
import sys
import time
import threading
import pygame
import random
from src import NeuralNetwork
import numpy as np

import keras
import matplotlib.pyplot as plt



class SnakeGame:
    def __init__(self, tick_rate):
        self.game_over_variable = 0
        self.snake_movement = 1
        self.snake_speed = tick_rate
        self.highest_score1 = 0
        self.highest_score2 = 0
        self.temp_moves = 0
        # Window size
        # defining colors
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.blue = pygame.Color(0, 0, 255)
        self.snake_position = [25, 25]
        self.snake_body = [[25, 25],
                           [24, 25],
                           [23, 25],
                           [22, 25]]
        self.fruit_position = [random.randrange(1, 49),
                               random.randrange(1, 49)]
        self.fruit_spawn = True
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.score = 0
        self.moved_fields = 0
        self.average = 0

    def rest_of_game(self):
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'
        self.moved_fields += 1
        self.temp_moves += 1
        if self.temp_moves > 250:
            self.score = 0
            self.moved_fields = 10
            self.snake_position = [51, 51]
        # Moving the snake
        if self.direction == 'UP':
            self.snake_position[1] -= self.snake_movement
        if self.direction == 'DOWN':
            self.snake_position[1] += self.snake_movement
        if self.direction == 'LEFT':
            self.snake_position[0] -= self.snake_movement
        if self.direction == 'RIGHT':
            self.snake_position[0] += self.snake_movement

        # Snake body growing mechanism
        # if fruits and snakes collide then scores
        # will be incremented by 10
        self.snake_body.insert(0, list(self.snake_position))
        if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
            self.score += 1
            self.fruit_spawn = False
        else:
            self.snake_body.pop()

        if not self.fruit_spawn:
            self.fruit_position = self.fruit_position = [random.randrange(1, 50),
                                                         random.randrange(1, 50)]

        self.fruit_spawn = True

    def play_with_ai(self, model):
        while not self.game_over_variable:
            output = model.predict(self.get_gamestate_4directions())
            action = np.argmax(output)
            if action == 0:
                self.change_to = 'UP'
            elif action == 1:
                self.change_to = 'DOWN'
            elif action == 2:
                self.change_to = 'LEFT'
            elif action == 3:
                self.change_to = 'RIGHT'
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.rest_of_game()
            if self.snake_position[0] < 0 or self.snake_position[0] > 50:
                break
            if self.snake_position[1] < 0 or self.snake_position[1] > 50:
                break
            for block in self.snake_body[1:]:
                if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                    break

    def get_gamestate_4directions(self):
        distance_from_wall_up = self.snake_position[1] - 1
        distance_from_wall_left = self.snake_position[0] - 1
        distance_from_wall_down = abs(self.snake_position[1] - 50 - 1)
        distance_from_wall_right = abs(self.snake_position[0] - 50 - 1)

        distance_from_food_up = self.snake_position[1] - self.fruit_position[1]
        distance_from_food_left = self.snake_position[0] - self.fruit_position[0]
        distance_from_food_down = self.fruit_position[1] - self.snake_position[1]
        distance_from_food_right = self.fruit_position[0] - self.snake_position[0]

        # distance from snake
        return np.array([[distance_from_wall_up, distance_from_wall_left, distance_from_wall_down,
                          distance_from_wall_right, distance_from_food_up, distance_from_food_left,
                          distance_from_food_down, distance_from_food_right]])

    def record(self, model):
        input_array = np.array([])
        player_moves = []
        while not self.game_over_variable:
            input_array = np.append(input_array, self.get_gamestate_4directions())
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.change_to = 'UP'
                    if event.key == pygame.K_DOWN:
                        self.change_to = 'DOWN'
                    if event.key == pygame.K_LEFT:
                        self.change_to = 'LEFT'
                    if event.key == pygame.K_RIGHT:
                        self.change_to = 'RIGHT'
                    if event.key == pygame.K_q:
                        self.game_over_variable = 1
            if self.change_to == 'UP' and self.direction != 'DOWN':
                self.direction = 'UP'

            if self.change_to == 'DOWN' and self.direction != 'UP':
                self.direction = 'DOWN'

            if self.change_to == 'LEFT' and self.direction != 'RIGHT':
                self.direction = 'LEFT'

            if self.change_to == 'RIGHT' and self.direction != 'LEFT':
                self.direction = 'RIGHT'
            self.moved_fields += 1
            self.temp_moves += 1
            if self.temp_moves > 250:
                self.score = 0
                self.moved_fields = 10
                self.snake_position = [51, 51]

            # Moving the snake
            if self.direction == 'UP':
                self.snake_position[1] -= self.snake_movement
                player_moves.append([1, 0, 0, 0])
            if self.direction == 'DOWN':
                self.snake_position[1] += self.snake_movement
                player_moves.append([0, 1, 0, 0])
            if self.direction == 'LEFT':
                self.snake_position[0] -= self.snake_movement
                player_moves.append([0, 0, 1, 0])
            if self.direction == 'RIGHT':
                self.snake_position[0] += self.snake_movement
                player_moves.append([0, 0, 0, 1])
            self.snake_body.insert(0, list(self.snake_position))
            if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
                self.score += 1
                self.fruit_spawn = False
                self.temp_moves = 0
            else:
                self.snake_body.pop()

            if not self.fruit_spawn:
                self.fruit_position = self.fruit_position = [random.randrange(1, 50),
                                                             random.randrange(1, 50)]

            self.fruit_spawn = True

            if self.snake_position[0] < 0 or self.snake_position[0] > 50:
                break
            if self.snake_position[1] < 0 or self.snake_position[1] > 50:
                break
            for block in self.snake_body[1:]:
                if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                    break

        input_array = input_array.reshape((-1, 8))
        player_moves = np.array(player_moves)
        print((input_array.shape, player_moves.shape))
        model.fit(input_array, player_moves, epochs=50)
        model.save('model_help1')

    def reset_game(self):
        self.snake_position = [25, 25]
        self.snake_body = [[25, 25],
                           [24, 25],
                           [23, 25],
                           [22, 25]]
        self.fruit_position = [random.randrange(1, 49),
                               random.randrange(1, 49)]
        self.fruit_spawn = True
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.score = 0
        self.moved_fields = 0
        self.temp_moves = 0


    def new_genetic_try(self, best_model, mode_save_name, highscore):
        scores_plot = []
        c = 0
        new_model = best_model
        self.highest_score1 = highscore
        for i in range(100):
            new_mutant = NeuralNetwork.new_mutant_10_percent(best_model)
            self.reset_game()
            while True:
                output = new_mutant.predict(self.get_gamestate_4directions(), verbose=0)
                action = np.argmax(output)
                if action == 0:
                    self.change_to = 'UP'
                elif action == 1:
                    self.change_to = 'DOWN'
                elif action == 2:
                    self.change_to = 'LEFT'
                elif action == 3:
                    self.change_to = 'RIGHT'
                self.rest_of_game()
                if self.snake_position[0] < 0 or self.snake_position[0] > 50:
                    break
                if self.snake_position[1] < 0 or self.snake_position[1] > 50:
                    break

                for block in self.snake_body[1:]:
                    if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                        break
            score = self.score * 50 + self.moved_fields
            scores_plot.append([score, c])
            c += 1
            if i % 25 == 0:
                print(i)
            if score > self.highest_score1:
                new_model = new_mutant
                print("New Highscore " + str(score))
                self.highest_score1 = score
        new_model.save(mode_save_name)
        iterations, scores = zip(*scores_plot)
        plt.plot(scores, iterations)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Scores over Iterations ' + mode_save_name)
        plt.show()
        print(mode_save_name + " / " + str((max(iterations))) + " / " + str(statistics.mean(iterations)))
        self.average = statistics.mean(iterations)


def genetic(input_model_name):
    pygame.font.init()
    game1 = SnakeGame(0)
    game2 = SnakeGame(0)
    game3 = SnakeGame(0)
    game4 = SnakeGame(0)
    game5 = SnakeGame(0)
    snakeNet1 = keras.models.load_model(input_model_name)
    high_score = 68
    thread1 = threading.Thread(target=game1.new_genetic_try, args=(snakeNet1, '../new_genetic1', high_score))
    thread2 = threading.Thread(target=game2.new_genetic_try, args=(snakeNet1, '../new_genetic2', high_score))
    thread3 = threading.Thread(target=game3.new_genetic_try, args=(snakeNet1, '../new_genetic3', high_score))
    thread4 = threading.Thread(target=game4.new_genetic_try, args=(snakeNet1, '../new_genetic4', high_score))
    thread5 = threading.Thread(target=game5.new_genetic_try, args=(snakeNet1, '../new_genetic5', high_score))

    thread1.start()
    time.sleep(1)
    thread2.start()
    time.sleep(1)
    thread3.start()
    time.sleep(1)
    thread4.start()
    time.sleep(1)
    thread5.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    print("Total Average: " + str((game1.average + game2.average + game3.average + game4.average + game5.average) / 5))
    f = open('averages.txt', 'a')
    f.write(
        "\n" + str((game1.average + game2.average + game3.average + game4.average + game5.average) / 5) + " // " + str(
            game1.average) + "/" +
        str(game2.average) + "/" +
        str(game3.average) + "/" +
        str(game4.average) + "/" +
        str(game5.average))
    f.close()


def record(model_name):
    snakeNetHelp = keras.models.load_model(model_name)
    screen1 = pygame.display.set_mode((500, 500))
    game = SnakeGame(5, screen1)
    game.record(snakeNetHelp)


def test(model_name):
    snakeNetHelp = keras.models.load_model(model_name)
    screen1 = pygame.display.set_mode((500, 500))
    game = SnakeGame(0)
    game.play_with_ai(snakeNetHelp)



genetic('../model_help1')
# 1 is the best
