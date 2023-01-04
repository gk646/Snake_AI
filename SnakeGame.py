import statistics
import sys
import time
import threading
import pygame
import random
import NeuralNetwork
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt


class SnakeGame:
    def __init__(self, tick_rate, screen):
        self.game_over_variable = 0
        self.snake_movement = 1
        self.snake_speed = tick_rate
        self.highest_score1 = 0
        self.highest_score2 = 0
        self.temp_moves = 0
        # Window size
        self.window = screen
        # defining colors
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.blue = pygame.Color(0, 0, 255)

        self.surface = pygame.Surface((50, 50))

        self.scaled_surface = pygame.transform.scale(self.surface, (200, 200))
        self.fps = pygame.time.Clock()
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
        self.surface.fill(self.black)

        for pos in self.snake_body:
            pygame.draw.rect(self.surface, self.green,
                             pygame.Rect(pos[0], pos[1], 1, 1))
            pygame.draw.rect(self.surface, self.white, pygame.Rect(
                self.fruit_position[0], self.fruit_position[1], 1, 1))

    def get_game_state(self):
        screen_array = pygame.surfarray.array3d(self.surface)
        gray_array = cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY)
        gray_array = gray_array.reshape(1, 50, 50)
        return gray_array

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

            scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
            self.window.blit(scaled_win, (0, 0))
            pygame.display.flip()
            # Refresh game screen
            pygame.display.update()
            # Frame Per Second /Refresh Rate
            self.fps.tick(self.snake_speed)

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
            self.surface.fill(self.black)

            for pos in self.snake_body:
                pygame.draw.rect(self.surface, self.green,
                                 pygame.Rect(pos[0], pos[1], 1, 1))
                pygame.draw.rect(self.surface, self.white, pygame.Rect(
                    self.fruit_position[0], self.fruit_position[1], 1, 1))
            if self.snake_position[0] < 0 or self.snake_position[0] > 50:
                break
            if self.snake_position[1] < 0 or self.snake_position[1] > 50:
                break
            for block in self.snake_body[1:]:
                if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                    break
            scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
            self.window.blit(scaled_win, (0, 0))
            pygame.display.flip()
            pygame.display.update()
            self.fps.tick(self.snake_speed)

        input_array = input_array.reshape((-1, 8))
        player_moves = np.array(player_moves)
        print((input_array.shape, player_moves.shape))
        model.fit(input_array, player_moves, epochs=25)
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

    def train_models_genetic(self, crossover_1, crossover_2):
        scores_plot = []
        c = 0
        for b in range(25):
            scores = []
            weights = []
            crossover_1, crossover_2 = NeuralNetwork.cross_over(crossover_1, crossover_2)
            print("First Loop - " + str(b))
            for i in range(50):
                half_random = NeuralNetwork.cross_over_into_new(crossover_1, crossover_2)
                self.reset_game()
                while True:
                    output = half_random.predict(self.get_game_state(), verbose=0)
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

                    scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
                    self.window.blit(scaled_win, (0, 0))
                    pygame.display.flip()
                score = self.score * 1000 + self.moved_fields
                weights.append(half_random.get_weights())
                scores.append(score)
            scores = np.array(scores)
            scores_plot.append([scores.mean(), c])
            c += 1
            if np.max(scores) > self.highest_score1:
                crossover_1 = NeuralNetwork.build_model()
                crossover_1.set_weights(weights[scores.argmax()])
                self.highest_score1 = np.max(scores)
            print("Highscore 1: " + str(self.highest_score1) + " ///  Mean: " + str(scores.mean()))
            scores = []
            weights = []
            print("Second Loop - " + str(b))
            for i in range(50):
                half_random2 = NeuralNetwork.cross_over_into_new(crossover_1, crossover_2)
                self.reset_game()
                while True:
                    output = half_random2.predict(self.get_game_state(), verbose=0)
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

                    # Touching the snake body
                    for block in self.snake_body[1:]:
                        if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                            break

                    scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
                    self.window.blit(scaled_win, (0, 0))
                    pygame.display.flip()
                score = self.score * 1000 + self.moved_fields
                weights.append(half_random2.get_weights())
                scores.append(score)
            scores = np.array(scores)
            scores_plot.append([scores.mean(), c])
            c += 1
            if np.max(scores) > self.highest_score2:
                crossover_2 = NeuralNetwork.build_model()
                crossover_2.set_weights(weights[scores.argmax()])
                self.highest_score2 = np.max(scores)
            print("Highscore 2: " + str(self.highest_score2) + " //// Mean: " + str(scores.mean()))
            iterations, scores = zip(*scores_plot)
            plt.plot(scores, iterations)
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.title('Scores over Iterations')
            plt.show()
            best_model, best_model2 = NeuralNetwork.cross_over(crossover_1, crossover_2)
            best_model.save('genetic1')
            best_model2.save('genetic2')

    def train_model_genetic_heuristic(self, crossover_1, crossover_2):
        scores_plot = []
        global_scores = np.array([])
        c = 0
        for b in range(20):
            scores = []
            weights = []
            print("First Loop - " + str(b))
            for i in range(25):
                new_mutant = NeuralNetwork.small_crossover_into_one(crossover_1, crossover_2)
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

                    scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
                    self.window.blit(scaled_win, (0, 0))
                    pygame.display.flip()
                    self.fps.tick(self.snake_speed)
                score = self.score * 50 + self.moved_fields
                global_scores = np.append(global_scores, score)
                if score > self.highest_score1:
                    weights.append(new_mutant.get_weights())
                    scores.append(score)
            scores = np.array(scores)
            scores_plot.append([global_scores.mean(), c])
            c += 1
            if scores.shape[0] != 0 and np.max(scores) > self.highest_score1:
                crossover_1 = NeuralNetwork.build_small_model()
                crossover_1.set_weights(weights[scores.argmax()])
                crossover_1 = NeuralNetwork.local_compile(crossover_1)
                self.highest_score1 = np.max(scores)
            print("Highscore 1: " + str(self.highest_score1) + " ///  Mean: " + str(global_scores.mean()))
            scores = []
            weights = []
            global_scores = np.empty(0, dtype=int)
            print("Second Loop - " + str(b))
            for i in range(25):
                new_mutant2 = NeuralNetwork.small_crossover_into_one(crossover_1, crossover_2)
                self.reset_game()
                while True:
                    output = new_mutant2.predict(self.get_gamestate_4directions(), verbose=0)

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

                    scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
                    self.window.blit(scaled_win, (0, 0))
                    pygame.display.flip()
                score = self.score * 50 + self.moved_fields
                global_scores = np.append(global_scores, score)
                if score > self.highest_score2:
                    weights.append(new_mutant2.get_weights())
                    scores.append(score)
            scores = np.array(scores)
            scores_plot.append([global_scores.mean(), c])
            c += 1
            if scores.shape[0] != 0 and np.max(scores) > self.highest_score2:
                crossover_2 = NeuralNetwork.build_small_model()
                crossover_2.set_weights(weights[scores.argmax()])
                crossover_2 = NeuralNetwork.local_compile(crossover_2)
                self.highest_score2 = np.max(scores)
            print("Highscore 2: " + str(self.highest_score2) + " //// Mean: " + str(global_scores.mean()))
            iterations, scores = zip(*scores_plot)
            plt.plot(scores, iterations)
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.title('Scores over Iterations')
            plt.show()
            best_model, best_model2 = NeuralNetwork.cross_over(crossover_1, crossover_2)
            best_model.save('4Directions1')
            best_model2.save('4Directions2')
            crossover_1, crossover_2 = NeuralNetwork.cross_over(crossover_1, crossover_2)

    def new_genetic_try(self, best_model, mode_save_name, highscore):
        scores_plot = []
        c = 0
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

                scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
                self.window.blit(scaled_win, (0, 0))
                pygame.display.update()
            score = self.score * 50 + self.moved_fields
            scores_plot.append([score, c])
            c += 1

            if i % 25 == 0:
                print(i)
            if score > self.highest_score1:
                best_model = new_mutant
                print("New Highscore " + str(score))
                self.highest_score1 = score
        best_model.save(mode_save_name)
        iterations, scores = zip(*scores_plot)

        plt.plot(scores, iterations)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Scores over Iterations')
        plt.show()
        print(mode_save_name + " / " + str((max(iterations))) + " / " + str(statistics.mean(iterations)))


def genetic():
    pygame.font.init()
    screen1 = pygame.display.set_mode((500, 500))
    screen2 = pygame.display.set_mode((500, 500))
    game1 = SnakeGame(0, screen1)
    game2 = SnakeGame(0, screen2)
    game3 = SnakeGame(0, screen2)
    game4 = SnakeGame(0, screen2)
    game5 = SnakeGame(0, screen2)
    snakeNet1 = keras.models.load_model('new_genetic4')
    high_score = 0
    thread1 = threading.Thread(target=game1.new_genetic_try, args=(snakeNet1, 'new_genetic1', high_score))
    thread2 = threading.Thread(target=game2.new_genetic_try, args=(snakeNet1, 'new_genetic2', high_score))
    thread3 = threading.Thread(target=game3.new_genetic_try, args=(snakeNet1, 'new_genetic3', high_score))
    thread4 = threading.Thread(target=game4.new_genetic_try, args=(snakeNet1, 'new_genetic4', high_score))
    thread5 = threading.Thread(target=game5.new_genetic_try, args=(snakeNet1, 'new_genetic5', high_score))

    thread1.start()
    time.sleep(1)
    thread2.start()
    time.sleep(1)
    thread3.start()
    time.sleep(1)
    thread4.start()
    time.sleep(1)
    thread5.start()


def record():
    snakeNetHelp = keras.models.load_model('model_help1')
    screen1 = pygame.display.set_mode((500, 500))
    game = SnakeGame(5, screen1)
    game.record(snakeNetHelp)

def test():
    snakeNetHelp = keras.models.load_model('new_genetic4')
    screen1 = pygame.display.set_mode((500, 500))
    game = SnakeGame(0, screen1)
    game.play_with_ai(snakeNetHelp)
test()
#4 is the best