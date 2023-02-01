import pygame
from copy import deepcopy
import math
from Net import Net
import time
import torch
import random
from evotorch.neuroevolution import NEProblem
import itertools
from multiprocessing import Process, Manager, Array
import numpy as np
import ctypes as ct
from custom_searcher import SimpleGA

white = (255,255,255)
green = (124, 252, 0)
black = (0, 0, 0)
red = (255, 0, 0)

# the array representation can be used, to draw the path on the screen
# for the AI we need the path as a weighted graph
def make_graph():
    # array of path is an array which is build as follows:
    # [[[x_start, y_start], [x_end, y_end]], [[x_start, y_start], [x_end, y_end]], ...]
    # x_start and y_start are as near as possible to the left upper corner of the screen. This is checked
    # one Block in the real game is 20 Pixel wide
    # that the array isn't too Long on the screen, we divide it in parts
    global Array_of_path
    Array_of_path = [[[200, 280], [320, 280]], [[200, 100], [300, 100]], [[140, 100], [140, 280]], [[100, 100], [100, 160]], [[100, 160], [180, 160]], [[180, 160], [180, 260]], 
                    [[100, 280], [140, 280]], [[100, 200], [100, 280]], [[100, 240], [180, 240]], [[180, 260], [200, 260]], [[100, 100], [140, 100]], [[250, 220], [250, 280]], 
                    [[200, 260], [200, 280]], [[180, 190], [220, 190]], [[140, 120], [200, 120]], [[200, 100], [200, 120]], [[100, 200], [180, 200]], [[250, 100], [250, 160]], 
                    [[300, 100], [300, 120]], [[300, 120], [360, 120]], [[360, 100], [360, 120]], [[360, 100], [360, 120]], [[360, 100], [400, 100]], [[400, 100], [400, 160]], 
                    [[340, 120], [340, 160]], [[320, 160], [400, 160]], [[320, 160], [320, 280]], [[280, 190], [320, 190]], [[380, 160], [380, 200]], [[320, 200], [400, 200]], 
                    [[400, 200], [400, 280]], [[320, 240], [400, 240]], [[360, 240], [360, 280]], [[360, 280], [400, 280]], [[220, 160], [220, 220]], [[220, 160], [280, 160]], 
                    [[220, 220], [280, 220]], [[280, 160], [280, 220]]]

def graph_to_grid():    
    # the array of path is converted to a grid representation
    # one block is 10 Pixel wide
    # the grid is 31 = (400 - 100) / 10 Pixel wide and 19 = (280 - 100) / 10 Pixel high
    global Array_of_path
    global grid_representation
    grid_representation = []
    for i in range(19):
        grid_representation.append([])
        for _ in range(31):
            grid_representation[i].append(0)
    for path in Array_of_path:
        x_start = int((path[0][0] - 100) / 10)
        y_start = int((path[0][1] - 100) / 10)
        x_end = int((path[1][0] - 100) / 10)
        y_end = int((path[1][1] - 100) / 10)
        grid_representation[y_start][x_start] = 1
        grid_representation[y_end][x_end] = 1
        if x_start == x_end:
            for i in range(min(y_start, y_end), max(y_start, y_end) + 1):
                grid_representation[i][x_start] = 1
        if y_start == y_end:
            for i in range(min(x_start, x_end), max(x_start, x_end) + 1):
                grid_representation[y_start][i] = 1

# Array, which defines where the detecting lights are
array_of_lights = [[160, 120], [140, 200], [180, 260], [250, 240], [320, 260], [340, 140], [360, 200]]
make_graph()
graph_to_grid()
starting_place_of_hider = (250, 280)
starting_place_of_seekers = [(200, 100), (250, 100), (300, 100)]



class Hider(object):
    def __init__(self, array_of_seekers, parent_index) -> None:
        self.x, self.y = deepcopy(starting_place_of_hider)
        self.radius = 10
        self.color = black
        self.current_path_index = 0
        self.normalize_input_layer = False
        self.input_layer = []
        self.array_of_seekers = array_of_seekers
        self.init_input_layer()
        self.update_input_layer()
        self.output_layer_size = 5
        self.step_of_game = 0
        # the output layer is a list of 5 values, with the expected reward for each action
        # the action are: 0 = stay, 1 = up, 2 = down, 3 = left, 4 = right
        # after the action is taken, the reward is calculated and the output layer is updated
        # with backpropagation and gradient descent the weights and biases are updated
        self.neuronal_network = make_net(len(self.input_layer), self.output_layer_size)
        self.last_3_places_over_game = []
        normalized_input = [(self.x - 100) / 400, (self.y - 100) / 280, self.step_of_game] if self.normalize_input_layer else [(self.x - 100) // 10, (self.y - 100) // 10, self.step_of_game]
        self.last_3_places_over_game.append(normalized_input)
        self.wrong_moves = 0
        self.wins = 0
        self.parent_index = parent_index
        self.sum_nearest_distance = 0

    def re_init(self):
        self.x, self.y = deepcopy(starting_place_of_hider)
        self.current_path_index = 0
        self.input_layer = []
        self.init_input_layer()
        self.last_3_places_over_game = []
        self.step_of_game = 0
        self.wrong_moves = 0

    def handle_network(self, should_collect_data, current_game_index, list_train_X_shared, list_train_y_shared):
        self.update_input_layer()
        output = self.neuronal_network(torch.tensor(self.input_layer, dtype=torch.float32))
        dist = 10
        move = torch.argmax(output)
        move_made = False
        array_of_moves = self.check_all_moves()
        if should_collect_data:
            list_train_X = np.frombuffer(list_train_X_shared.get_obj()).reshape((population_size, 4, fitness_bases_on_n_games, list_of_games[0].max_number_of_turns, len_input_layer))
            list_train_y = np.frombuffer(list_train_y_shared.get_obj()).reshape((population_size, 4, fitness_bases_on_n_games, list_of_games[0].max_number_of_turns, len_output_layer))
            list_train_X[self.parent_index][0][current_game_index][self.step_of_game] = self.input_layer
            train_y = [x * y for x, y in zip(output.tolist(), array_of_moves)]
            train_y[0] /= 2 # after a time the players would just stay still, so the reward for staying still is halved
            list_train_y[self.parent_index][0][current_game_index][self.step_of_game] = train_y
        arr_moves = ["stay", "up", "down", "left", "right"]
        while not move_made:
            if self.change_path(arr_moves[move]) or array_of_moves[move] == 1:
                # if move is 0, the hider stays still
                if move == 1:
                    self.y -= dist
                elif move == 2:
                    self.y += dist
                elif move == 3:
                    self.x -= dist
                elif move == 4:
                    self.x += dist
                move_made = True
            else:
                output[move] = 0
                move = torch.argmax(output)
                self.wrong_moves += 1
            if [self.x, self.y] in array_of_lights and move_made:
                if self.normalize_input_layer:
                    self.last_3_places_over_game.append([(self.x - 100) / 400, (self.y - 100) / 280, self.step_of_game])
                else:
                    self.last_3_places_over_game.append([(self.x - 100) // 10, (self.y - 100) // 10, self.step_of_game])
                if len(self.last_3_places_over_game) > 3:
                    self.last_3_places_over_game.pop(0)
        self.step_of_game += 1

    def check_all_moves(self):
        # check all moves the hider can make
        # this is used for reinforcement learning
        # the hider can make 5 moves
        # 0 = stay, 1 = up, 2 = down, 3 = left, 4 = right
        # if the player can go: stay, down, right, the output will be [1, 0, 1, 0, 1]
        moves = [1] # the player can always stay
        if Array_of_path[self.current_path_index][0][1] < self.y or self.change_path("up", change_path=False):
            moves.append(1)
        else:
            moves.append(0)
        if Array_of_path[self.current_path_index][1][1] > self.y or self.change_path("down", change_path=False):
            moves.append(1)
        else:
            moves.append(0)
        if Array_of_path[self.current_path_index][0][0] < self.x or self.change_path("left", change_path=False):
            moves.append(1)
        else:
            moves.append(0)
        if Array_of_path[self.current_path_index][1][0] > self.x or self.change_path("right", change_path=False):
            moves.append(1)
        else:
            moves.append(0)
        return moves

    def change_path(self, direction, change_path=True):
        for index, Element in enumerate(Array_of_path):
            if (index != self.current_path_index
                and (self.x == Element[0][0]
                    and self.y >= Element[0][1]
                    and self.y <= Element[1][1]
                    or self.y == Element[0][1]
                    and self.x >= Element[0][0]
                    and self.x <= Element[1][0])
                and (direction == "down"
                    and Array_of_path[index][1][1] > self.y
                    or direction == "left"
                    and Array_of_path[index][0][0] < self.x
                    or direction == "right"
                    and Array_of_path[index][1][0] > self.x
                    or direction == "up"
                    and Array_of_path[index][0][1] < self.y)):
                if change_path:
                    self.current_path_index = index
                return True
        return False

    def init_input_layer(self):
        # when using the index of the grid as an Input we can normlize them by dividing them by the max value of the grid
        # but this is not necessary good. Thus we define the "Normalized Input Index". If set to True we use the normalized Input
        # we initialize the input layer
        # the Hider sees all Seekers
        # Furthermore it know the state of the maze and its own position
        if self.normalize_input_layer:
            self.input_layer.append((self.x - 100) / 400)
            self.input_layer.append((self.y - 100) / 280)
            # first six neurons ar build like this:
            # 1. (x-coordinate of the Seeker 0 - 100) / max_x_coordinate; 2. (y-coordinate of the Seeker 0 - 100) / max_y_coordinate
            # 3. (x-coordinate of the Seeker 1 - 100) / max_x_coordinate; 4. (y-coordinate of the Seeker 1 - 100) / max_y_coordinate
            # 5. (x-coordinate of the Seeker 2 - 100) / max_x_coordinate; 6. (y-coordinate of the Seeker 2 - 100) / max_y_coordinate
            for seeker in self.array_of_seekers:
                self.input_layer.append((seeker.x - 100) / 400)
                self.input_layer.append((seeker.y - 100) / 280)
        
        else:
            # first two neurons are build are his x position index and his y position index. The index is caculated by first subtracting 100 and then deviding the x or y coordinate by 10
            self.input_layer.append((self.x - 100) // 10)
            self.input_layer.append((self.y - 100) // 10)
            # next six neurons are build as follows:
            # 1. x-coordinate index of the Seeker 0; 2. y-coordinate index of the Seeker 0
            # 3. x-coordinate index of the Seeker 1; 4. y-coordinate index of the Seeker 1
            # 5. x-coordinate index of the Seeker 2; 6. y-coordinate index of the Seeker 2
            for seeker in self.array_of_seekers:
                self.input_layer.append((seeker.x - 100) // 10)
                self.input_layer.append((seeker.y - 100) // 10)

        # the next neuron is the distance to the closest seeker
        self.input_layer.append(self.distance_to_closest_seeker() / 400)
        # the next 589 neurons are the flattend maze
        for row in grid_representation:
            for element in row:
                self.input_layer.append(element)
        # patch the input layer to the correct size with 0
        # input layer from seeker = 605
        # input layer from hider = 598
        for _ in range(7):
            self.input_layer.append(0)

    def update_input_layer(self):
        # when using the index of the grid as an Input we can normlize them by dividing them by the max value of the grid
        # but this is not necessary good. Thus we define the "Normalized Input Index". If set to True we use the normalized Input
        # we initialize the input layer
        # the Hider sees all Seekers
        # Furthermore it know the state of the maze and its own position
        if self.normalize_input_layer:
             # first two neurons are build are his x position - 100 / max_x_coordinate and his y position - 100 / max_y_coordinate
            self.input_layer[0] = (self.x - 100) / 400
            self.input_layer[1] = (self.y - 100) / 280
            # next six neurons are build as follows:
            # 1. (x-coordinate of the Seeker 0 - 100) / max_x_coordinate; 2. (y-coordinate of the Seeker 0 - 100) / max_y_coordinate
            # 3. (x-coordinate of the Seeker 1 - 100) / max_x_coordinate; 4. (y-coordinate of the Seeker 1 - 100) / max_y_coordinate
            # 5. (x-coordinate of the Seeker 2 - 100) / max_x_coordinate; 6. (y-coordinate of the Seeker 2 - 100) / max_y_coordinate
            for index, seeker in enumerate(self.array_of_seekers):
                self.input_layer[2 + 2 * index] = (seeker.x - 100) / 400
                self.input_layer[3 + 2 * index] = (seeker.y - 100) / 280
        
        else:
            # first two neurons are build are his x position index and his y position index. The index is caculated by first subtracting 100 and then deviding the x or y coordinate by 10
            self.input_layer[0] = (self.x - 100) // 10
            self.input_layer[1] = (self.y - 100) // 10
            # next six neurons are build as follows:
            # 1. x-coordinate index of the Seeker 0; 2. y-coordinate index of the Seeker 0
            # 3. x-coordinate index of the Seeker 1; 4. y-coordinate index of the Seeker 1
            # 5. x-coordinate index of the Seeker 2; 6. y-coordinate index of the Seeker 2
            for index, seeker in enumerate(self.array_of_seekers):
                self.input_layer[2 + 2 * index] = (seeker.x - 100) // 10
                self.input_layer[3 + 2 * index] = (seeker.y - 100) // 10
        # the next neuron is the distance to the closest seeker
        self.input_layer[11] = self.distance_to_closest_seeker() / 400

    def distance_to_closest_seeker(self):
        distances = [
            math.sqrt((self.x - seeker.x) ** 2 + (self.y - seeker.y) ** 2)
            for seeker in self.array_of_seekers]
        return min(distances)

class Seeker(object):
    def __init__(self, index, parent_index) -> None:
        self.x, self.y = deepcopy(starting_place_of_seekers[index])
        self.radius = 10
        self.color = red
        self.current_path_index = 1
        self.index = index
        self.normalize_input_layer = False
        self.output_layer_size = 5
        # the output layer is a list of 5 values, with the expected reward for each action
        # the action are: 0 = stay, 1 = up, 2 = down, 3 = left, 4 = right
        # after the action is taken, the reward is calculated and the output layer is updated
        # with backpropagation and gradient descent the weights and biases are updated
        self.step_of_game = 0
        self.wrong_moves = 0
        self.wins = 0
        self.parent_index = parent_index
        self.sum_nearest_distance = 0

    def re_init(self):
        self.x, self.y = deepcopy(starting_place_of_seekers[self.index])
        self.current_path_index = 1
        self.step_of_game = 0
        self.wrong_moves = 0

    def init_array_of_seeker(self, array_of_seekers):
        self.array_of_seekers = array_of_seekers
        self.init_input_layer()
        self.neuronal_network = make_net(len(self.input_layer), self.output_layer_size)

    def handle_network(self, hiding_player, should_collect_data, current_game_index, list_train_X_shared, list_train_y_shared):
        self.update_input_layer(hiding_player)
        output = self.neuronal_network(torch.tensor(self.input_layer, dtype=torch.float32))
        dist = 10
        move = torch.argmax(output)
        move_made = False
        array_of_moves = self.check_all_moves()
        if should_collect_data:
            list_train_X = np.frombuffer(list_train_X_shared.get_obj()).reshape((population_size, 4, fitness_bases_on_n_games, list_of_games[0].max_number_of_turns, len_input_layer))
            list_train_y = np.frombuffer(list_train_y_shared.get_obj()).reshape((population_size, 4, fitness_bases_on_n_games, list_of_games[0].max_number_of_turns, len_output_layer))
            list_train_X[self.parent_index][self.index + 1][current_game_index][self.step_of_game] = self.input_layer
            train_y = [x * y for x, y in zip(output.tolist(), array_of_moves)]
            train_y[0] /= 2 # after a time the players would just stay still, so the reward for staying still is halved
            list_train_y[self.parent_index][self.index + 1][current_game_index][self.step_of_game] = train_y
        arr_moves = ["stay", "up", "down", "left", "right"]
        while not move_made:
            if self.change_path(arr_moves[move]) or array_of_moves[move] == 1:
                if move == 1:
                    self.y -= dist
                elif move == 2:
                    self.y += dist
                elif move == 3:
                    self.x -= dist
                elif move == 4:
                    self.x += dist
                move_made = True
            else:
                output[move] = 0
                move = torch.argmax(output)
                self.wrong_moves += 1
        self.step_of_game += 1

    def check_all_moves(self):
        # check all moves the hider can make
        # this is used for reinforcement learning
        # the hider can make 5 moves
        # 0 = stay, 1 = up, 2 = down, 3 = left, 4 = right
        # if the player can go: stay, down, right, the output will be [1, 0, 1, 0, 1]
        moves = [1] # the player can always stay
        if Array_of_path[self.current_path_index][0][1] < self.y or self.change_path("up", change_path=False):
            moves.append(1)
        else:
            moves.append(0)
        if Array_of_path[self.current_path_index][1][1] > self.y or self.change_path("down", change_path=False):
            moves.append(1)
        else:
            moves.append(0)
        if Array_of_path[self.current_path_index][0][0] < self.x or self.change_path("left", change_path=False):
            moves.append(1)
        else:
            moves.append(0)
        if Array_of_path[self.current_path_index][1][0] > self.x or self.change_path("right", change_path=False):
            moves.append(1)
        else:
            moves.append(0)
        return moves

    def change_path(self, direction, change_path=True):
        for index, Element in enumerate(Array_of_path):
            if (index != self.current_path_index
                and (self.x == Element[0][0]
                    and self.y >= Element[0][1]
                    and self.y <= Element[1][1]
                    or self.y == Element[0][1]
                    and self.x >= Element[0][0]
                    and self.x <= Element[1][0])
                and (direction == "down"
                    and Array_of_path[index][1][1] > self.y
                    or direction == "left"
                    and Array_of_path[index][0][0] < self.x
                    or direction == "right"
                    and Array_of_path[index][1][0] > self.x
                    or direction == "up"
                    and Array_of_path[index][0][1] < self.y)):
                if change_path:
                    self.current_path_index = index
                return True
        return False

    def init_input_layer(self):
        # as statet here:
        # "Emergence of Communication and Coordination in Multi-Agent Systems with Independent Neural Networks" by Kenji Doya, Kimitoshi Yamazaki, and Seiryo Aonuma: This book discusses the use of independent NNs for agents in a multi-agent system, and the benefits of using independent NNs over a shared NN. The authors show that independent NNs can enable the agents to adapt more quickly and effectively to changing circumstances, and to achieve better performance in tasks such as reinforcement learning and multi-agent coordination.
        # Each Seeker has its own NN
        # when using the index of the grid as an Input we can normlize them by dividing them by the max value of the grid
        # but this is not necessary good. Thus we define the "Normalized Input Index". If set to True we use the normalized Input
        # we initialize the input layer
        # the Hider sees all Seekers
        # Furthermore it know the state of the maze and its own position


        self.input_layer = []
        for index in range(3):
            if self.normalize_input_layer:
                # first six neurons are build as follows:
                # 1. (x-coordinate of the Seeker 0 - 100) / max_x_coordinate; 2. (y-coordinate of the Seeker 0 - 100) / max_y_coordinate
                # 3. (x-coordinate of the Seeker 1 - 100) / max_x_coordinate; 4. (y-coordinate of the Seeker 1 - 100) / max_y_coordinate
                # 5. (x-coordinate of the Seeker 2 - 100) / max_x_coordinate; 6. (y-coordinate of the Seeker 2 - 100) / max_y_coordinate
                self.input_layer.append((self.array_of_seekers[index].x - 100) / 400)
                self.input_layer.append((self.array_of_seekers[index].y - 100) / 280)


            else:
                # first six neurons are build as follows:
                # 1. x-coordinate index of the Seeker 0; 2. y-coordinate index of the Seeker 0
                # 3. x-coordinate index of the Seeker 1; 4. y-coordinate index of the Seeker 1
                # 5. x-coordinate index of the Seeker 2; 6. y-coordinate index of the Seeker 2
                self.input_layer.append(self.array_of_seekers[index].x // 10)
                self.input_layer.append(self.array_of_seekers[index].y // 10)

        # the next 9 neurons are the last 3 know places of the hider.
        # the first 3 neurons are the x and y coordinates and the number of moves ago the hider was there
        # the next pair of 3-neurons are the same for the second last and third last known place of the hider
        # at the beginning the hider has no known places and thus the neurons are set to 0
        self.input_layer.extend(0 for _ in range(9))
        # the next neuron is the step of the game
        self.input_layer.append(0)
        # the next 589 neurons are the flattend maze
        for row in grid_representation:
            self.input_layer.extend(iter(row))

    
    def update_input_layer(self, hiding_player):
        if self.normalize_input_layer:
            for index, seeker in enumerate(self.array_of_seekers):
                self.input_layer[2 * index] = (seeker.x - 100) / 400
                self.input_layer[2 * index + 1] = (seeker.y - 100) / 280
            for index, places in enumerate(hiding_player.last_3_places_over_game):
                self.input_layer[6 + 3 * index] = places[0] / 400
                self.input_layer[6 + 3 * index + 1] = places[1] / 280
                self.input_layer[6 + 3 * index + 2] = places[2]

        else:
            for index, seeker in enumerate(self.array_of_seekers):
                self.input_layer[2 * index] = (seeker.x - 100) // 10
                self.input_layer[2 * index + 1] = (seeker.y - 100) // 10
            for index, places in enumerate(hiding_player.last_3_places_over_game):
                self.input_layer[6 + 3 * index] = places[0] // 10
                self.input_layer[6 + 3 * index + 1] = places[1] // 10
                self.input_layer[6 + 3 * index + 2] = places[2]
        
        self.input_layer[15] = self.step_of_game

class Game(object):
    # this is a class that will handle one game
    def __init__(self, index):
        self.width = 800
        self.height = 600
        self.array_of_seekers = [Seeker(0, index), Seeker(1, index), Seeker(2, index)]
        for seeker in self.array_of_seekers:
            seeker.init_array_of_seeker(self.array_of_seekers)
        self.hiding_player = Hider(self.array_of_seekers, index)
        self.running = True
        self.number_of_turns = 0
        pygame.init()
        self.max_number_of_turns = 150
        self.index = index
    
    def re_init(self):
        # after a game is finished we re-initialize the game
        # we also have to do this because of the evolution
        for seeker in self.array_of_seekers:
            seeker.re_init()
        self.hiding_player.re_init()
        self.running = True
        self.number_of_turns = 0

    def play(self, should_collect_data, current_game_index, list_train_X, list_train_y, list_of_distance):
        global list_of_wins
        # play the game
        for seeker in self.array_of_seekers:
            seeker.handle_network(self.hiding_player, should_collect_data, current_game_index, list_train_X, list_train_y)
        if self.collision():
            list_of_wins[self.index] = -1
            return False
        self.hiding_player.handle_network(should_collect_data, current_game_index, list_train_X, list_train_y)
        if self.collision():
            list_of_wins[self.index] = -1
            return False
        self.number_of_turns += 1
        if self.number_of_turns == self.max_number_of_turns:
            self.running = False
            list_of_wins[self.index] = 1
            return False
        self.check_distance(list_of_distance)
        return True

    def collision(self):
        for seeker in self.array_of_seekers:
            if (self.hiding_player.x + 2 * self.hiding_player.radius >= seeker.x and
                self.hiding_player.x - 2 * self.hiding_player.radius <= seeker.x and
                self.hiding_player.y == seeker.y or
                self.hiding_player.y + 2 * self.hiding_player.radius >= seeker.y and
                self.hiding_player.y - 2 * self.hiding_player.radius <= seeker.y and
                self.hiding_player.x == seeker.x):
                self.running = False
                return True
        return False

    def check_distance(self, list_of_distance):
        distance_arr = np.frombuffer(list_of_distance.get_obj()).reshape(population_size, 4)
        for seeker_index, seeker in enumerate(self.array_of_seekers):
            # we take the manhatten distance, which is computational much faster than the real distance and is still a good indicator for the distance
            if distance_arr[self.index][seeker_index + 1] > ((seeker.x - self.hiding_player.x) // 10)**2 + ((seeker.y - self.hiding_player.y) // 10)**2:
                distance_arr[self.index][seeker_index + 1] = ((seeker.x - self.hiding_player.x) // 10)**2 + ((seeker.y - self.hiding_player.y) // 10)**2
            if distance_arr[self.index][0] > ((seeker.x - self.hiding_player.x) // 10)**2 + ((seeker.y - self.hiding_player.y) // 10)**2:
                distance_arr[self.index][0] = ((seeker.x - self.hiding_player.x) // 10)**2 + ((seeker.y - self.hiding_player.y) // 10)**2



def draw_path(screen, Array_of_path):
    for path in Array_of_path:
        pygame.draw.line(screen, black, path[0], path[1], 3)

def wrong_move(game):
    wrong_move = sum(seeker.wrong_moves for seeker in game.array_of_seekers)
    wrong_move += game.hiding_player.wrong_moves
    return wrong_move / (game.number_of_turns * 4)

def rotate_players_in_games(list_of_games):
    # to evalute the games we have to shuffle the hider. For simplicity we don't shuffle the seekers
    first_hider = list_of_games[0].hiding_player
    for index in range(len(list_of_games) - 1):
        list_of_games[index].hiding_player = list_of_games[index + 1].hiding_player
        list_of_games[index].hiding_player.parent_index = index
    list_of_games[-1].hiding_player = first_hider

def event_handling():
    for event in pygame.event.get():
        global list_of_games
        if event.type == pygame.QUIT:
            with open('readme.txt', 'w') as f:
                hiding_NN = flatten_list_of_NN([game.hiding_player.neuronal_network for game in list_of_games])
                searcher_1_NN = flatten_list_of_NN([game.array_of_seekers[0].neuronal_network for game in list_of_games])
                searcher_2_NN = flatten_list_of_NN([game.array_of_seekers[1].neuronal_network for game in list_of_games])
                searcher_3_NN = flatten_list_of_NN([game.array_of_seekers[2].neuronal_network for game in list_of_games])
                f.write(f'hiding_NN = {str(hiding_NN.tolist())}seacher_1_NN = {str(searcher_1_NN.tolist())}seacher_2_NN = {str(searcher_2_NN.tolist())}seacher_3_NN = {str(searcher_3_NN.tolist())}')
            pygame.quit()
            quit()

def flatten_list_of_NN(list_of_NN):
    list_of_NN_flattened = []
    for index, NN in enumerate(list_of_NN):
        helplist = [param.data.flatten().tolist() for name, param in NN.named_parameters() if param.requires_grad]
        helplist[0][0] = index
        list_of_NN_flattened.append(deepcopy(list(itertools.chain(*helplist))))
    return torch.tensor(list_of_NN_flattened)

def create_problem_and_searcher(type_of_player):
    global run_with_NN
    net_structure =  make_net(605, 5)
    if type_of_player == 'hider':
        problem = NEProblem("max", net_structure, device='cpu', eval_dtype=torch.float32)
        searcher = SimpleGA(problem, popsize=population_size, num_elites=int(0.2 * population_size), num_parents=int(0.2 * population_size), mutation_power=0.1, nn_values=flatten_list_of_NN([game.hiding_player.neuronal_network for game in list_of_games]))
        searcher.population.set_evals(torch.tensor([game.hiding_player.wins + game.hiding_player.sum_nearest_distance / 500 for game in list_of_games]))
    elif type_of_player == 'seeker':
        problem = [NEProblem("max", net_structure, device='cpu', eval_dtype=torch.float32) for _ in range(3)]
        searcher = [SimpleGA(probl, popsize=population_size, num_elites=int(0.2 * population_size), num_parents=int(0.2 * population_size), mutation_power=0.1, nn_values=flatten_list_of_NN([game.array_of_seekers[index].neuronal_network for game in list_of_games])) for index, probl in enumerate(problem)]
        for i in range(3):
            searcher[i].population.set_evals(torch.tensor([game.array_of_seekers[i].wins - game.array_of_seekers[i].sum_nearest_distance / 500 for game in list_of_games]))
    return problem, searcher

def train(list_of_hiders, list_of_array_of_seekers):
    global list_train_X, list_train_y
    train_X_all = np.frombuffer(list_train_X.get_obj()).reshape((population_size, 4, fitness_bases_on_n_games, list_of_games[0].max_number_of_turns, len_input_layer))
    train_y_all = np.frombuffer(list_train_y.get_obj()).reshape((population_size, 4, fitness_bases_on_n_games, list_of_games[0].max_number_of_turns, len_output_layer))
    train_X_all = np.array(train_X_all, dtype=np.float32)
    train_y_all = np.array(train_y_all, dtype=np.float32)
    for hider in list_of_hiders:
        train_X = train_X_all[hider.parent_index, 0, :, :, :]
        train_y = train_y_all[hider.parent_index, 0, :, :, :]
        train_individual(hider, train_X, train_y)
    for array in list_of_array_of_seekers:
        for seeker_index, seeker in enumerate(array):
            train_X = train_X_all[seeker.parent_index, seeker_index + 1, :, :, :]
            train_y = train_y_all[seeker.parent_index, seeker_index + 1, :, :, :]
            train_individual(seeker, train_X, train_y)
    list_train_X = Array(ct.c_float, population_size * 4 * fitness_bases_on_n_games * list_of_games[0].max_number_of_turns * len_input_layer * 2)
    list_train_y = Array(ct.c_float, population_size * 4 * fitness_bases_on_n_games * list_of_games[0].max_number_of_turns * len_output_layer * 2)
        

def train_individual(player, train_X, train_y):
    train_X, train_y = zip(*random.sample(list(zip(train_X, train_y)), len(train_X)))
    train_X, train_y = np.array(train_X), np.array(train_y)
    player.neuronal_network.train(training_epochs,torch.tensor(train_X),torch.tensor(train_y),list_of_games[0].max_number_of_turns * 2)

def play_game(game, should_collect_data, current_game_index, list_train_X, list_train_y, list_of_distance):
    while game.play(should_collect_data, current_game_index, list_train_X, list_train_y, list_of_distance):
        pass
    game.running = True

def run_the_games(list_of_games, should_collect_data, current_game_index, list_train_X, list_train_y, list_of_distance):
    processes = []
    for game in list_of_games:
        process = Process(target=play_game, args=(game, should_collect_data, current_game_index, list_train_X, list_train_y, list_of_distance))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    # we get a list of wins and loosesnetwork_out of the hider and seekers in list_of_wins
    # if the hider wins, the element at index from the game is 1, else -1
    distances = np.frombuffer(list_of_distance.get_obj()).reshape(population_size, 4)
    for index, game in enumerate(list_of_games):
        if list_of_wins[index] == 1:
            game.hiding_player.wins += 1
        else:
            for seeker in game.array_of_seekers:
                seeker.wins += 1
        game.hiding_player.sum_nearest_distance += distances[game.hiding_player.parent_index][0]
        for seeker_index, seeker in enumerate(game.array_of_seekers):
            seeker.sum_nearest_distance += distances[seeker.parent_index][seeker_index + 1]
    rotate_players_in_games(list_of_games)
    event_handling()

def reset_wins_and_distance():
    distances = np.frombuffer(list_of_distance.get_obj()).reshape(population_size, 4)
    for game in list_of_games:
        game.hiding_player.wins = 0
        for seeker in game.array_of_seekers:
            seeker.wins = 0
    for game_index, player_index in itertools.product(range(population_size), range(4)):
        distances[game_index][player_index] = 1000 # this is the distance, and because the distance is always smaller than 1000, we can use this as a default value

def make_net(input_size, output_size):
    return Net([input_size, 46, 32, 32, 32, 16, 16, 16, 16, 16, output_size], ["leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "softmax"], 0.01, 0.9)

population_size = 50
list_of_games = [Game(i) for i in range(population_size)]
fitness_bases_on_n_games = 10 # how many games should be played to evaluate the fitness of a player
train_every_n_games = 5 # how many games should be played before the players are trained with backpropagation
training_epochs = 5 # how many epochs per training
cycle = 1
len_input_layer = 605 # length from input layer from seeker and hider are 605
len_output_layer = 5 # 5 possible outputs
list_of_wins = Manager().list([0 for _ in range(population_size)]) # multithreading does some wired shit, so we save the wins of the hider in this list, sorted after the games, like list_of_games
list_of_distance = Array(ct.c_float, population_size * 4 * 2) # for mulithreading we need to use this array, which is a shared memory array.
reset_wins_and_distance()
list_train_X = Array(ct.c_float, population_size * 4 * fitness_bases_on_n_games * list_of_games[0].max_number_of_turns * len_input_layer * 2)
list_train_y = Array(ct.c_float, population_size * 4 * fitness_bases_on_n_games * list_of_games[0].max_number_of_turns * len_output_layer * 2)

while True:
    screen = pygame.display.set_mode((200, 300))
    pygame.display.set_caption(f'Mini-Game Wii-U, Game: {str(cycle)}')
    pygame.init()
    # every seeker and every hider plays 10 games
    for current_game_index in range(fitness_bases_on_n_games):
        # let this specific set of games play
        for game in list_of_games:
            game.re_init()
        should_collect_data = current_game_index % train_every_n_games == 0
        run_the_games(list_of_games, should_collect_data, current_game_index, list_train_X, list_train_y, list_of_distance)
    distances = np.frombuffer(list_of_distance.get_obj()).reshape(population_size, 4)
    # the underlying problem with training is that it takes compared to the evoulationary process a lot of time
    # to solve this problem we have mulitples solutions:
    # 1. We only train every 10th cycle
    # 2. We only train the best 15% of the players
    # 3. We also only collect the train data every 10th cycle of the best 15% of the players
    # the last step is very important, because to mitigate race conditions, we have to collect the train data wiht a Array from the multiprocessing library
    # I run the program with mulitple settings:
    # without mulitthreading: 547
    # with mulitthreading: 86

    if cycle % train_every_n_games == 0:
        # train the players with backpropagation, only the best 15 % of the players are trained
        # collect all the hider and array_of_seekers from the games and sort them after wins
        list_of_hiders = sorted([game.hiding_player for game in list_of_games], key = lambda hider : hider.wins, reverse = True)
        list_of_array_of_seekers = sorted([game.array_of_seekers for game in list_of_games], key = lambda array : array[0].wins, reverse = True)
        list_of_hiders = list_of_hiders[:int(len(list_of_hiders) * 0.15)]
        list_of_array_of_seekers = list_of_array_of_seekers[:int(len(list_of_array_of_seekers) * 0.15)]
        train(list_of_hiders, list_of_array_of_seekers)
    
    # Create the problem
    problem_hider, searcher_hider = create_problem_and_searcher('hider')
    problem_seekers, searcher_seekers = create_problem_and_searcher('seeker')
    # run the Evolutionary Algorithm
    # only once, because before the next run, we have to reavaluate the fitness function with letting the players play
    searcher_hider.run(1)
    for searcher in searcher_seekers:
        searcher.run(1)
    # reset the wins and the nearest distance
    reset_wins_and_distance()
    # reassign the NNs to the players
    for index, game in enumerate(list_of_games):
        game.hiding_player.neuronal_network.set_weights_and_biases(searcher_hider.population[index])
        for i in range(3):
            game.array_of_seekers[i].neuronal_network.set_weights_and_biases(searcher_seekers[i].population[index])
    cycle += 1
    print(cycle)
