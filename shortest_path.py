import numpy as np
import random
import math
from time import sleep
from PIL import Image, ImageDraw

import cv2
from skimage import morphology
import networkx as nx


def prog(img, b=64):
    return (img >= b).astype(np.uint8) * 255

def get_reachable_image(input_image, start_point):
    image = np.where(input_image > 0, 1, 0)

    reachable_image = np.zeros(image.shape)
    reachable_image[start_point[1]][start_point[0]] = 1

    image_size_x = image.shape[1]
    image_size_y = image.shape[0]

    reachable_stack = [start_point]
    while len(reachable_stack) > 0:
        current_point = reachable_stack.pop()

        min_x = max(0, current_point[0] - 1)
        max_x = min(image_size_x - 1, current_point[0] + 1)
        min_y = max(0, current_point[1] - 1)
        max_y = min(image_size_y - 1, current_point[1] + 1)

        neighbors = [(x, y) for y in range(min_y, max_y + 1) for x in range(min_x, max_x + 1)]

        for point in neighbors:
            x = point[0]
            y = point[1]
            if image[y][x] == 1 and reachable_image[y][x] == 0:
                reachable_image[y][x] = 1
                reachable_stack.append(point)

    return reachable_image

def get_weighted_image(input_image):
    weighted_image = input_image
    eroded_image = input_image

    while True:
        eroded_image = cv2.erode(eroded_image, np.ones((3, 3)))

        if eroded_image.max() == 0:
            break

        weighted_image = weighted_image + eroded_image

    return weighted_image

def get_skeletonized_image(input_image,weighted_image):
    skeleton = morphology.skeletonize(input_image)
    skeletonized_image = weighted_image
    skeletonized_image[~skeleton] = 0
    return skeletonized_image

def get_reachable_points(skeletonized_image):
    reachable_points = set()
    for y in range(skeletonized_image.shape[0]):
        for x in range(skeletonized_image.shape[1]):
            if skeletonized_image[y, x] > 0:
                reachable_points.add((x, y))
    return reachable_points

def get_intersection_points(reachable_points, start_point, target):
    intersection_points = set()
    for point in reachable_points:
        neighbors = [(x, y) for x in range(point[0] - 1, point[0] + 2) for y in range(point[1] - 1, point[1] + 2)]
        neighbors = [i for i in neighbors if i in reachable_points and i != point]
        if len(neighbors) > 2:
            intersection_points.add(point)

    if start_point not in intersection_points:
        intersection_points.add(start_point)

    if target not in intersection_points:
        intersection_points.add(target)

    return intersection_points

def get_paths(reachable_points, intersection_points):
    available_points = reachable_points.copy()
    paths = []

    for intersection_point in intersection_points:
        neighbors = [(x, y) for x in range(intersection_point[0] - 1, intersection_point[0] + 2) for y in range(intersection_point[1] - 1, intersection_point[1] + 2)]
        neighbors = [i for i in neighbors if i in available_points and i != intersection_point]

        # Intersection next to another intersection
        for end in [i for i in neighbors if i in intersection_points]:
            # Avoid path duplication
            if [end, intersection_point] not in paths:
                paths.append([intersection_point, end])

        path_beginings = [i for i in neighbors if i not in intersection_points]
        for path_begining in path_beginings:
            path_point = path_begining
            path = [intersection_point, path_point]
            last_point = intersection_point

            while True:
                neighbors = [(x, y) for x in range(path_point[0] - 1, path_point[0] + 2) for y in range(path_point[1] - 1, path_point[1] + 2)]
                neighbors = [i for i in neighbors if i != path_point and i != last_point and i in available_points]

                if len(neighbors) == 0:
                    # No append to avoid some dead ends
                    break

                # No more than 1 next point possible
                last_point = path_point
                path_point = neighbors[0]
                path.append(path_point)

                if path_point in intersection_points:
                    paths.append(path)
                    break

            # Avoid path duplication
            for i in range(1, len(path) - 1):
                available_points.remove(path[i])

    return paths

def get_graph(skeletonized_image, paths):
    vertices = set()
    edges = set()
    vertices_costs = {}
    edges_costs = {}
    path_by_edge = {}

    for path in paths:
        begin = path[0]
        if begin not in vertices:
            vertices.add(begin)
            vertices_costs[begin] = 1 / skeletonized_image[begin[1], begin[0]]

        end = path[-1]
        if end not in vertices:
            vertices.add(end)
            vertices_costs[end] = 1 / skeletonized_image[end[1], end[0]]

        edge = ((begin, end))
        edges.add(edge)

        cost = 0
        for i in range(1, len(path) - 1):
            cost += 1 / skeletonized_image[path[i][1], path[i][0]]

        edges_costs[edge] = cost

        path_by_edge[edge] = path

        edges_by_vertice = {}
        for edge in edges:
            end_points = [edge[0], edge[1]]
            for end_point in end_points:
                if end_point not in edges_by_vertice.keys():
                    edges_by_vertice[end_point] = []

                edges_by_vertice[end_point].append(edge)

    return {
        "vertices": vertices,
        "edges": edges,
        "vertices_costs": vertices_costs,
        "edges_costs": edges_costs,
        "path_by_edge": path_by_edge,
        "edges_by_vertice": edges_by_vertice
    }

class Ant:
    def __init__(self, start_point, target, graph, pheromone_by_edge, peheromone_strength, random_respawn_probability):
        self.vertices = graph["vertices"]
        self.edges = graph["edges"]
        self.vertices_costs = graph["vertices_costs"]
        self.edges_costs = graph["edges_costs"]
        self.edges_by_vertice = graph["edges_by_vertice"]

        self.start_point = start_point
        self.target = target

        self.random_respawn_probability = random_respawn_probability

        self.respawn()

        self.pheromone_by_edge = pheromone_by_edge
        self.pheromone_strength = peheromone_strength

        self.visited_edges = set()
        self.moves_cost = 0

    def move(self):
        moves = self.edges_by_vertice[self.position]
        moves = [i for i in moves if i not in self.visited_edges]
        if len(moves) == 0:
            # Going in a loop => die
            self.respawn()
            return

        move_probabilities = [1 + self.pheromone_by_edge[i] for i in moves]
        probabilities_sum = sum(move_probabilities)
        move_probabilities = [i / probabilities_sum for i in move_probabilities]
        move = random.choices(moves, move_probabilities)[0]

        self.position = move[0] if move[0] != self.position else move[1]
        self.visited_edges.add(move)
        self.moves_cost += self.edges_costs[move] + self.vertices_costs[self.position]

        if self.position == self.target:
            for edge in self.visited_edges:
                self.pheromone_by_edge[edge] += self.pheromone_strength / math.log(self.moves_cost)

            self.respawn()


    def respawn(self):
        self.position = random.choice([i for i in self.vertices]) if random.randrange(int(100 / self.random_respawn_probability)) < 100 else self.start_point
        self.visited_edges = set()
        self.moves_cost = 0


def find_shortest_path(img, x1, y1, x2, y2):
    sleep(1)
    X = np.linspace(0, 1, 64)
    Y = np.linspace(0, 1, 64)
    r = 1.5
    v1 = sum([sum([1 for x in X if (x-r)**2+(y-r)**2 <= r*r]) for y in Y]) / 64**2
    v2 = sum([sum([1 for x in X if (x-r)**2+y**2 <= r*r]) for y in Y]) / 64**2


    mask = np.array([[v1, v2, v1], [v2, 1.0, v2], [v1, v2, v1]])

    for i in range(4):
        mask = mask * mask
        s = sum(sum(mask)) - mask[1, 1]
        mask[1, 1] = s
        mask = mask / s

    mask = -mask
    mask[1, 1] = -mask[1, 1]

    gray = img[::, ::, 0]
    gray = cv2.GaussianBlur(gray,(5,5), 0)
    gray01 = gray.astype(float) / 255
    dst = gray.astype(float) / 255

    dst = cv2.filter2D(dst, -1, mask)
    _min = dst.min()
    _max = dst.max()
    dst = (dst-_min)/(_max-_min)

    dst = dst**2

#################################################################
    input_image = prog(dst*255, 55)
    start_point = (int(x1), int(y1))
    target = (int(x2), int(y2))
    reachable_image = get_reachable_image(input_image, start_point)
    weighted_image = get_weighted_image(reachable_image)
    skeletonized_image = get_skeletonized_image(reachable_image,weighted_image)
    reachable_points = get_reachable_points(skeletonized_image)
    
    intersection_points = get_intersection_points(reachable_points, start_point, target)


    paths = get_paths(reachable_points, intersection_points)

    graph = get_graph(skeletonized_image, paths)

    pheromone_by_edge = {}
    for edge in graph["edges"]:
        pheromone_by_edge[edge] = 0

    for iteration in range(10):
        evaporation_rate = math.pow(math.e, -0.01 * iteration)

        num_ants = 500
        pheromone_strength = 100
        ants = [Ant(start_point, target, graph, pheromone_by_edge, pheromone_strength, 2 / max(1, iteration)) for _ in range(num_ants)]

        for _ in range(200):
            for ant in ants:
                ant.move()

            for edge in pheromone_by_edge.keys():
                pheromone_by_edge[edge] *= evaporation_rate

    graph_nx  = nx.Graph()

    for vertices, phero in pheromone_by_edge.items():
        graph_nx.add_edge(vertices[0], vertices[1], weight=10-phero)

    shortest_path = nx.shortest_path(graph_nx, source=start_point, target=target)
    #print(type(x1), x2, y1, y2)
    #print(shortest_path)
    coordinates = [(int(x1), int(y1)), (307, 199), (350, 250), (400, 320),
                   (450, 380), (512, 429), (int(x2), int(y2))]
    img_with_path = draw_shortest_path(img, shortest_path)
    return img_with_path


def draw_shortest_path(img, coordinates):
    img_copy = Image.fromarray(img)
    draw = ImageDraw.Draw(img_copy)
    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)
    return np.array(img_copy)
