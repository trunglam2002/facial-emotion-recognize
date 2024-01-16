from math import sqrt
import dlib
import cv2
from matplotlib.colors import to_rgb
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import torch
from torch_geometric.utils.convert import from_networkx
from transform import apply_gaussian_noise, apply_random_rotation, apply_random_flip


def extract_facial_landmarks(image):
    # Load the facial landmarks predictor
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    # Load an example image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    detector = dlib.get_frontal_face_detector()
    dets = detector(gray, 0)
    facial_landmarks = []
    for i, rect in enumerate(dets, 1):
        # Draw rectangle around the face
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        landmark = predictor(gray, rect)
        landmarks_xy = np.array(
            [(landmark.part(j).x, landmark.part(j).y) for j in range(17, 68)]) * sqrt(310*310+311*311) / sqrt(w*w+h*h)
        facial_landmarks.append(landmarks_xy)

    # Return an empty list if no landmarks are found
    return facial_landmarks if facial_landmarks else []


def build_graph_with_master_node(facial_landmarks):
    G = nx.Graph()

    # Adding facial landmarks as nodes with sequential numbering from 0 to 50
    for i, landmark in enumerate(facial_landmarks):
        for j, point in enumerate(landmark):
            node_position = tuple(point)
            G.add_node(i * 51 + j, color='blue')  # Thêm index
            G.nodes[i * 51 + j]['pos'] = node_position

    edges = []
    edge_weights = []

    # Construct edges using Delaunay method
    for i, landmark in enumerate(facial_landmarks):
        tri = Delaunay(landmark)
        for simplex in tri.simplices:
            node_indices = [
                i * 51 + simplex[0], i * 51 + simplex[1], i * 51 + simplex[2]]

            # Check if the edge involves the master node
            edges.extend([(node_indices[0], node_indices[1]), (node_indices[1], node_indices[2]),
                          (node_indices[2], node_indices[0])])

            # Update the logic for adding edges
            weight1 = np.linalg.norm(
                np.array(G.nodes[node_indices[0]]['pos']) - np.array(G.nodes[node_indices[1]]['pos']))
            weight2 = np.linalg.norm(
                np.array(G.nodes[node_indices[1]]['pos']) - np.array(G.nodes[node_indices[2]]['pos']))
            weight3 = np.linalg.norm(
                np.array(G.nodes[node_indices[2]]['pos']) - np.array(G.nodes[node_indices[0]]['pos']))

            # Update the logic for adding edges
            G.add_edge(node_indices[0], node_indices[1],
                       weight=weight1, color='blue')
            G.add_edge(node_indices[1], node_indices[2],
                       weight=weight2, color='blue')
            G.add_edge(node_indices[2], node_indices[0],
                       weight=weight3, color='blue')

            edge_weights.extend([weight1, weight2, weight3])

    # Adding the master node at index 51 (without pos)
    master_node_index = 51
    master_node = tuple(np.mean(facial_landmarks[:, 13, :], axis=0))
    # Set color of master node to red
    G.add_node(master_node_index, color='red')
    G.nodes[master_node_index]['pos'] = master_node

    for node_index in range(51):  # Connect master node to other nodes
        edges.extend([(master_node_index, node_index)])
        weight = np.linalg.norm(
            np.array(G.nodes[master_node_index]['pos']) - np.array(G.nodes[node_index]['pos']))
        edge_weights.extend([weight])
        G.add_edge(master_node_index, node_index, color='red', weight=weight)

    G.x = torch.tensor(facial_landmarks, dtype=torch.float64).view(2, -1)

    return G


def draw_graph_on_image(G, image):
    for edge in G.edges():
        node1 = G.nodes[edge[0]]
        node2 = G.nodes[edge[1]]

        # Màu mặc định là white nếu không có thuộc tính 'color'
        color = G[edge[0]][edge[1]].get('color', 'white')
        color = tuple(int(255 * x) for x in to_rgb(color))

        cv2.line(image, (int(node1['pos'][0]), int(node1['pos'][1])),
                 (int(node2['pos'][0]), int(node2['pos'][1])), color, 1)  # Độ dày là 5

    plt.imshow(image)
    plt.axis('equal')
    plt.show()


def main():
    image_path = 'images_test/tplam9.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np = np.array(image)

    facial_landmarks = np.array(extract_facial_landmarks(image))
    # print(facial_landmarks.shape)
    # noisy_image, noisy_landmarks = apply_random_rotation(
    #     image_np, facial_landmarks)
    G = build_graph_with_master_node(facial_landmarks)

    draw_graph_on_image(G, image)


if __name__ == "__main__":
    main()
