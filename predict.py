from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
from model1 import DirectionalGraphNeuralNetwork
from Delaunay import build_graph_with_master_node, draw_graph_on_image, extract_facial_landmarks
import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2

class_names_vietnamese = ['Tức giận', 'Khinh thường', 'Ghê tởm', 'Sợ hãi',
                          'Hạnh phúc', 'Buồn bã', 'Ngạc nhiên']

image = cv2.imread(
    'images_test/3.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Load the model architecture
num_node_features = 1  # Replace with the actual number of node features
num_edge_features = 1  # Replace with the actual number of edge features
num_classes = 7  # Replace with the actual number of classes

loaded_model = DirectionalGraphNeuralNetwork(
    num_node_features, num_edge_features, num_classes)
# Load the state dictionary
loaded_model.load_state_dict(torch.load(
    'save_models/model5.pth', map_location=torch.device('cpu')))
# Set the model in evaluation mode
loaded_model.eval()
facial_landmarks = np.array(extract_facial_landmarks(image))
if len(facial_landmarks) != 0:

    plt.imshow(image)
    plt.title(f"Facial Landmarks and Predicted Class")

    for face_idx in range(len(facial_landmarks)):
        landmarks = facial_landmarks[face_idx]
        landmarks = np.reshape(landmarks, (1, 51, 2))
        G = build_graph_with_master_node(landmarks)
        node_feature = torch.tensor(
            G.nodes, dtype=torch.float32).view(1, -1).t()
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        edge_weight = torch.tensor([G[u][v]['weight']
                                   for u, v in G.edges], dtype=torch.float32)

        with torch.no_grad():
            output = loaded_model(node_feature, edge_index,
                                  edge_weight, batch=None)

        # Áp dụng softmax để chuyển đổi thành xác suất
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output).item()
        predicted_class_name = class_names_vietnamese[predicted_class]

        center_x = int(landmarks[0, 19, 0])
        center_y = int(landmarks[0, 19, 1])

        plt.text(center_x, center_y + face_idx * 50,
                 f"Face {face_idx} : {predicted_class_name}",
                 color="red", fontsize=12, weight='bold')

        # In ra xác suất của từng lớp dưới dạng phần trăm
        print(f"Face {face_idx}: [", end="")
        for i in range(num_classes):
            print(f'{class_names_vietnamese[i]}: {probs[0][i] * 100:.2f}%',
                  end=", " if i < num_classes - 1 else "")
        print("]")
    plt.show()

else:
    print('None face are detected')
