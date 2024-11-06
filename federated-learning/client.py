import grpc
import numpy as np
import tensorflow as tf
import argparse
from federated_model_pb2 import ModelUpdateRequest, ModelStatsRequest, ClientContributionRequest, PredictionRequest
from federated_model_pb2_grpc import FederatedLearningStub

class FederatedClient:
    def __init__(self, client_id, server_address='localhost:50051'):
        self.client_id = client_id
        self.channel = grpc.insecure_channel(server_address)
        self.stub = FederatedLearningStub(self.channel)
        self.model = self.build_local_model()
        
    def build_local_model(self):
        """Build the same model architecture as server"""
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_local_model(self, x_train, y_train, epochs=1, batch_size=32):
        """Train the local model on client's data"""
        x_train = x_train.reshape(-1, 784)  # Flatten the images
        
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history

    def get_model_weights(self):
        """Get model weights in a structured format"""
        weights = self.model.get_weights()
        weight_shapes = []
        for w in weights:
            weight_shapes.extend(list(w.shape))
            weight_shapes.append(-1)  # Add separator
        
        flattened = np.concatenate([w.flatten() for w in weights])
        return {
            'weights': flattened.tolist(),
            'shapes': weight_shapes
        }

    def update_server(self):
        """Send local model updates to server"""
        try:
            weights_data = self.get_model_weights()
            response = self.stub.UpdateModel(
                ModelUpdateRequest(
                    client_id=self.client_id,
                    model_update=weights_data['weights'],
                    weight_shapes=weights_data['shapes']
                )
            )
            return response.ack
        except grpc.RpcError as e:
            print(f"RPC error: {e}")
            return False

    def get_stats(self):
        """Get current model stats from server"""
        try:
            stats = self.stub.GetModelStats(ModelStatsRequest())
            return stats.accuracy, stats.loss
        except grpc.RpcError as e:
            print(f"RPC error: {e}")
            return None, None

    def get_contribution(self):
        """Get client's contribution count"""
        try:
            response = self.stub.GetClientContribution(
                ClientContributionRequest(client_id=self.client_id)
            )
            return response.contribution
        except grpc.RpcError as e:
            print(f"RPC error: {e}")
            return 0

    def predict(self, input_data):
        """Make prediction using the server model"""
        try:
            input_data = input_data.reshape(-1, 784)
            prediction = self.stub.Predict(
                PredictionRequest(input_data=input_data.flatten().tolist())
            )
            return np.array(prediction.result)
        except grpc.RpcError as e:
            print(f"RPC error: {e}")
            return None

def load_mnist_data():
    """Load and preprocess MNIST data"""
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    return x_train, y_train

def validate_near_address(address: str) -> bool:
    """Validate NEAR account ID format"""
    # NEAR account IDs must be at least 2 characters
    if len(address) < 2:
        return False
    
    # Must contain only lowercase letters, digits, or the following characters: - _ .
    valid_chars = set('abcdefghijklmnopqrstuvwxyz0123456789-_.')
    if not all(c in valid_chars for c in address):
        return False
    
    # Cannot start or end with special characters
    if address[0] in '-_.' or address[-1] in '-_.':
        return False
    
    return True

def simulate_federated_learning(wallet_address, server_address='localhost:50051', rounds=5):
    """Run federated learning with the given wallet address"""
    if not validate_near_address(wallet_address):
        print("Error: Invalid NEAR wallet address")
        return

    print(f"\nStarting Federated Learning Client")
    print(f"Wallet Address: {wallet_address}")
    print(f"Server Address: {server_address}")
    print(f"Training Rounds: {rounds}")
    
    # Create client instance
    client = FederatedClient(wallet_address, server_address)
    
    # Load MNIST data
    x_train, y_train = load_mnist_data()
    
    # Use a subset of data for training
    data_size = 1000  # Adjust this value based on your needs
    x_train = x_train[:data_size]
    y_train = y_train[:data_size]
    
    for round in range(rounds):
        print(f"\nTraining Round {round + 1}/{rounds}")
        
        # Train local model
        client.train_local_model(x_train, y_train, epochs=1)
        
        # Update server
        success = client.update_server()
        print(f"Update success: {success}")
        
        # Get stats
        accuracy, loss = client.get_stats()
        if accuracy is not None and loss is not None:
            print(f"Global model - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        
        # Get contribution
        contribution = client.get_contribution()
        print(f"Your contribution count: {contribution}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--address', type=str, required=True,
                      help='Your NEAR account ID (e.g., alice.near)')
    parser.add_argument('--server', type=str, default='localhost:50051',
                      help='Server address (default: localhost:50051)')
    parser.add_argument('--rounds', type=int, default=5,
                      help='Number of training rounds (default: 5)')
    
    args = parser.parse_args()
    
    simulate_federated_learning(
        wallet_address=args.address,
        server_address=args.server,
        rounds=args.rounds
    )
