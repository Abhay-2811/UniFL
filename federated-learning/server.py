import grpc
from concurrent import futures
import sqlite3
import tensorflow as tf
import numpy as np
from federated_model_pb2 import ModelUpdateResponse, ModelStatsResponse, ClientContributionResponse, PredictionResponse
from federated_model_pb2_grpc import FederatedLearningServicer, add_FederatedLearningServicer_to_server
from flask import Flask, jsonify
from flask_cors import CORS
import threading
import datetime
from flask_cors import cross_origin
import json
from collections import deque
import time
import asyncio
from near_api.account import Account
from near_api.signer import Signer
from near_api.providers import JsonProvider
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from dotenv import load_dotenv
import subprocess
load_dotenv()


# Database setup with proper schema
conn = sqlite3.connect("federated_learning.db", check_same_thread=False)
cursor = conn.cursor()

# Drop existing table if it exists
cursor.execute('DROP TABLE IF EXISTS contributions')

# Create table with all required columns
cursor.execute('''CREATE TABLE IF NOT EXISTS contributions (
    client_id TEXT PRIMARY KEY, 
    contribution INTEGER DEFAULT 0,
    total_accuracy_impact FLOAT DEFAULT 0.0,
    average_impact FLOAT DEFAULT 0.0,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)''')
conn.commit()

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],  # Allow Next.js development server
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# NEAR contract configuration
NEAR_NETWORK = "testnet"
NEAR_CONTRACT_ID = "unifl-test1.testnet"  # Contract address
NEAR_ACCOUNT_ID = os.getenv("NEAR_ACCOUNT_ID", "unifl.testnet")  # Contract owner
NEAR_PRIVATE_KEY = os.getenv("NEAR_PRIVATE_KEY")  # Get from unifl-test.testnet credentials

print(f"NEAR_ACCOUNT_ID: {NEAR_ACCOUNT_ID}")
print(f"NEAR_PRIVATE_KEY: {NEAR_PRIVATE_KEY}")


class FederatedLearningServer(FederatedLearningServicer):
    def __init__(self):
        self.global_model = self.build_model()
        self.client_updates = []
        self.min_clients_for_aggregation = 2
        self.round = 0
        # Load test data for evaluation
        (_, _), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_test = self.x_test.astype('float32') / 255.0
        self.x_test = self.x_test.reshape(-1, 784)  # Flatten the images
        self.model_stats_history = []
        self.previous_accuracy = 0.0  # Track previous accuracy for impact calculation
        self.server_logs = deque(maxlen=100)  # Keep last 100 logs
        self.log_event("Server initialized. Waiting for contributors...")
        self.contribution_weights = {}  # Track contribution weights for weighted averaging
        
        # Initialize NEAR connection
        try:
            # Verify NEAR CLI is available
            result = subprocess.run(['near', 'state', NEAR_ACCOUNT_ID], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                self.log_event("NEAR contract connection initialized", "success")
            else:
                raise Exception(f"Failed to verify NEAR account: {result.stderr}")
        except Exception as e:
            self.log_event(f"Failed to initialize NEAR connection: {str(e)}", "error")
            import traceback
            self.log_event(f"Traceback: {traceback.format_exc()}", "error")

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.executor = ThreadPoolExecutor(max_workers=3)

    def run_async(self, coro):
        """Run coroutine in the event loop"""
        return self.loop.run_until_complete(coro)

    def build_model(self):
        """
        Builds a CNN model for image classification (MNIST-like datasets)
        Input shape: (28, 28, 1) - grayscale images
        Output: 10 classes
        """
        model = tf.keras.Sequential([
            # Input layer - expects flattened images (784 pixels)
            tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
            
            # First Convolutional Block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense layers
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

    def calculate_contribution_weight(self, client_id):
        """Calculate weight for a client's contribution based on their history"""
        try:
            cursor.execute(
                "SELECT contribution, average_impact FROM contributions WHERE client_id = ?", 
                (client_id,)
            )
            result = cursor.fetchone()
            if result:
                contribution_count, avg_impact = result
                # Give more weight to contributors with good history
                base_weight = min(contribution_count, 10) / 10  # Cap at 10 contributions
                impact_multiplier = 1 + max(0, avg_impact)  # Positive impact increases weight
                return max(0.1, base_weight * impact_multiplier)  # Minimum weight of 0.1
            return 0.1  # Default weight for new contributors
        except sqlite3.Error:
            return 0.1

    async def update_near_contract(self, client_id: str, impact: float):
        """Update contributor score on NEAR contract using NEAR CLI"""
        try:
            cmd = [
                'near',
                'call',
                NEAR_CONTRACT_ID,
                'update_contributor_score',
                f'{{"contributor": "{client_id}", "impact": {impact}}}',
                '--accountId',
                NEAR_ACCOUNT_ID
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.log_event(f"Updated NEAR contract for {client_id}", "success")
            else:
                self.log_event(f"Failed to update NEAR contract: {result.stderr}", "error")
        except Exception as e:
            self.log_event(f"Failed to update NEAR contract: {str(e)}", "error")

    async def check_distribution(self):
        """Check and trigger token distribution using NEAR CLI"""
        try:
            # Get next distribution time
            view_cmd = [
                'near',
                'view',
                NEAR_CONTRACT_ID,
                'get_next_distribution_time',
                '{}'
            ]
            view_result = subprocess.run(view_cmd, capture_output=True, text=True)
            if view_result.returncode != 0:
                raise Exception(f"Failed to get distribution time: {view_result.stderr}")

            # Parse and log the next distribution time
            next_dist_time = int(view_result.stdout.strip().split("'")[1])
            current_time = int(time.time() * 1_000_000_000)
            
            self.log_event(
                f"Distribution check - Next: {datetime.datetime.fromtimestamp(next_dist_time / 1_000_000_000)}, "
                f"Current: {datetime.datetime.fromtimestamp(current_time / 1_000_000_000)}",
                "info"
            )

            if next_dist_time <= current_time:
                # Trigger distribution
                dist_cmd = [
                    'near',
                    'call',
                    NEAR_CONTRACT_ID,
                    'distribute_tokens',
                    '{}',
                    '--accountId',
                    NEAR_ACCOUNT_ID
                ]
                dist_result = subprocess.run(dist_cmd, capture_output=True, text=True)
                if dist_result.returncode == 0:
                    self.log_event("Token distribution completed", "success")
                else:
                    self.log_event(f"Distribution failed: {dist_result.stderr}", "error")
            else:
                self.log_event(f"Too early for distribution. {(next_dist_time - current_time) / 1_000_000_000} seconds remaining", "info")
        except Exception as e:
            self.log_event(f"Distribution check failed: {str(e)}", "error")

    def aggregate_models(self, updates, shapes, client_ids):
        """Performs federated averaging on client model updates with weighted contributions"""
        self.log_event(f"Starting model aggregation with {len(client_ids)} contributors")
        
        # Calculate weights for each contributor
        weights = [self.calculate_contribution_weight(client_id) for client_id in client_ids]
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        # Reconstruct weight shapes
        weight_shapes = []
        idx = 0
        while idx < len(shapes):
            current_shape = []
            while idx < len(shapes) and shapes[idx] != -1:
                current_shape.append(shapes[idx])
                idx += 1
            if current_shape:
                weight_shapes.append(tuple(current_shape))
            idx += 1
        
        # Convert updates to numpy arrays and reshape them
        reshaped_updates = []
        for update in updates:
            weights = []
            start_idx = 0
            for shape in weight_shapes:
                size = np.prod(shape)
                weight = np.array(update[start_idx:start_idx + size]).reshape(shape)
                weights.append(weight)
                start_idx += size
            reshaped_updates.append(weights)
        
        # Weighted average of the weights
        averaged_weights = []
        for weights_list in zip(*reshaped_updates):
            weighted_sum = sum(w * np.array(weight) for w, weight in zip(normalized_weights, weights_list))
            averaged_weights.append(weighted_sum)
        
        # Get previous metrics for comparison
        previous_loss, previous_accuracy = self.global_model.evaluate(
            self.x_test, self.y_test, 
            verbose=0
        )
        
        # Update global model
        self.global_model.set_weights(averaged_weights)
        
        # Evaluate after aggregation
        test_loss, test_accuracy = self.global_model.evaluate(
            self.x_test, self.y_test, 
            verbose=0
        )
        
        # If performance degrades significantly, rollback the update
        if test_loss > previous_loss * 1.5:  # 50% increase in loss
            self.log_event(
                f"Detected unstable update (loss spike). Rolling back...",
                "warning"
            )
            self.global_model.set_weights(self.global_model.get_weights())
            test_loss, test_accuracy = previous_loss, previous_accuracy
        
        # Run async operations synchronously using the event loop
        try:
            self.loop.run_until_complete(self.check_distribution())
        except Exception as e:
            self.log_event(f"Error in async operation: {str(e)}", "error")
        
        # Calculate improvement
        improvement = (test_accuracy - self.previous_accuracy) * 100
        
        # Log appropriate message based on performance change
        if self.previous_accuracy == 0:
            self.log_event(
                f"Initial model accuracy: {(test_accuracy * 100):.2f}%",
                "info"
            )
        else:
            if improvement > 0:
                self.log_event(
                    f"Model improved by {improvement:.2f}%! New accuracy: {(test_accuracy * 100):.2f}%",
                    "success"
                )
            elif improvement < 0:
                self.log_event(
                    f"Model accuracy decreased by {abs(improvement):.2f}%. Current: {(test_accuracy * 100):.2f}%",
                    "warning"
                )
            else:
                self.log_event(
                    f"Model accuracy unchanged. Current: {(test_accuracy * 100):.2f}%",
                    "info"
                )
        
        # Update stats and return
        self.previous_accuracy = test_accuracy
        self.round += 1
        
        # Store stats history
        self.model_stats_history.append({
            'accuracy': float(test_accuracy),
            'loss': float(test_loss),
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        return True

    def calculate_contribution_impact(self, client_id, new_accuracy):
        """Calculate the impact and update NEAR contract"""
        # Calculate accuracy change as percentage points
        if self.previous_accuracy == 0:
            accuracy_change = new_accuracy * 100  # First contribution
        else:
            accuracy_change = (new_accuracy - self.previous_accuracy) * 100

        # Debug logging
        self.log_event(
            f"Calculating impact - Previous: {self.previous_accuracy*100:.2f}%, "
            f"New: {new_accuracy*100:.2f}%, "
            f"Change: {accuracy_change:.2f}%",
            "info"
        )
        
        try:
            # Get existing contributor data
            cursor.execute(
                "SELECT contribution, total_accuracy_impact FROM contributions WHERE client_id = ?", 
                (client_id,)
            )
            result = cursor.fetchone()
            
            if result:
                current_contribution, current_total_impact = result
                new_contribution = current_contribution + 1
                new_total_impact = current_total_impact + accuracy_change
                new_average_impact = new_total_impact / new_contribution
            else:
                new_contribution = 1
                new_total_impact = accuracy_change
                new_average_impact = accuracy_change

            # Debug database update
            self.log_event(
                f"Updating database for {client_id}: "
                f"Contribution: {new_contribution}, "
                f"Total Impact: {new_total_impact:.2f}%, "
                f"Avg Impact: {new_average_impact:.2f}%",
                "info"
            )

            # Update database with new values
            cursor.execute('''
                INSERT OR REPLACE INTO contributions (
                    client_id, 
                    contribution,
                    total_accuracy_impact,
                    average_impact,
                    last_update
                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                client_id, 
                new_contribution,
                new_total_impact,
                new_average_impact
            ))
            conn.commit()

            # Debug database read
            cursor.execute(
                "SELECT contribution, total_accuracy_impact, average_impact FROM contributions WHERE client_id = ?",
                (client_id,)
            )
            stored_data = cursor.fetchone()
            self.log_event(
                f"Stored data for {client_id}: "
                f"Contribution: {stored_data[0]}, "
                f"Total Impact: {stored_data[1]:.2f}%, "
                f"Avg Impact: {stored_data[2]:.2f}%",
                "info"
            )

            # Update NEAR contract asynchronously
            asyncio.create_task(self.update_near_contract(client_id, accuracy_change))
            
        except sqlite3.Error as e:
            self.log_event(f"Database error: {e}", "error")
            
        return accuracy_change

    def UpdateModel(self, request, context):
        """Send local model updates to server"""
        client_id = request.client_id
        model_weights = np.array(request.model_update)
        weight_shapes = list(request.weight_shapes)
        
        # Store client update with ID
        self.client_updates.append((model_weights, client_id))
        
        # Aggregate models if we have enough updates
        if len(self.client_updates) >= self.min_clients_for_aggregation:
            updates, client_ids = zip(*self.client_updates)
            
            # Get previous metrics for comparison
            previous_loss, previous_accuracy = self.global_model.evaluate(
                self.x_test, self.y_test, 
                verbose=0
            )
            
            try:
                # Perform model aggregation
                success = self.aggregate_models(updates, weight_shapes, client_ids)
                
                if success:
                    # Get new metrics
                    test_loss, test_accuracy = self.global_model.evaluate(
                        self.x_test, self.y_test, 
                        verbose=0
                    )
                    
                    # Calculate impact
                    accuracy_change = (test_accuracy - previous_accuracy) * 100
                    
                    # Update database and NEAR contract synchronously
                    for cid in client_ids:
                        try:
                            # Get existing contributor data
                            cursor.execute(
                                "SELECT contribution, total_accuracy_impact FROM contributions WHERE client_id = ?", 
                                (cid,)
                            )
                            result = cursor.fetchone()
                            
                            if result:
                                current_contribution, current_total_impact = result
                                new_contribution = current_contribution + 1
                                new_total_impact = current_total_impact + accuracy_change
                                new_average_impact = new_total_impact / new_contribution
                            else:
                                new_contribution = 1
                                new_total_impact = accuracy_change
                                new_average_impact = accuracy_change

                            # Update database
                            cursor.execute('''
                                INSERT OR REPLACE INTO contributions (
                                    client_id, 
                                    contribution,
                                    total_accuracy_impact,
                                    average_impact,
                                    last_update
                                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                            ''', (
                                cid, 
                                new_contribution,
                                new_total_impact,
                                new_average_impact
                            ))
                            conn.commit()

                            # Run NEAR update synchronously
                            self.run_async(self.update_near_contract(cid, accuracy_change))
                            
                        except sqlite3.Error as e:
                            self.log_event(f"Database error for {cid}: {e}", "error")
                
                # Run distribution check synchronously
                self.run_async(self.check_distribution())
                
            except Exception as e:
                self.log_event(f"Error in model update: {str(e)}", "error")
                import traceback
                self.log_event(f"Traceback: {traceback.format_exc()}", "error")
                return ModelUpdateResponse(ack=False)
            
            self.client_updates = []  # Clear updates after aggregation

        return ModelUpdateResponse(ack=True)

    async def aggregate_models_async(self, updates, shapes, client_ids):
        """Async version of aggregate_models"""
        success = self.aggregate_models(updates, shapes, client_ids)
        if success:
            # Run NEAR contract updates in parallel
            await asyncio.gather(
                self.update_near_contract(client_ids[0], 0.1),  # Example impact
                self.check_distribution()
            )
        return success

    def GetModelStats(self, request, context):
        """Evaluate the global model on test data"""
        try:
            # Evaluate model on test data
            test_loss, test_accuracy = self.global_model.evaluate(
                self.x_test, self.y_test, 
                verbose=0
            )
            print(f"\nGlobal Model Evaluation - Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")
            return ModelStatsResponse(accuracy=float(test_accuracy), loss=float(test_loss))
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return ModelStatsResponse(accuracy=0.0, loss=0.0)

    def GetClientContribution(self, request, context):
        try:
            cursor.execute(
                "SELECT contribution FROM contributions WHERE client_id = ?", 
                (request.client_id,)
            )
            result = cursor.fetchone()
            contribution = result[0] if result else 0
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            contribution = 0
            
        return ClientContributionResponse(
            client_id=request.client_id, 
            contribution=contribution
        )

    def Predict(self, request, context):
        # Convert input data to correct shape and make prediction
        input_data = np.array(request.input_data).reshape(1, 784)  # Reshape for single prediction
        predictions = self.global_model.predict(input_data)
        return PredictionResponse(result=predictions.flatten().tolist())

    def log_event(self, message, event_type="info"):
        """Add a log entry with timestamp"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": message,
            "type": event_type  # info, success, warning, error
        }
        self.server_logs.append(log_entry)
        print(f"[{log_entry['type'].upper()}] {message}")

    @app.route('/debug')
    @cross_origin()
    def get_debug_info(self):
        cursor.execute('SELECT * FROM contributions')
        rows = cursor.fetchall()
        debug_info = {
            'contributions': [
                {
                    'client_id': row[0],
                    'contribution': row[1],
                    'total_impact': row[2],
                    'average_impact': row[3],
                    'last_update': row[4]
                }
                for row in cursor.fetchall()
            ],
            'model_stats': self.model_stats_history
        }
        return jsonify(debug_info)

server_instance = None

@app.route('/stats')
@cross_origin()
def get_stats():
    if server_instance:
        return jsonify(server_instance.model_stats_history)
    return jsonify([])

@app.route('/contributors')
@cross_origin()
def get_contributors():
    cursor.execute('''
        SELECT 
            client_id, 
            contribution, 
            total_accuracy_impact,
            average_impact,
            last_update 
        FROM contributions
        ORDER BY average_impact DESC
    ''')
    contributors = []
    for row in cursor.fetchall():
        # Debug print raw data
        print(f"Raw DB row: {row}")
        
        # Convert to proper types and scale
        contributor = {
            'client_id': row[0],
            'contribution': int(row[1]),
            'total_impact': float(row[2]) if row[2] is not None else 0.0,
            'average_impact': float(row[3]) if row[3] is not None else 0.0,
            'last_update': row[4]
        }
        contributors.append(contributor)
        print(f"Processed contributor data: {contributor}")
    
    return jsonify(contributors)

@app.route('/logs')
@cross_origin()
def get_logs():
    if server_instance:
        return jsonify(list(server_instance.server_logs))
    return jsonify([])

def serve():
    global server_instance
    
    # Verify NEAR credentials
    if not NEAR_PRIVATE_KEY or not NEAR_ACCOUNT_ID:
        print("Error: NEAR credentials not found in environment variables")
        print("Please set NEAR_PRIVATE_KEY and NEAR_ACCOUNT_ID")
        return
    
    # Create gRPC server
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_instance = FederatedLearningServer()
    add_FederatedLearningServicer_to_server(server_instance, grpc_server)
    grpc_server.add_insecure_port('[::]:50051')
    grpc_server.start()
    
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8000))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        grpc_server.stop(0)

if __name__ == "__main__":
    serve()
