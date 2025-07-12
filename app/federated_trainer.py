# federated_trainer.py

import time
import random
import numpy as np
from client import Client
from middle_server import MiddleServer
from global_server import GlobalServer
from data_loader import INPUT_DIM
from config import (
    NUM_CLIENTS, NUM_MIDDLE_SERVERS, COMMUNICATION_ROUNDS, LOCAL_EPOCHS,
    LEARNING_RATE, BATCH_SIZE, NUM_SAMPLED_CLIENTS_PER_ROUND
)

def hierarchical_training(clients_data, X_test_tensor, y_test_tensor):
    print("\n--- Setting up Federated System ---")
    start_time_setup = time.time()

    all_clients = [
        Client(client_id=i+1, data=clients_data.get(f'client_{i+1}'), input_dim=INPUT_DIM)
        for i in range(NUM_CLIENTS)
    ]
    NUM_ACTIVE_CLIENTS_INIT = sum(1 for c in all_clients if c.has_data)
    print(f"Initialized {len(all_clients)} clients ({NUM_ACTIVE_CLIENTS_INIT} with data).")

    middle_servers = [
        MiddleServer(server_id=i+1, input_dim=INPUT_DIM)
        for i in range(NUM_MIDDLE_SERVERS)
    ]
    if not middle_servers:
        raise SystemExit("No middle servers initialized.")
    print(f"Initialized {len(middle_servers)} middle servers.")

    global_server = GlobalServer(middle_servers=middle_servers, input_dim=INPUT_DIM)
    end_time_setup = time.time()
    print(f"System setup took: {end_time_setup - start_time_setup:.2f}s")

    print("\n--- Starting Hierarchical Federated Training ---")
    history = {
        'round': [], 'acc': [], 'loss': [], 'f1': [], 'auc': [], 'time': [], 'avg_client_loss': []
    }

    total_training_start_time = time.time()
    current_global_params = global_server.get_model_params()

    for r in range(COMMUNICATION_ROUNDS):
        round_start_time = time.time()
        print(f"\n--- Round {r+1}/{COMMUNICATION_ROUNDS} ---")

        clients_with_data = [c for c in all_clients if c.has_data]
        if len(clients_with_data) < NUM_SAMPLED_CLIENTS_PER_ROUND:
            active_clients = clients_with_data
        else:
            active_clients = random.sample(clients_with_data, NUM_SAMPLED_CLIENTS_PER_ROUND)

        print(f"  Selected {len(active_clients)} clients.")

        if not active_clients:
            print("  No clients active. Skipping round.")
            history['time'].append(time.time() - round_start_time)
            continue

        # Clear and assign clients to middle servers
        for s in middle_servers:
            s.assign_clients_for_round([])

        random.shuffle(active_clients)
        cursor = 0
        for i, server in enumerate(middle_servers):
            num_for_this = len(active_clients) // NUM_MIDDLE_SERVERS + (1 if i < len(active_clients) % NUM_MIDDLE_SERVERS else 0)
            assigned = active_clients[cursor: cursor + num_for_this]
            server.assign_clients_for_round(assigned)
            cursor += num_for_this

        global_server.distribute_model_to_middle_servers(current_global_params)
        for server in middle_servers:
            if server.has_clients:
                server.distribute_model_to_clients(server.get_model_params())

        client_params_map = {}
        client_losses = []
        for client in active_clients:
            if client.has_data:
                loss = client.train(epochs=LOCAL_EPOCHS)
                client_params_map[client.client_id] = client.get_model_params()
                client_losses.append(loss)
        avg_client_loss = np.mean([
            l for l in client_losses if l is not None and not np.isnan(l)
        ]) if client_losses else 0
        print(f"  Avg Client Loss: {avg_client_loss:.4f}")

        middle_aggregated_params_list = []
        for server in middle_servers:
            if server.has_clients:
                params = [client_params_map.get(c.client_id) for c in server.clients]
                aggregated = server.aggregate_client_models(params)
                if aggregated is not None:
                    middle_aggregated_params_list.append(aggregated)

        current_global_params = global_server.aggregate_middle_server_models(middle_aggregated_params_list)
        if current_global_params is None:
            print("  Warn: Global aggregation failed. Reusing previous.")
            current_global_params = global_server.get_model_params()

        global_acc, global_loss, global_f1, global_auc = global_server.evaluate_global_model(X_test_tensor, y_test_tensor)

        history['round'].append(r+1)
        history['acc'].append(global_acc)
        history['loss'].append(global_loss)
        history['f1'].append(global_f1)
        history['auc'].append(global_auc)
        history['time'].append(time.time() - round_start_time)
        history['avg_client_loss'].append(avg_client_loss)

        print(f"  Round {r+1} done in {history['time'][-1]:.2f}s")
        print(f"  Global Test --> Acc: {global_acc:.2f}% | Loss: {global_loss:.4f} | F1: {global_f1:.4f} | AUC: {global_auc:.4f}")

        for server in middle_servers:
            server.assign_clients_for_round([])

    total_time = time.time() - total_training_start_time
    avg_round_time = np.mean(history['time']) if history['time'] else 0
    print("\n--- Federated Training Completed ---")
    print(f"Total Time: {total_time:.2f}s | Avg Round: {avg_round_time:.2f}s")

    final_acc, final_loss, final_f1, final_auc = global_server.evaluate_global_model(X_test_tensor, y_test_tensor)
    print(f"\n--- Final Global Model ---")
    print(f"Acc: {final_acc:.2f}% | Loss: {final_loss:.4f} | F1: {final_f1:.4f} | AUC: {final_auc:.4f}")

    print("\n--- Final Client Evaluations ---")
    client_test_accs = []
    client_test_f1s = []
    for client in all_clients:
        if client.has_data:
            acc, f1 = client.evaluate_on_global_test(X_test_tensor, y_test_tensor, BATCH_SIZE * 4)
            client_test_accs.append(acc)
            client_test_f1s.append(f1)
            print(f"  Client {client.client_id}: Acc = {acc:.2f}%, F1 = {f1:.4f}")

    if client_test_accs:
        print(f"\nAvg Client Acc: {np.mean(client_test_accs):.2f}% (Std: {np.std(client_test_accs):.2f}%)")
        print(f"Avg Client F1: {np.mean(client_test_f1s):.4f} (Std: {np.std(client_test_f1s):.4f})")

    return history, client_test_accs, client_test_f1s
