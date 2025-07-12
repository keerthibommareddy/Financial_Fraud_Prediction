# plots.py

import matplotlib.pyplot as plt
import numpy as np

def plot_full_history(history, client_test_accuracy=None, client_test_f1=None, num_clients=12):
    print("\n--- 11. Plotting Results ---")
    try:
        rounds_range = history['round']
        if not rounds_range:
            raise ValueError("No rounds found in history.")

        plt.figure(figsize=(24, 12))

        # 1. Global Accuracy
        plt.subplot(2, 4, 1)
        plt.plot(rounds_range, history['acc'], 'bo-', label='Accuracy')
        plt.axhline(90, color='grey', linestyle='--', label='90% Target')
        plt.title('Global Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Acc %')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.xticks(np.arange(0, max(rounds_range) + 1, step=max(1, len(rounds_range) // 5)))

        # 2. Global Loss
        plt.subplot(2, 4, 2)
        plt.plot(rounds_range, history['loss'], 'rs-', label='Loss')
        plt.title('Global Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.xticks(np.arange(0, max(rounds_range) + 1, step=max(1, len(rounds_range) // 5)))

        # 3. Global F1 Score
        plt.subplot(2, 4, 3)
        plt.plot(rounds_range, history['f1'], 'g^-', label='F1')
        plt.title('Global F1')
        plt.xlabel('Round')
        plt.ylabel('F1 Score')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.xticks(np.arange(0, max(rounds_range) + 1, step=max(1, len(rounds_range) // 5)))
        plt.ylim(0, 1)

        # 4. Global AUC
        plt.subplot(2, 4, 4)
        plt.plot(rounds_range, history['auc'], 'p-', color='purple', label='AUC')
        plt.title('Global AUC')
        plt.xlabel('Round')
        plt.ylabel('AUC')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.xticks(np.arange(0, max(rounds_range) + 1, step=max(1, len(rounds_range) // 5)))
        plt.ylim(0, 1)

        # 5. Average Client Loss
        plt.subplot(2, 4, 5)
        plt.plot(rounds_range, history['avg_client_loss'], 'x-', color='orange', label='Avg Client Loss')
        plt.title('Avg Client Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.xticks(np.arange(0, max(rounds_range) + 1, step=max(1, len(rounds_range) // 5)))

        # 6. Final Client Accuracy Histogram
        if client_test_accuracy:
            plt.subplot(2, 4, 6)
            plt.hist(client_test_accuracy, bins=max(5, num_clients // 2), edgecolor='k')
            plt.title('Final Client Acc Distr.')
            plt.xlabel('Acc %')
            plt.ylabel('# Clients')
            plt.grid(True, axis='y', linestyle=':')
            mean_acc = np.mean(client_test_accuracy)
            plt.axvline(mean_acc, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_acc:.2f}%')
            plt.legend()

        # 7. Final Client F1 Score Histogram
        if client_test_f1:
            plt.subplot(2, 4, 7)
            plt.hist(client_test_f1, bins=max(5, num_clients // 2), edgecolor='k', color='green')
            plt.title('Final Client F1 Distr.')
            plt.xlabel('F1 Score')
            plt.ylabel('# Clients')
            plt.grid(True, axis='y', linestyle=':')
            mean_f1 = np.mean(client_test_f1)
            plt.axvline(mean_f1, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_f1:.4f}')
            plt.legend()

        plt.tight_layout(pad=2.5)
        plt.suptitle('HFL Performance (10k Sample, Uneven Non-IID, 10 Rounds, High Local Epochs)', fontsize=16, y=1.03)
        plt.show()
        print("Plots generated successfully.")
    except Exception as e:
        print(f"Plotting error: {e}")
