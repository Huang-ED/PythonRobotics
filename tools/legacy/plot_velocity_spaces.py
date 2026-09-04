import json
import matplotlib.pyplot as plt
import numpy as np

def plot_dwa_log(log_file):
    with open(log_file, 'r') as f:
        data = json.load(f)
    data = data['log_entries']

    for entry in data:
        dw = entry["dynamic_window"]
        adm = np.array(entry["admissible"])
        inadm = np.array(entry["inadmissible"])

        plt.clf()
        plt.title(f'Dynamic Window (Iteration {entry["iteration"]})')
        plt.xlabel('v (m/s)')
        plt.ylabel('omega (rad/s)')
        plt.xlim(dw[0], dw[1])
        plt.ylim(dw[2], dw[3])

        if inadm.size > 0:
            plt.scatter(inadm[:, 0], inadm[:, 1], c='red', s=10, label='Inadmissible')
        if adm.size > 0:
            plt.scatter(adm[:, 0], adm[:, 1], c='green', s=10, label='Admissible')

        plt.legend()
        plt.grid(True)
        plt.pause(0.1)  # Control animation speed

    plt.show()

if __name__ == '__main__':
    plot_dwa_log("dwa_log_20250217_180757.json")  # Replace with your filename