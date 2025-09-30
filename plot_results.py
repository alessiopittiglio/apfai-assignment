import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = 'results'
PLOT_DIR = 'plots'

strong_file = os.path.join(RESULTS_DIR, 'strong_scaling_results.txt')
weak_file = os.path.join(RESULTS_DIR, 'weak_scaling_results.txt')
throughput_file = os.path.join(RESULTS_DIR, 'compare_cpu_gpu_results.txt')

def read_results(file_path):
    df = pd.read_csv(file_path, sep='\s+')
    time_cols = [col for col in df.columns if col.startswith('t')]
    df['t_mean'] = df[time_cols].mean(axis=1)
    return df

if __name__ == "__main__":
    strong_df = read_results(strong_file)
    t1 = strong_df['t_mean'].iloc[0]
    strong_df['speedup'] = t1 / strong_df['t_mean']
    strong_df['efficiency'] = strong_df['speedup'] / strong_df['p']

    fig1, ax1 = plt.subplots()
    ax1.plot(strong_df['p'], strong_df['speedup'], marker='s', linestyle='-', label='Speedup')
    ax1.plot(strong_df['p'], strong_df['p'], linestyle='--', label='p')
    ax1.set_title('Speedup')
    ax1.set_xlabel('N. of cores (p)')
    ax1.set_ylabel('Speedup')
    ax1.set_xticks(strong_df['p'])
    ax1.legend()
    ax1.grid(True)
    fig1.savefig(os.path.join(PLOT_DIR, 'strong_scaling_speedup.pdf'))

    fig2, ax2 = plt.subplots()
    ax2.plot(strong_df['p'], strong_df['efficiency'], marker='o', linestyle='-', label='Strong scaling efficiency')
    ax2.set_title('Strong scaling efficiency')
    ax2.set_xlabel('N. of cores (p)')
    ax2.set_ylabel('Efficiency')
    ax2.set_xticks(strong_df['p'])
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True)
    fig2.savefig(os.path.join(PLOT_DIR, 'strong_scaling_efficiency.pdf'))

    weak_df = read_results(weak_file)
    t1 = weak_df['t_mean'].iloc[0]
    weak_df['efficiency'] = t1 / weak_df['t_mean']

    fig3, ax3 = plt.subplots()
    ax3.plot(weak_df['p'], weak_df['efficiency'], marker='o', linestyle='-', label='Weak scaling efficiency')
    ax3.set_title('Weak scaling efficiency')
    ax3.set_xlabel('N. of cores (p)')
    ax3.set_ylabel('Efficiency')
    ax3.set_xticks(weak_df['p'])
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.grid(True)
    fig3.savefig(os.path.join(PLOT_DIR, 'weak_scaling_efficiency.pdf'))

    df = pd.read_csv(throughput_file, sep='\s+')

    fig4, ax4 = plt.subplots()
    ax4.plot(df['Problem_Size_(N)'], df['OMP_Time_(s)'], marker='s', linestyle='-', label='OpenMP')
    ax4.plot(df['Problem_Size_(N)'], df['CUDA_Global_Time_(s)'], marker='s', linestyle='-', label='CUDA')
    # ax4.plot(df['Problem_Size_(N)'], df['CUDA_Shared_Time_(s)'], marker='s', linestyle='-', label='CUDA')
    ax4.set_title('Wall-clock times')
    ax4.set_xlabel('Problem size (N)')
    ax4.set_ylabel('Execution time (s)')
    ax4.legend()
    ax4.grid(True)
    fig4.savefig(os.path.join(PLOT_DIR, 'wall_clock.pdf'))

    fig5, ax5 = plt.subplots()
    ax5.plot(df['Problem_Size_(N)'], df['OMP_Tput_(Mneurons/s)'], marker='s', linestyle='-', label='OpenMP')
    # ax5.plot(df['Problem_Size_(N)'], df['CUDA_Shared_Tput_(Mneurons/s)'], marker='s', linestyle='-', label='CUDA')
    ax5.plot(df['Problem_Size_(N)'], df['CUDA_Global_Tput_(Mneurons/s)'], marker='s', linestyle='-', label='CUDA')
    ax5.set_title('Throughput (Mneurons/s)')
    ax5.set_xlabel('Problem size (N)')
    ax5.set_ylabel('Throughput')
    ax5.legend()
    ax5.grid(True)

    # x_last = df['Problem_Size_(N)'].iloc[-1]
    # y_last_cuda = df['CUDA_Shared_Tput_(Mneurons/s)'].iloc[-1]
    # gbps_cuda = y_last_cuda * 0.004 # 4 bytes per neuron

    # ax5.annotate(
    #     f"{gbps_cuda:.2f} GB/s",
    #     xy=(x_last, y_last_cuda),
    #     xytext=(x_last*0.7, y_last_cuda*1.1),
    # )

    # y_last_omp = df['OMP_Tput_(Mneurons/s)'].iloc[-1]
    # gbps_omp = y_last_omp * 0.004

    # ax5.annotate(
    #     f"{gbps_omp:.2f} GB/s",
    #     xy=(x_last, y_last_omp),
    #     xytext=(x_last*0.7, y_last_omp*1.1),
    # )

    fig5.savefig(os.path.join(PLOT_DIR, 'throughput.pdf'))

    df['Speedup_Shared'] = df['OMP_Time_(s)'] / df['CUDA_Shared_Time_(s)']
    df['Speedup_Global'] = df['OMP_Time_(s)'] / df['CUDA_Global_Time_(s)']

    fig6, ax6 = plt.subplots()
    N_vals = df['Problem_Size_(N)'].values
    num_bars = len(N_vals)
    x = np.arange(num_bars)
    width = 0.35

    bars_shared = ax6.bar(x - width/2, df['Speedup_Shared'], width=width, color='green', label='CUDA Shared')
    bars_global = ax6.bar(x + width/2, df['Speedup_Global'], width=width, color='blue', label='CUDA Global')

    for bar in bars_shared:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2 - 0.1, height + 0.05, f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    for bar in bars_global:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2 + 0.1, height + 0.05, f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    ax6.set_title('CUDA speedup vs. OpenMP')
    ax6.set_xlabel('Problem size (N)')
    ax6.set_ylabel('Speedup')
    ax6.set_xticks(x)
    ax6.set_xticklabels(N_vals, rotation=30, ha='right')
    ax6.set_ylim(0, max(df['Speedup_Shared'].max(), df['Speedup_Global'].max()) * 1.15)
    ax6.legend()
    ax6.grid(True, axis='y', linestyle='--', alpha=0.7)
    fig6.tight_layout()
    fig6.savefig(os.path.join(PLOT_DIR, 'cuda_speedup_vs_openmp.pdf'))
