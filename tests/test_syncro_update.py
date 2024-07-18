import copy
import time

from pyroadacoustics.pyroadacoustics.environment import Environment
from pyroadacoustics.pyroadacoustics.soundSource import SoundSource
from matplotlib import pyplot as plt
from scipy.io import wavfile
import scipy
import numpy as np
import os

if __name__ == "__main__":
    current_working_directory = os.getcwd()

    # print output to the console
    print(current_working_directory)

    rl_fs = 20
    fs = 8000
    fs_control = 20
    temperature = 24
    pressure = 1
    rel_humidity = 50
    simulation_params: dict = {
                "interp_method": "Sinc",
                "include_reflected_path": False,
                "include_air_absorption": False,
        }
    env = Environment(fs=fs,
                      fs_update=fs_control,
                      temperature=temperature,
                      pressure=pressure,
                      rel_humidity=rel_humidity)

    env.set_simulation_params(simulation_params["interp_method"],
                              simulation_params["include_reflected_path"],
                              simulation_params["include_air_absorption"])

    # Source signal
    T = 1
    N = int(T * fs)  # Total number of samples
    t = np.linspace(0.0, T, N, endpoint=False)
    # t = np.array(list(range(fs * 10)), dtype=float) / fs
    F1 = 1000.0  # Frequency of signal 1
    F2 = 100.0  # Frequency of signal 2

    src_signal = np.sin(F1 * 2 * np.pi * t) + np.sin(F2 * 2 * np.pi * t)

    # Visualize source signal
    plt.figure()
    ff, tt, Sxx = scipy.signal.spectrogram(src_signal, fs=fs)
    plt.pcolormesh(tt, ff, Sxx, shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    env.add_source(position=np.array([3., 10., 1.]),
                   signal=src_signal)

    env.add_microphone_array(np.array([[0., 0., 1.],
                                       [0., 0.5, 1.],
                                       [0., 1., 1.]]))
    # env.add_microphone_array(np.array([[0., 0., 1.]]))

    # env.plot_environment()

    elapsed_time = 0.0
    ts = 1 / fs_control
    vel = 30.0  # m/s
    signals = None
    new_position = copy.deepcopy(env.source.position)
    start_t = time.time()
    while elapsed_time < T:
        new_position[1] -= vel * ts
        env.move_source(new_position)
        if elapsed_time == 0.:
            signals = env.simulate(init=True)
        else:
            curr_signals = env.simulate()
            signals = np.concatenate([signals, curr_signals], axis=1)
        # env.source.position = copy.deepcopy(new_position)
        elapsed_time += ts
    print("Simulation time: {} s - Real time elapsed: {:.2f} s".format(T, time.time() - start_t))
    # plot signals received at the 3 microphones
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t[:len(signals[0, :])], signals[0, :])
    ax[1].plot(t[:len(signals[1, :])], signals[1, :])
    ax[2].plot(t[:len(signals[2, :])], signals[2, :])
    # fig, ax = plt.subplots()
    # ax.plot(t[:len(signals[0, :])], signals[0, :])
    plt.show()

    plt.figure()
    ff, tt, Sxx = scipy.signal.spectrogram(signals[0, :], fs=fs)
    plt.pcolormesh(tt, ff, Sxx, shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()