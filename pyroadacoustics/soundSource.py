import copy

import numpy as np

class SoundSource:
    """
    A class that defines a sound source. The sound source is assumed to be point-like and 
    omnidirectional, and emits an arbitrary sound signal. It is initially located in a 
    defined position, and moves along an arbitrary trajectory defined as a series of segments
    connecting a set of points, given as input.

    Attributes
    ----------
    position: np.ndarray
        1D array that contains initial position of sound source as a set of 3 cartesian coordinates `[x,y,z]`
    dir_pattern: str
            Directivity pattern of the source (omnidirectional, subcardioid, cardioid, supercardioid, hypercardioid, figure 8)
    orientation: float
        Angle in degrees towards which the directivity pattern is oriented
    signal: np.ndarray
        1D array that contains samples of signal emitted by the sound source
    fs: int
        Sampling frequency of the emitted signal
    is_static: bool
        True if the source is static during the simulation
    static_simduration: int
        If the source is static, defines the duration of the simulations in seconds. If the source moves,
        the simulation duration is defined by the time it takes to travel the whole trajectory

    Methods
    -------
    set_trajectory(positions, speed):
        Defines a trajectory from a given set of N positions (`positions`) and N-1 velocities (`speed`). The speed
        is assumed to be constant between each couple of positions.
    set_signal(signal):
        Setter for signal attribute: assigns given signal to the sound source
    """

    def __init__(
            self,
            position=np.array([0., 0., 1.]),
            dir_pattern='omnidirectional',
            orientation=0,
            fs=8000,
            update_fs=20
        ) -> None:
        """
        Create SoundSource object by defining its initial position, trajectory, 
        emitted signal and sampling frequency.

        Parameters
        ----------
        position : ndarray
            1D Array containing 3 cartesian coordinates `[x,y,z]` that define initial source position, 
            by default [0,0,1]
        dir_pattern: str
            Directivity pattern of the source (omnidirectional, subcardioid, cardioid, supercardioid, hypercardioid, figure 8)
        orientation: float
            Angle in degrees towards which the directivity pattern is oriented
        fs : int, optional
            Sampling frequency of the emitted signal, by default 8000
        is_static: Bool, optional
            True if the source is static
        static_simduration: float, optional
            If the source is static, defines the duration of the simulations in seconds. If the source moves,
            the simulation duration is defined by the time it takes to travel the whole trajectory
        """

        self.position = position
        self.dir_pattern = dir_pattern
        self.src_orientation = orientation
        self.signal = None
        self.fs = fs
        self.update_fs = update_fs
        self.signal_interval = int(self.fs / self.update_fs)
        self.current_index = 0
        trajectory_count = self.fs / self.update_fs
        self.trajectory = np.tile(self.position, (round(trajectory_count), 1))

    def set_source_index(self, new_index: int):
        self.current_index = copy.deepcopy(new_index)

    def get_source_index(self):
        return copy.deepcopy(self.current_index)

    def increase_source_index(self):
        self.current_index = self.current_index + self.signal_interval

    def extract_current_interval(self, start_index, update_index=False) -> np.ndarray:
        n = len(self.signal)

        end_index = start_index + self.signal_interval

        if update_index:
            self.current_index = end_index

        return np.concatenate([self.signal[start_index:min(end_index, n)], self.signal[:max(end_index - n, 0)]])


        
    def set_signal(self, signal: np.ndarray) -> None:
        """
        Setter for signal attribute: assigns given signal to the sound source

        Parameters
        ----------
        signal : np.ndarray
            1D Array containing samples of the source signal
        """
        self.signal = signal
    # def set_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
    #     """
    #     Defines a trajectory for the sound source from a set of N positions.
    #
    #     Parameters
    #     ----------
    #     positions : ndarray
    #         2D Array containing N sets of 3 cartesian coordinates `[x,y,z]` defining the desired trajectory positions.
    #         Each couple of subsequent points define a straight segment on the overall trajectory
    #
    #     Returns
    #     -------
    #     trajectory: ndarray
    #         2D Array containing N' sets of 3 cartesian coordinates `[x,y,z]` defining the full sampled trajectory
    #
    #     Raises
    #     ------
    #     RuntimeError
    #         if 'trajectory' is assigned to a static source
    #     RuntimeError
    #         if yhe length of trajectory do not match the interval between two update_fs points
    #
    #     Modifies
    #     --------
    #     trajectory
    #         Parameter is updated with the new computed trajectory
    #
    #     """
    #
    #     if self.is_static:
    #         raise RuntimeError("Cannot assign trajectory to static source")
    #
    #     if len(trajectory) == self.fs / self.update_fs:
    #         self.trajectory = trajectory
    #         self.position = self.trajectory[-1]
    #     else:
    #         raise RuntimeError("Length of the trajectory is {} while it must be {}".format(len(trajectory),
    #                                                                                        self.fs / self.update_fs))
    #     return trajectory

    def set_trajectory(self, new_position: np.ndarray = None) -> np.ndarray:
        """
        Defines a trajectory for the sound source from the current position to the new one.
        The trajectory is defined as the positions covered by the sound source at each sample of the simulation.
        Therefore, given the first and the last points, the method first computes a set of N-1 interpolation points
        connecting the two positions.

        Parameters
        ----------
        new_position : ndarray
            Array containing the 3 cartesian coordinates `[x,y,z]` defining the new position of the source.

        Returns
        -------
        trajectory: ndarray
            2D Array containing N' sets of 3 cartesian coordinates `[x,y,z]` defining the full sampled trajectory

        Raises
        ------
        RuntimeError
            if a trajectory is assigned to a static source

        Modifies
        --------
        trajectory
            Parameter is updated with the new computed trajectory

        """

        # Number of samples of the trajectory
        trajectory_count = round(self.fs / self.update_fs)

        if new_position is None or np.array_equal(new_position, self.position):
            self.trajectory = np.tile(self.position, (trajectory_count, 1))
        else:
            # Create a new array with N points for each dimension
            trajectory = np.zeros((trajectory_count, 3))

            # For each dimension...
            for i in range(3):
                # ...use linspace to create N evenly spaced values between the start and end points
                trajectory[:, i] = np.linspace(self.position[i], new_position[i], trajectory_count, endpoint=False)
            self.position = copy.deepcopy(new_position)
            self.trajectory = copy.deepcopy(trajectory)

        return self.trajectory
