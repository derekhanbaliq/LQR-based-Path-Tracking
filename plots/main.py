import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
    """
    diffs = trajectory - point
    distance = np.sum(diffs ** 2, 1)
    # distance_list = distance.tolist()
    # index = distance_list.index(min(distance_list))
    index = np.argmin(distance)
    return index


class Benchmark:
    def __init__(self, map_name='example'):
        self.map_name = map_name

        # benchmark [s; x; y; theta; gama; v_x; ax_mps2]
        self.data = np.loadtxt('log/raceline/' + self.map_name + '_raceline.csv', delimiter=';', skiprows=0)
        if map_name == 'example' or map_name == 'icra':
            self.data[:, 3] += np.pi / 2

        self.x = self.data[:, 1]
        self.y = self.data[:, 2]
        self.theta = self.data[:, 3]

        self.gama = self.data[:, 4]
        self.v = self.data[:, 5]
        self.trajectory = self.data[:, 1:3]


class LQRSteeringResults:
    def __init__(self, map_name='example'):
        self.map_name = map_name
        # observation ['time','x','y','theta',v_x']; action ['time','speed','steer']
        self.observation_data = pd.read_excel('log/lqr_steering/' + self.map_name + '_observation.xlsx')
        self.action_data = pd.read_excel('log/lqr_steering/' + self.map_name + '_action.xlsx')
        self.error_data = pd.read_excel('log/lqr_steering/' + self.map_name + '_errors.xlsx')
        self.time = np.array(self.observation_data[['time']].to_records().tolist())[:, 1]
        self.lap_time = self.time[-1]
        self.x = np.array(self.observation_data[['x']].to_records().tolist())[:, 1]
        self.y = np.array(self.observation_data[['y']].to_records().tolist())[:, 1]
        self.theta = np.array(self.observation_data[['theta']].to_records().tolist())[:, 1]
        self.v_x = np.array(self.observation_data[['v_x']].to_records().tolist())[:, 1]
        self.speed = np.array(self.action_data[['speed']].to_records().tolist())[:, 1]
        self.steer = np.array(self.action_data[['steer']].to_records().tolist())[:, 1]
        self.lateral_error = np.array(self.error_data[['e_l']].to_records().tolist())[:, 1]
        self.heading_error = np.array(self.error_data[['e_theta']].to_records().tolist())[:, 1]


class LQRSteeringSpeedResults:
    def __init__(self, map_name='example'):
        self.map_name = map_name
        # observation['time','x','y','theta',v_x'];action['time','speed','steer'];errors[e_l, e_theta]
        self.observation_data = pd.read_excel('log/lqr_steering_speed/' + self.map_name + '_observation.xlsx')
        self.action_data = pd.read_excel('log/lqr_steering_speed/' + self.map_name + '_action.xlsx')
        self.error_data = pd.read_excel('log/lqr_steering_speed/' + self.map_name + '_errors.xlsx')
        self.time = np.array(self.observation_data[['time']].to_records().tolist())[:, 1]
        self.lap_time = self.time[-1]
        self.x = np.array(self.observation_data[['x']].to_records().tolist())[:, 1]
        self.y = np.array(self.observation_data[['y']].to_records().tolist())[:, 1]
        self.theta = np.array(self.observation_data[['theta']].to_records().tolist())[:, 1]
        self.v_x = np.array(self.observation_data[['v_x']].to_records().tolist())[:, 1]
        self.speed = np.array(self.action_data[['speed']].to_records().tolist())[:, 1]
        self.steer = np.array(self.action_data[['steer']].to_records().tolist())[:, 1]
        self.lateral_error = np.array(self.error_data[['e_l']].to_records().tolist())[:, 1]
        self.heading_error = np.array(self.error_data[['e_theta']].to_records().tolist())[:, 1]


class PurePursuitResults:

    def __init__(self, map_name='example'):
        self.map_name = map_name
        self.benchmark = Benchmark(self.map_name)
        # observation['time','x','y','theta',v_x'];action['time','speed','steer'];errors[e_l, e_theta]
        self.observation_data = pd.read_excel('log/pure_pursuit/' + self.map_name + '_observation.xlsx')
        self.action_data = pd.read_excel('log/pure_pursuit/' + self.map_name + '_action.xlsx')
        self.time = np.array(self.observation_data[['time']].to_records().tolist())[:, 1]
        self.lap_time = self.time[-1]
        self.x = np.array(self.observation_data[['x']].to_records().tolist())[:, 1]
        self.y = np.array(self.observation_data[['y']].to_records().tolist())[:, 1]
        self.theta = np.array(self.observation_data[['theta']].to_records().tolist())[:, 1]
        self.v_x = np.array(self.observation_data[['v_x']].to_records().tolist())[:, 1]
        self.speed = np.array(self.action_data[['speed']].to_records().tolist())[:, 1]
        self.steer = np.array(self.action_data[['steer']].to_records().tolist())[:, 1]

        # calculate nearest points & lateral error & heading error
        lateral_error = []
        heading_error = []

        for i in range(len(self.x)):
            L = 0.33
            current_point = np.array([self.x[i] + L * np.cos(self.theta[i]), self.y[i] + L * np.sin(self.theta[i])])
            nearest_point_index = calc_nearest_point(current_point, self.benchmark.trajectory)
            nearest_point = self.benchmark.trajectory[nearest_point_index]
            heading_error_i = self.benchmark.theta[nearest_point_index] - self.theta[i]
            heading_error_i = (heading_error_i + np.pi) % (2 * np.pi) - np.pi
            vector = np.array([[np.cos(self.theta[i] - np.pi / 2.0)], [np.sin(self.theta[i] - np.pi / 2.0)]])
            lateral_error_i = -np.dot((nearest_point - current_point).T, vector)[0]
            heading_error.append(heading_error_i)
            lateral_error.append(lateral_error_i)

        self.lateral_error = np.array(lateral_error)
        self.heading_error = np.array(heading_error)


class StanleyResults:

    def __init__(self, map_name='example'):
        self.map_name = map_name
        self.benchmark = Benchmark(self.map_name)
        # observation['time','x','y','theta',v_x'];action['time','speed','steer'];errors[e_l, e_theta]
        self.observation_data = pd.read_excel('log/stanley/' + self.map_name + '_observation.xlsx')
        self.action_data = pd.read_excel('log/stanley/' + self.map_name + '_action.xlsx')
        self.time = np.array(self.observation_data[['time']].to_records().tolist())[:, 1]
        self.lap_time = self.time[-1]
        self.x = np.array(self.observation_data[['x']].to_records().tolist())[:, 1]
        self.y = np.array(self.observation_data[['y']].to_records().tolist())[:, 1]
        self.theta = np.array(self.observation_data[['theta']].to_records().tolist())[:, 1]
        self.v_x = np.array(self.observation_data[['v_x']].to_records().tolist())[:, 1]
        self.speed = np.array(self.action_data[['speed']].to_records().tolist())[:, 1]
        self.steer = np.array(self.action_data[['steer']].to_records().tolist())[:, 1]

        # calculate nearest points & lateral error & heading error
        lateral_error = []
        heading_error = []

        for i in range(len(self.x)):
            L = 0.33
            current_point = np.array([self.x[i] + L * np.cos(self.theta[i]), self.y[i] + L * np.sin(self.theta[i])])
            nearest_point_index = calc_nearest_point(current_point, self.benchmark.trajectory)
            nearest_point = self.benchmark.trajectory[nearest_point_index]
            heading_error_i = self.benchmark.theta[nearest_point_index] - self.theta[i]
            heading_error_i = (heading_error_i + np.pi) % (2 * np.pi) - np.pi
            vector = np.array([[np.cos(self.theta[i] - np.pi / 2.0)], [np.sin(self.theta[i] - np.pi / 2.0)]])
            lateral_error_i = -np.dot((nearest_point - current_point).T, vector)[0]
            heading_error.append(heading_error_i)
            lateral_error.append(lateral_error_i)

        self.lateral_error = np.array(lateral_error)
        self.heading_error = np.array(heading_error)


class DataAnalysis:
    def __init__(self, map_name='example'):
        self.map_name = map_name

        # load benchmark data, and simulation results:
        # lqr_steering_result(lsr); lqr_steering_speed_result (lssr);
        self.benchmark = Benchmark(self.map_name)
        self.lsr = LQRSteeringResults(self.map_name)
        self.lssr = LQRSteeringSpeedResults(self.map_name)
        self.pp = PurePursuitResults(self.map_name)
        self.s = StanleyResults(self.map_name)

        self.figure_count = 0

    def lap_time_comparison(self):
        self.figure_count += 1
        plt.figure(self.figure_count)
        X = ["LQR_steering", "LQR_steering_speed", "Pure_pursuit", "Stanley"]
        Y = [self.lsr.lap_time, self.lssr.lap_time, self.pp.lap_time, self.s.lap_time]
        plt.title("Lap Time Comparison")
        plt.bar(X, Y, width=0.5)
        plt.ylabel("Lap Time (s)")
        for a, b in zip(X, Y):
            plt.text(a, b, round(b, 2), ha='center', va='bottom')
        plt.savefig('./results/' + self.map_name + '/' + self.map_name + '_Lap Time Comparison.jpg')

    def trajectory_comparison(self):
        # plot
        self.figure_count += 1
        plt.figure(self.figure_count)
        plt.plot(self.benchmark.x, self.benchmark.y, label='Benchmark', linewidth=1)
        plt.plot(self.lsr.x, self.lsr.y, label='LQR_steering', linewidth=1)
        plt.plot(self.lssr.x, self.lssr.y, label='LQR_steering_speed', linewidth=1)
        plt.plot(self.pp.x, self.pp.y, label='Pure_pursuit', linewidth=1)
        # plt.plot(self.s.x, self.s.y, label='Stanley', linewidth=1)
        plt.title('Waypoints Comparison')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc="upper right")
        plt.savefig('./results/' + self.map_name + '/' + self.map_name + '_Waypoints Comparison.jpg')

    def speed_comparison(self):
        self.figure_count += 1
        plt.figure(self.figure_count)
        plt.plot(self.lsr.time, self.lsr.v_x, label='LQR_steering', linewidth=1)
        plt.plot(self.lssr.time, self.lssr.v_x, label='LQR_steering_speed', linewidth=1)
        plt.plot(self.pp.time, self.pp.v_x, label='Pure_pursuit', linewidth=1)
        plt.plot(self.s.time, self.s.v_x, label='Stanley', linewidth=1)
        plt.title('Speed Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.legend(loc="upper right")
        plt.savefig('./results/' + self.map_name + '/' + self.map_name + '_Speed Comparison.jpg')

    def lateral_error_comparison(self):
        # value compare
        lsr_max = max(abs(self.lsr.lateral_error))
        lsr_sum = sum(abs(self.lsr.lateral_error))
        lssr_max = max(abs(self.lssr.lateral_error))
        lssr_sum = sum(abs(self.lssr.lateral_error))
        pp_max = max(abs(self.pp.lateral_error))
        pp_sum = sum(abs(self.pp.lateral_error))
        s_max = max(abs(self.s.lateral_error))
        s_sum = sum(abs(self.s.lateral_error))
        # plot bar chart
        self.figure_count += 1
        plt.figure(self.figure_count)
        X = ["LQR_steering", "LQR_steering_speed", "Pure_pursuit", "Stanley"]
        Y = [lsr_max, lssr_max, pp_max, s_max]
        plt.title("Max Lateral Error Comparison")
        plt.bar(X, Y, width=0.5)
        plt.ylabel("Max Lateral Error (m)")
        for a, b in zip(X, Y):
            plt.text(a, b, round(b, 2), ha='center', va='bottom')
        plt.savefig('./results/' + self.map_name + '/' + self.map_name + '_Max Lateral Error Comparison.jpg')

        self.figure_count += 1
        plt.figure(self.figure_count)
        X = ["LQR_steering", "LQR_steering_speed", "Pure_pursuit", "Stanley"]
        Y = [lsr_sum, lssr_sum, pp_sum, s_sum]
        plt.title("Total Lateral Error Comparison")
        plt.bar(X, Y, width=0.5)
        plt.ylabel("Total Lateral Error (m)")
        for a, b in zip(X, Y):
            plt.text(a, b, round(b, 2), ha='center', va='bottom')
        plt.savefig('./results/' + self.map_name + '/' + self.map_name + '_Total Lateral Error Comparison.jpg')

        # plot
        self.figure_count += 1
        plt.figure(self.figure_count)
        plt.plot(self.lsr.time, self.lsr.lateral_error, label='LQR_steering', linewidth=1)
        plt.plot(self.lssr.time, self.lssr.lateral_error, label='LQR_steering_speed', linewidth=1)
        plt.plot(self.pp.time, self.pp.lateral_error, label='Pure_pursuit', linewidth=1)
        plt.plot(self.s.time, self.s.lateral_error, label='Stanley', linewidth=1)
        plt.title('Lateral Error Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Lateral Error (m)')
        plt.legend(loc="upper right")
        plt.savefig('./results/' + self.map_name + '/' + self.map_name + '_Lateral Error Comparison.jpg')

    def heading_error_comparison(self):
        # value compare
        lsr_max = max(abs(self.lsr.heading_error))
        lsr_sum = sum(abs(self.lsr.heading_error))
        lssr_max = max(abs(self.lssr.heading_error))
        lssr_sum = sum(abs(self.lssr.heading_error))
        pp_max = max(abs(self.pp.heading_error))
        pp_sum = sum(abs(self.pp.heading_error))
        s_max = max(abs(self.s.heading_error))
        s_sum = sum(abs(self.s.heading_error))

        # plot bar chart
        self.figure_count += 1
        plt.figure(self.figure_count)
        X = ["LQR_steering", "LQR_steering_speed", "Pure_pursuit", "Stanley"]
        Y = [lsr_max, lssr_max, pp_max, s_max]
        plt.title("Max Heading Error Comparison")
        plt.bar(X, Y, width=0.5)
        plt.ylabel("Max Heading Error (rad)")
        for a, b in zip(X, Y):
            plt.text(a, b, round(b, 2), ha='center', va='bottom')
        plt.savefig('./results/' + self.map_name + '/' + self.map_name + '_Max Heading Error Comparison.jpg')

        self.figure_count += 1
        plt.figure(self.figure_count)
        X = ["LQR_steering", "LQR_steering_speed", "Pure_pursuit", "Stanley"]
        Y = [lsr_sum, lssr_sum, pp_sum, s_sum]
        plt.title("Total Heading Error Comparison")
        plt.bar(X, Y, width=0.5)
        plt.ylabel("Total Heading Error (rad)")
        for a, b in zip(X, Y):
            plt.text(a, b, round(b, 2), ha='center', va='bottom')
        plt.savefig('./results/' + self.map_name + '/' + self.map_name + '_Total Heading Error Comparison.jpg')

        # plot
        self.figure_count += 1
        plt.figure(self.figure_count)
        plt.plot(self.lsr.time, self.lsr.heading_error, label='LQR_steering', linewidth=1)
        plt.plot(self.lssr.time, self.lssr.heading_error, label='LQR_steering_speed', linewidth=1)
        plt.plot(self.pp.time, self.pp.heading_error, label='Pure_pursuit', linewidth=1)
        plt.plot(self.s.time, self.s.heading_error, label='Stanley', linewidth=1)
        plt.title('Heading Error Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Heading Error (rad)')
        plt.legend(loc="upper right")
        plt.savefig('./results/'+self.map_name+'/'+self.map_name+'_Heading Error Comparison.jpg')


if __name__ == '__main__':
    # DataAnalysis(map_name): map_name: 'Spielberg'/'MoscowRaceway'/'Catalunya'/'example'/'icra'
    data = DataAnalysis('example')
    data.trajectory_comparison()
    data.lap_time_comparison()
    data.speed_comparison()
    data.lateral_error_comparison()
    data.heading_error_comparison()
    plt.show()
