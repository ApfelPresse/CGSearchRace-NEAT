import math

import numpy as np
from CGSearchRace.Constants import Constants
from CGSearchRace.Referee import Referee
from CGSearchRace.Tracks import tracks

from visualize_run import plot_current_frame, convert_to_gif


def min_max_scaler(value: float, min_value: float, max_value: float, range_from: float = -6,
                   range_to: float = 6):
    width = range_to - range_from
    return (value - min_value) / (max_value - min_value) * width + range_from


def distance_line_and_point(x, y, check_x, check_y, angle):
    length = 10000
    endy = y + length * math.sin(angle)
    endx = x + length * math.cos(angle)
    p1 = np.array([x, y])
    p2 = np.array([endx, endy])
    p3 = np.array([check_x, check_y])

    d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    if d < Constants.CheckpointRadius:
        d = 0
    return d


def simulate(net, create_gif=False):
    Constants.MAX_TIME = 100
    Constants.Laps = 2

    total_score = 0
    tr = [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    track_subset = [tracks[i] for i in tr]
    for i, track in enumerate(track_subset):
        ref = Referee(track)
        total_score += run_track(net, ref, create_gif)
    return -total_score


def distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def run_track(net, ref, create_gif):
    offset = 2000
    max_x = Constants.Width + offset
    max_y = Constants.Height + offset
    max_thrust = Constants.CAR_MAX_THRUST
    max_distance = math.sqrt(max_x ** 2 + max_y ** 2)
    images = []

    for i in range(Constants.MAX_TIME):
        cp = ref.game.checkpoints
        cp_id1 = ref.game.get_next_checkpoint_id()
        cp_id2 = ref.game.get_next_checkpoint_id(2)

        input_net = create_net_input({
            "max_distance": max_distance,
            "max_x": max_x,
            "max_y": max_y,
            "offset": offset,
            "angle": ref.game.car.angle,
            "vx": ref.game.car.vx,
            "vy": ref.game.car.vy,
            "car_x": ref.game.car.x,
            "car_y": ref.game.car.y,
            "check1_x": cp[cp_id1].x,
            "check1_y": cp[cp_id1].y,
            "check2_x": cp[cp_id2].x,
            "check2_y": cp[cp_id2].y,
            "dist_check1": distance([cp[cp_id1].x, cp[cp_id1].y], [ref.game.car.x, ref.game.car.y]),
            "dist_check2": distance([cp[cp_id2].x, cp[cp_id2].y], [ref.game.car.x, ref.game.car.y]),
            "angle_distance_check1": distance_line_and_point(ref.game.car.x, ref.game.car.y, cp[cp_id1].x, cp[cp_id1].y,
                                                             ref.game.car.angle),
        })

        input_net = np.round(input_net, decimals=3).tolist()
        predict = net.activate(input_net)

        input_x = int(predict[0] * max_x)
        input_y = int(predict[1] * max_y)
        input_thrust = int(predict[2] * max_thrust)

        ref.game.input = f"{input_x} {input_y} {input_thrust}"
        # ref.game.input = f"{cp[cp_id1].x} {cp[cp_id1].y} 50"
        ref.game_turn()

        if create_gif and i % 2 == 0:
            images.append(plot_current_frame(cp, cp_id1, ref.game.car))

        if ref.game.isDone:
            if create_gif:
                convert_to_gif(f"track", images)

            bonus = 20 if (i + 1) != Constants.MAX_TIME else 0
            # thrust_ra = (np.mean(thrust_ratio) // Constants.CAR_MAX_THRUST) * 10
            return (i + 1) - ref.game.currentCheckpoint - bonus


def create_net_input(params):
    input_net = [
        min_max_scaler(params["check1_x"], -params["offset"], params["max_x"]),
        min_max_scaler(params["check1_y"], -params["offset"], params["max_y"]),
        # min_max_scaler(params["check2_x"], -params["offset"], params["max_x"]),
        # min_max_scaler(params["check2_y"], -params["offset"], params["max_y"]),
        min_max_scaler(params["car_x"], -params["offset"], params["max_x"]),
        min_max_scaler(params["car_y"], -params["offset"], params["max_y"]),
        min_max_scaler(params["dist_check1"], 0, params["max_distance"]),
        min_max_scaler(params["dist_check2"], 0, params["max_distance"]),
        min_max_scaler(params["angle_distance_check1"], 0, params["max_distance"]),
        min_max_scaler(params["vx"], -1000, 2000),
        min_max_scaler(params["vy"], -1000, 2000),
    ]
    return input_net
