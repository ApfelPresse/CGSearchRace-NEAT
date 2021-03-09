import math
import time

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
    Constants.MAX_TIME = 300
    Constants.Laps = 2

    total_score = 0
    tr = [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, -1, -2, -3, -4, -5, 31]
    # tr = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, -1, -2, -3, -4, -5, 31]
    track_subset = [tracks[i] for i in tr]
    for i, track in enumerate(track_subset):
        ref = Referee(track)
        total_score += run_track(net, ref, create_gif)
    return -total_score


def distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def run_track(net, ref, create_gif):
    offset = 6000
    max_x = Constants.Width + offset
    max_y = Constants.Height + offset
    max_thrust = Constants.CAR_MAX_THRUST
    max_distance = math.sqrt(max_x ** 2 + max_y ** 2)
    images = []

    # last_checkpoint = None
    # drive_thru_error = 0
    # activate = True
    #
    # drive_thru_error_list = []

    # last_offset_x = 0
    # last_offset_y = 0
    # last_thrust = 0

    check_distances = []
    for i in range(Constants.MAX_TIME):
        cp = ref.game.checkpoints
        cp_id1 = ref.game.get_next_checkpoint_id()
        cp_id2 = ref.game.get_next_checkpoint_id(2)

        # if not last_checkpoint or last_checkpoint != cp_id1:
        #     last_checkpoint = cp_id1
        #     drive_thru_error_list.append(drive_thru_error)
        #     drive_thru_error = 0
        #     activate = False

        dist_check1 = distance([cp[cp_id1].x, cp[cp_id1].y], [ref.game.car.x, ref.game.car.y])

        # if dist_check1 < 3000:
        #     activate = True

        # if activate:
        #     drive_thru_error += (dist_check1 // 10000)
        # drive_thru_error = drive_thru_error * 2

        input_net = create_net_input({
            # "last_offset_x": last_offset_x,
            # "last_offset_y": last_offset_y,
            # "last_thrust": last_thrust,
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
            "dist_check1": dist_check1,
            "dist_check2": distance([cp[cp_id2].x, cp[cp_id2].y], [ref.game.car.x, ref.game.car.y]),
            "angle_distance_check1": distance_line_and_point(ref.game.car.x, ref.game.car.y, cp[cp_id1].x, cp[cp_id1].y,
                                                             ref.game.car.angle),
        })
        error_dist = int(distance([cp[cp_id1].x, cp[cp_id1].y], [ref.game.car.x, ref.game.car.y]))
        if error_dist < 2:
            error_dist = 2
        check_distances.append(error_dist)

        input_net = np.round(input_net, decimals=4).tolist()
        predict = net.activate(input_net)

        offset_x = (int(predict[0] * 3000) * 2) - 3000
        offset_y = (int(predict[1] * 3000) * 2) - 3000
        input_thrust = int(predict[2] * max_thrust)

        ref.game.input = f"{cp[cp_id1].x + offset_x} {cp[cp_id1].y + offset_y} {input_thrust}"
        ref.game_turn()

        if create_gif and i % 2 == 0:
            images.append(plot_current_frame(cp, cp_id1, ref.game.car))

        if ref.game.isDone:
            if create_gif:
                convert_to_gif(f"track_{time.time()}", images)

            # drive_te = int(np.sum(drive_thru_error_list))
            distances = int(np.sum(check_distances) // 3000)
            bonus = 100 if (i + 1) != Constants.MAX_TIME else 0
            return (i + 1) ** 3 - ref.game.currentCheckpoint ** 2 - bonus + distances


def create_net_input(params):
    input_net = [
        min_max_scaler(params["check1_x"], -params["offset"], params["max_x"]),
        min_max_scaler(params["check1_y"], -params["offset"], params["max_y"]),
        min_max_scaler(params["car_x"], -params["offset"], params["max_x"]),
        min_max_scaler(params["car_y"], -params["offset"], params["max_y"]),
        min_max_scaler(params["dist_check1"], 0, params["max_distance"]),
        min_max_scaler(params["dist_check2"], 0, params["max_distance"]),
        min_max_scaler(params["angle_distance_check1"], 0, params["max_distance"]),
        min_max_scaler(params["vx"], -1000, 2000),
        min_max_scaler(params["vy"], -1000, 2000)
    ]
    return input_net
