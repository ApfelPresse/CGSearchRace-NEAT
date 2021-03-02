import math

from CGSearchRace.Constants import Constants
from CGSearchRace.Referee import Referee
from CGSearchRace.Tracks import tracks

from visualize_run import plot_current_frame, convert_to_gif


def min_max_scaler(value: float, min_value: float, max_value: float, range_from: float = -6,
                   range_to: float = 6):
    width = range_to - range_from
    return (value - min_value) / (max_value - min_value) * width + range_from


def simulate(net, create_gif=False):
    score = 0
    for i, track in enumerate(tracks):
        ref = Referee(track)

        score_, looses = run_track(net, ref, create_gif)
        score += score_

        diff_checkpoints = ref.game.totalCheckpoints - ref.game.currentCheckpoint
        score += diff_checkpoints * 100

        if looses:
            score += Constants.MAX_TIME * 2

    return score


def run_track(net, ref, create_gif):
    offset = 4000
    max_x = Constants.Width + offset
    max_y = Constants.Height + offset
    max_thrust = Constants.CAR_MAX_THRUST
    max_time = Constants.MAX_TIME

    images = []

    looses = False
    score = 0
    for i in range(max_time):
        current_checkpoint = ref.game.get_next_checkpoint_id()
        check_x = ref.game.checkpoints[current_checkpoint].x
        check_y = ref.game.checkpoints[current_checkpoint].y

        current_checkpoint = ref.game.get_next_checkpoint_id(2)
        check_x_2 = ref.game.checkpoints[current_checkpoint].x
        check_y_2 = ref.game.checkpoints[current_checkpoint].y

        current_checkpoint = ref.game.get_next_checkpoint_id(3)
        check_x_3 = ref.game.checkpoints[current_checkpoint].x
        check_y_3 = ref.game.checkpoints[current_checkpoint].y

        angle = ref.game.car.angle

        vx = ref.game.car.vx
        vy = ref.game.car.vx

        distance_next_checkpoint = math.sqrt(((check_x - ref.game.car.x) ** 2) + ((check_y - ref.game.car.y) ** 2))
        max_distance = math.sqrt(max_x ** 2 + max_y ** 2)

        input_net = create_net_input(angle, check_x, check_x_2, check_x_3, check_y, check_y_2, check_y_3,
                                     distance_next_checkpoint, max_distance, max_x, max_y, offset, ref, vx, vy)

        predict = net.activate(input_net)

        input_x = int(predict[0] * max_x)
        input_y = int(predict[1] * max_y)
        input_thrust = int(predict[2] * max_thrust)

        ref.game.input = f"{input_x} {input_y} {input_thrust}"
        ref.game_turn()

        if create_gif and i % 2 == 0:
            images.append(plot_current_frame(ref.game.checkpoints, ref.game.get_next_checkpoint_id(), ref.game.car))

        # exit earlier
        if i > (ref.game.currentCheckpoint + 1) * 35:
            looses = True
            score += i
            break

        if ref.game.isDone:
            if i == max_time - 1:
                looses = True
            score += i
            break

    if create_gif:
        convert_to_gif(f"track", images)
        exit(0)
    return score, looses


def create_net_input(angle, check_x, check_x_2, check_x_3, check_y, check_y_2, check_y_3, distance_next_checkpoint,
                     max_distance, max_x, max_y, offset, ref, vx, vy):
    input_net = [
        min_max_scaler(check_x, -offset, max_x),
        min_max_scaler(check_y, -offset, max_y),
        min_max_scaler(check_x_2, -offset, max_x),
        min_max_scaler(check_y_2, -offset, max_y),
        min_max_scaler(check_x_3, -offset, max_x),
        min_max_scaler(check_y_3, -offset, max_y),
        min_max_scaler(ref.game.car.x, -offset, max_x),
        min_max_scaler(ref.game.car.y, -offset, max_y),
        min_max_scaler(distance_next_checkpoint, 0, max_distance),
        min_max_scaler(angle, 0, (2 * Constants.PI)),
        min_max_scaler(vx, -1000, 2000),
        min_max_scaler(vy, -1000, 2000),
    ]
    return input_net
