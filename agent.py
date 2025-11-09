import math
import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import random
from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "Participant67"
AGENT_NAME = "Mayank's_Agent"

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        op_agent = GLOBAL_GAME.agent1 if player_number == 2 else GLOBAL_GAME.agent2
        my_boosts_remaining = my_agent.boosts_remaining
        op_boosts_remaining = op_agent.boosts_remaining
   
    # -----------------begin-code------------------
    # We'll try to load a saved DQN policy (Option-B CNN) and use it to pick a direction.
    # If loading or inference fails, fall back to the previous simple heuristic.
    try:
        import torch
        import numpy as _np
        import playstyles as _ps

        # New QNetwork (Option B) matching rl_trainer.py
        class QNetwork(torch.nn.Module):
            def __init__(self, h, w, num_channels=11, num_actions=4):
                super(QNetwork, self).__init__()
                self.conv1 = torch.nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
                self.bn1 = torch.nn.BatchNorm2d(32)
                self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
                self.bn2 = torch.nn.BatchNorm2d(64)
                self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
                self.bn3 = torch.nn.BatchNorm2d(128)
                self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.head = torch.nn.Linear(128, num_actions)

            def forward(self, x):
                x = torch.relu(self.bn1(self.conv1(x)))
                x = torch.relu(self.bn2(self.conv2(x)))
                x = torch.relu(self.bn3(self.conv3(x)))
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                return self.head(x)

        # Build state tensor similar to rl_trainer.get_current_state(), including style channels
        board = GLOBAL_GAME.board.grid
        BOARD_HEIGHT = GLOBAL_GAME.board.height
        BOARD_WIDTH = GLOBAL_GAME.board.width
        H, W = BOARD_HEIGHT, BOARD_WIDTH

        my_pos = my_agent.trail[-1]
        op_pos = op_agent.trail[-1]

        style_list = _ps.get_all_styles()
        n_styles = len(style_list)
        state_np = _np.zeros((11 + n_styles, H, W), dtype=_np.float32)

        # Trails
        for y in range(H):
            for x in range(W):
                cell = board[y][x]
                if cell == (1 if player_number == 1 else 2):
                    state_np[0, y, x] = 1.0
                elif cell == (2 if player_number == 1 else 1):
                    state_np[1, y, x] = 1.0

        # Heads
        my_x, my_y = my_pos
        op_x, op_y = op_pos
        state_np[0, my_y, my_x] = 0.0
        state_np[2, my_y, my_x] = 1.0
        state_np[1, op_y, op_x] = 0.0
        state_np[3, op_y, op_x] = 1.0

        # Boost availability
        state_np[4, :, :] = my_agent.boosts_remaining / 5.0

        # Turn count
        turn_count = state.get("turn_count", GLOBAL_GAME.turns)
        state_np[5, :, :] = turn_count / 200.0

        # Flood-fill areas helper
        def _flood_fill_area_local(start_pos):
            visited = set()
            queue = [start_pos]
            area = 0
            while queue:
                pos = queue.pop(0)
                if pos in visited:
                    continue
                visited.add(pos)
                area += 1
                for d in Direction:
                    dx, dy = d.value
                    npos = ((pos[0] + dx) % W, (pos[1] + dy) % H)
                    if npos not in visited and board[npos[1]][npos[0]] == 0:
                        queue.append(npos)
            return area

        state_np[6, :, :] = _flood_fill_area_local(my_pos) / float(H * W)
        state_np[7, :, :] = _flood_fill_area_local(op_pos) / float(H * W)

        # Distance between heads
        dist = math.sqrt(((my_x - op_x) % W) ** 2 + ((my_y - op_y) % H) ** 2)
        max_dist = math.sqrt(W ** 2 + H ** 2)
        state_np[8, :, :] = dist / max_dist

        # Direction history
        def _encode_dir(agent, channel_idx):
            if len(agent.trail) > 1:
                prev = agent.trail[-2]
                curr = agent.trail[-1]
                dx = (curr[0] - prev[0]) % W
                dy = (curr[1] - prev[1]) % H
                if dx == 1 and dy == 0:
                    dir_idx = 3
                elif dx == W - 1 and dy == 0:
                    dir_idx = 2
                elif dx == 0 and dy == 1:
                    dir_idx = 1
                elif dx == 0 and dy == H - 1:
                    dir_idx = 0
                else:
                    return
                state_np[channel_idx, :, :] = (dir_idx + 1) / 4.0

        _encode_dir(my_agent, 9)
        _encode_dir(op_agent, 10)

        # Style channels: use lightweight detector to produce a soft distribution
        try:
            # compute probabilities similar to rl_trainer._detect_opponent_style
            probs = _np.zeros(n_styles, dtype=_np.float32)
            my_area = _flood_fill_area_local(my_pos)
            op_area = _flood_fill_area_local(op_pos)
            if op_area > my_area * 1.3:
                for i, fn in enumerate(style_list):
                    if fn.__name__ == 'area_maximizer':
                        probs[i] = 0.9
                        break
            if len(op_agent.trail) >= 3:
                d0 = math.hypot((op_agent.trail[-1][0] - my_agent.trail[-1][0]) % W, (op_agent.trail[-1][1] - my_agent.trail[-1][1]) % H)
                d1 = math.hypot((op_agent.trail[-2][0] - my_agent.trail[-1][0]) % W, (op_agent.trail[-2][1] - my_agent.trail[-1][1]) % H)
                if d0 < d1:
                    for i, fn in enumerate(style_list):
                        if fn.__name__ == 'chaser_policy':
                            probs[i] = max(probs[i], 0.8)
                            break
            border_count = sum(1 for (x, y) in op_agent.trail if x in (0, W-1) or y in (0, H-1))
            if border_count >= max(3, len(op_agent.trail)//3):
                for i, fn in enumerate(style_list):
                    if fn.__name__ == 'wall_hugger' or fn.__name__ == 'perimeter_claimer':
                        probs[i] = max(probs[i], 0.7)
                        break

            if probs.sum() == 0:
                probs += 1.0 / float(max(1, n_styles))
            else:
                probs = probs + 0.05
                probs = probs / probs.sum()

            for i, v in enumerate(probs.tolist()):
                state_np[11 + i, :, :] = float(v)
        except Exception:
            # leave style channels zero if detection fails
            pass

        # Symmetry handling: the saved model was trained with the agent starting top-left.
        # If current head is in bottom-right quadrant, rotate the observation 180deg
        # so the network sees a similar frame.
        flip_needed = (my_x >= W // 2 and my_y >= H // 2)
        if flip_needed:
            # rotate 180 = flip H and W
            state_np = _np.flip(_np.flip(state_np, axis=1), axis=2).copy()

        input_tensor = torch.from_numpy(state_np.copy()).unsqueeze(0)

        num_input_channels = 11 + n_styles
        model = QNetwork(H, W, num_channels=num_input_channels, num_actions=4)
        model.eval()

        model_path = os.path.join(os.path.dirname(__file__), 'dqn_policy_net.pth')
        if os.path.exists(model_path):
            try:
                state_d = torch.load(model_path, map_location='cpu')
                # Attempt to load, and adapt conv1 weights if channel mismatch
                try:
                    model.load_state_dict(state_d)
                except RuntimeError:
                    # try to adapt conv1 weights if present
                    sd = dict(state_d)
                    policy_sd = model.state_dict()
                    conv_key = None
                    for k in sd.keys():
                        if 'conv1.weight' in k:
                            conv_key = k
                            break
                    if conv_key is not None and conv_key in policy_sd:
                        w_loaded = sd[conv_key]
                        w_expected = policy_sd[conv_key]
                        if w_loaded.shape != w_expected.shape:
                            # expand loaded channels by repeating or copying
                            new_w = w_expected.clone()
                            min_in = min(w_loaded.shape[1], new_w.shape[1])
                            new_w[:, :min_in, :, :] = w_loaded[:, :min_in, :, :]
                            if new_w.shape[1] > w_loaded.shape[1]:
                                # fill remaining channels with the mean of loaded weights
                                mean_ch = w_loaded.mean(dim=1, keepdim=True)
                                for c in range(w_loaded.shape[1], new_w.shape[1]):
                                    new_w[:, c:c+1, :, :] = mean_ch
                            sd[conv_key] = new_w
                    # fill other missing keys from current model
                    for k, v in policy_sd.items():
                        if k not in sd:
                            sd[k] = v
                    model.load_state_dict(sd)
            except Exception:
                # model load failed; fall back to heuristic
                raise
        else:
            raise FileNotFoundError

        with torch.no_grad():
            q_values = model(input_tensor).squeeze(0).numpy()

        # Determine valid actions
        SAFE = []
        dir_to_idx = {Direction.UP:0, Direction.DOWN:1, Direction.LEFT:2, Direction.RIGHT:3}
        for d in Direction:
            dx, dy = d.value
            nx = (my_pos[0] + dx) % W
            ny = (my_pos[1] + dy) % H
            if board[ny][nx] == 0:
                reversible = False
                if len(my_agent.trail) > 1:
                    prev = my_agent.trail[-2]
                    if ((prev[0] - my_pos[0]) % W, (prev[1] - my_pos[1]) % H) == d.value:
                        reversible = True
                if not reversible:
                    SAFE.append(d)

        if SAFE:
            mask = [-1e9] * 4
            for d in SAFE:
                mask[dir_to_idx[d]] = float(q_values[dir_to_idx[d]])

            # bias with lightweight opponent detector + counter policy
            try:
                def _detect_opponent_style_simple():
                    my_a = _flood_fill_area_local(my_pos)
                    op_a = _flood_fill_area_local(op_pos)
                    if op_a > my_a * 1.3:
                        return 'area_maximizer'
                    if len(op_agent.trail) >= 3:
                        d0 = math.hypot((op_agent.trail[-1][0] - my_agent.trail[-1][0]) % W, (op_agent.trail[-1][1] - my_agent.trail[-1][1]) % H)
                        d1 = math.hypot((op_agent.trail[-2][0] - my_agent.trail[-1][0]) % W, (op_agent.trail[-2][1] - my_agent.trail[-1][1]) % H)
                        if d0 < d1:
                            return 'chaser_policy'
                    border_count = sum(1 for (x, y) in op_agent.trail if x in (0, W-1) or y in (0, H-1))
                    if border_count >= max(3, len(op_agent.trail)//3):
                        return 'wall_hugger'
                    return 'random_policy'

                detected = _detect_opponent_style_simple()
                counter_map = {
                    'chaser_policy': _ps.conservative_policy,
                    'area_maximizer': _ps.cut_off_specialist,
                    'wall_hugger': _ps.center_controller,
                    'random_policy': _ps.area_maximizer,
                }
                counter_fn = counter_map.get(detected)
                if counter_fn is not None:
                    my_valid_moves = []
                    for d in SAFE:
                        dx, dy = d.value
                        nx = (my_pos[0] + dx) % W
                        ny = (my_pos[1] + dy) % H
                        can_boost_local = False
                        if my_agent.boosts_remaining > 0:
                            nx2 = (nx + dx) % W
                            ny2 = (ny + dy) % H
                            if board[ny2][nx2] == 0:
                                can_boost_local = True
                        my_valid_moves.append((d, can_boost_local))
                    rec_dir, rec_boost = counter_fn(GLOBAL_GAME, my_agent, op_agent, my_valid_moves)
                    if rec_dir in dir_to_idx:
                        bias_idx = dir_to_idx[rec_dir]
                        mask[bias_idx] = mask[bias_idx] + 2.0
            except Exception:
                pass

            best_idx = int(_np.argmax(mask))
            inv_map = {0:Direction.UP,1:Direction.DOWN,2:Direction.LEFT,3:Direction.RIGHT}
            chosen_dir = inv_map[best_idx]

            # if we rotated the input, rotate the chosen action back (180deg)
            if flip_needed:
                flip_map = {Direction.UP:Direction.DOWN, Direction.DOWN:Direction.UP, Direction.LEFT:Direction.RIGHT, Direction.RIGHT:Direction.LEFT}
                chosen_dir = flip_map[chosen_dir]

            # decide boost
            use_boost_flag = False
            if my_agent.boosts_remaining > 0:
                dx, dy = chosen_dir.value
                nx2 = (my_pos[0] + dx) % W
                ny2 = (my_pos[1] + dy) % H
                nx3 = (nx2 + dx) % W
                ny3 = (ny2 + dy) % H
                if board[ny2][nx2] == 0 and board[ny3][nx3] == 0 and (random.random() < 0.35 or turn_count > 150):
                    use_boost_flag = True

            DIRECTION_MAP = {Direction.UP: "UP", Direction.DOWN: "DOWN", Direction.LEFT: "LEFT", Direction.RIGHT: "RIGHT"}
            move = DIRECTION_MAP[chosen_dir] + (":BOOST" if use_boost_flag else "")
            return jsonify({"move": move}), 200
        else:
            raise RuntimeError

    except Exception:
        # Fallback: original simple heuristic (keep previous behaviour)
        #variables
        my_pos = my_agent.trail[-1]
        op_pos = op_agent.trail[-1]
        board = GLOBAL_GAME.board.grid
        BOARD_HEIGHT = GLOBAL_GAME.board.height
        BOARD_WIDTH = GLOBAL_GAME.board.width
        MOVE_VAL = {
            Direction.UP: (0,-1),
            Direction.DOWN: (0,1),
            Direction.LEFT: (-1,0),
            Direction.RIGHT: (1,0)
        }
        EMPTY_CELL = 0
        turn_count = state.get("turn_count", 0)
        DIRECTION_MAP = {
            Direction.UP: "UP",
            Direction.DOWN : "DOWN",
            Direction.LEFT : "LEFT",
            Direction.RIGHT : "RIGHT"
        }

        def is_valid_move(pos, direction):
            x, y = pos
            if direction == Direction.UP:
                return board[(y - 1) % BOARD_HEIGHT][x] == EMPTY_CELL
            if direction == Direction.DOWN:
                return board[(y + 1) % BOARD_HEIGHT][x] == EMPTY_CELL
            if direction == Direction.LEFT:
                return board[y][(x - 1) % BOARD_WIDTH] == EMPTY_CELL
            if direction == Direction.RIGHT:
                return board[y][(x + 1) % BOARD_WIDTH] == EMPTY_CELL

        def get_safe_directions(pos):
            safe_dirs = []
            for direction in Direction:
                if is_valid_move(pos, direction):
                    safe_dirs.append(DIRECTION_MAP[direction])
            return safe_dirs

        if len(get_safe_directions(my_pos)) > 0:
            move = get_safe_directions(my_pos)[0]
        else:
            move = "RIGHT"

        return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
