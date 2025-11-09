import random
from collections import deque
import numpy as np
from case_closed_game import Direction

# Helper: flood fill area (torus-aware)
def _flood_fill_area(game, start_pos):
    W, H = game.board.width, game.board.height
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
            if npos not in visited and game.board.get_cell_state(npos) == 0:
                queue.append(npos)
    return area


def area_maximizer(game, agent, opponent, valid_moves):
    """Choose the move that maximizes reachable area for the agent.

    valid_moves: list of (Direction, can_boost)
    Returns (Direction, use_boost)
    """
    if not valid_moves:
        return agent.direction, False

    best = None
    best_score = -1e9
    head = agent.trail[-1]
    for d, can_boost in valid_moves:
        dx, dy = d.value
        next_pos = ((head[0] + dx) % game.board.width, (head[1] + dy) % game.board.height)
        if can_boost and agent.boosts_remaining > 0:
            second_pos = ((next_pos[0] + dx) % game.board.width, (next_pos[1] + dy) % game.board.height)
            area = _flood_fill_area(game, second_pos)
        else:
            area = _flood_fill_area(game, next_pos)
        # small tie-breaker: prefer moves that increase distance from opponent head
        op_head = opponent.trail[-1]
        dist = np.sqrt(((next_pos[0] - op_head[0]) % game.board.width) ** 2 + ((next_pos[1] - op_head[1]) % game.board.height) ** 2)
        score = area + 0.05 * dist
        if score > best_score:
            best_score = score
            best = (d, can_boost)

    # Decide boost heuristically
    d, can_boost = best
    use_boost = False
    if can_boost and agent.boosts_remaining > 0:
        use_boost = random.random() < 0.3
    return d, use_boost


def random_policy(game, agent, opponent, valid_moves):
    # Choose a random valid move.
    if not valid_moves:
        return agent.direction, False
    d, can_boost = random.choice(valid_moves)
    use_boost = can_boost and (random.random() < 0.25)
    return d, use_boost


def chaser_policy(game, agent, opponent, valid_moves):
    # minimize distance to opponent head
    if not valid_moves:
        return agent.direction, False
    best = None
    best_dist = 1e9
    op_head = opponent.trail[-1]
    head = agent.trail[-1]
    for d, can_boost in valid_moves:
        dx, dy = d.value
        next_pos = ((head[0] + dx) % game.board.width, (head[1] + dy) % game.board.height)
        dist = np.sqrt(((next_pos[0] - op_head[0]) % game.board.width) ** 2 + ((next_pos[1] - op_head[1]) % game.board.height) ** 2)
        if dist < best_dist:
            best_dist = dist
            best = (d, can_boost)
    d, can_boost = best
    use_boost = False
    if can_boost and agent.boosts_remaining > 0:
        use_boost = random.random() < 0.2
    return d, use_boost


def conservative_policy(game, agent, opponent, valid_moves):
    # maximize distance from opponent
    if not valid_moves:
        return agent.direction, False
    best = None
    best_dist = -1
    op_head = opponent.trail[-1]
    head = agent.trail[-1]
    for d, can_boost in valid_moves:
        dx, dy = d.value
        next_pos = ((head[0] + dx) % game.board.width, (head[1] + dy) % game.board.height)
        dist = np.sqrt(((next_pos[0] - op_head[0]) % game.board.width) ** 2 + ((next_pos[1] - op_head[1]) % game.board.height) ** 2)
        if dist > best_dist:
            best_dist = dist
            best = (d, can_boost)
    d, can_boost = best
    use_boost = can_boost and (random.random() < 0.15)
    return d, use_boost

def perimeter_claimer(game, agent, opponent, valid_moves):
    #secure board's exterior edges, prioritize extending trail along longest open boundary
    if not valid_moves:
        return agent.direction, False
    best = None
    best_score = -1e9
    head = agent.trail[-1]
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        dx, dy = d.value
        nx = (head[0] + dx) % W
        ny = (head[1] + dy) % H
        # score edges higher
        edge_bonus = 1.0 if (nx == 0 or ny == 0 or nx == W-1 or ny == H-1) else 0.0
        area = _flood_fill_area(game, (nx, ny))
        score = area + 5.0 * edge_bonus
        if score > best_score:
            best_score = score
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.25)

def escape_artist(game, agent, opponent, valid_moves):
    #ensures movements have multiple exists, avoids corners and dead ends
    if not valid_moves:
        return agent.direction, False
    best = None
    best_score = -1e9
    head = agent.trail[-1]
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        dx, dy = d.value
        nx = (head[0] + dx) % W
        ny = (head[1] + dy) % H
        # count empty neighbors from that position
        empties = 0
        for dd in Direction:
            ex = (nx + dd.value[0]) % W
            ey = (ny + dd.value[1]) % H
            if game.board.get_cell_state((ex, ey)) == 0:
                empties += 1
        score = empties
        # avoid corners
        if (nx == 0 or ny == 0 or nx == W-1 or ny == H-1) and empties < 2:
            score -= 2
        if score > best_score:
            best_score = score
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.2)

def proximity_hunter(game, agent, opponent, valid_moves):
    #rapidly encloses opponent, prioritizes increasing proximity to opponent's head
    return chaser_policy(game, agent, opponent, valid_moves)

def cut_off_specialist(game, agent, opponent, valid_moves):
    #focuses on slicing off opponent's potential paths, aims to block escape routes, prioritizes reducing opponents area
    if not valid_moves:
        return agent.direction, False
    best = None
    best_delta = -1e9
    head = agent.trail[-1]
    op_head = opponent.trail[-1]
    base_op_area = _flood_fill_area(game, op_head)
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        dx, dy = d.value
        nx = (head[0] + dx) % W
        ny = (head[1] + dy) % H
        # approximate: if move is nearer to opponent, prefer it
        dist = np.sqrt(((nx - op_head[0]) % W) ** 2 + ((ny - op_head[1]) % H) ** 2)
        delta = -dist
        if delta > best_delta:
            best_delta = delta
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.25)

def head_to_head_challenger(game, agent, opponent, valid_moves):
    #high risk strategy that gets to opponents head, goal to get one turn away from collision and force opponent to crash
    if not valid_moves:
        return agent.direction, False
    # aggressively minimize distance and prefer moves that place head adjacent to opponent
    op_head = opponent.trail[-1]
    best = None
    best_score = 1e9
    head = agent.trail[-1]
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        dx, dy = d.value
        nx = (head[0] + dx) % W
        ny = (head[1] + dy) % H
        dist = np.sqrt(((nx - op_head[0]) % W) ** 2 + ((ny - op_head[1]) % H) ** 2)
        if dist < best_score:
            best_score = dist
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and agent.boosts_remaining > 0 and random.random() < 0.5)

def path_predictor(game, agent, opponent, valid_moves):
    #aims to create longest continous linear path
    # prefer moves that continue straight if possible, else area_maximizer
    if not valid_moves:
        return agent.direction, False
    # infer last direction
    last_dir = None
    if len(agent.trail) > 1:
        prev = agent.trail[-2]
        head = agent.trail[-1]
        for d in Direction:
            if ((head[0] - prev[0]) % game.board.width, (head[1] - prev[1]) % game.board.height) == d.value:
                last_dir = d
                break
    if last_dir:
        for d, can_boost in valid_moves:
            if d == last_dir:
                return d, (can_boost and random.random() < 0.4)
    return area_maximizer(game, agent, opponent, valid_moves)

def mirror_maneuverer(game, agent, opponent, valid_moves):
    #mimics opponents moves with slight delay, aims to maintain equal distance from opponent
    if not valid_moves:
        return agent.direction, False
    # try to pick move that mirrors opponent's last move
    if len(opponent.trail) > 1:
        prev = opponent.trail[-2]
        head = opponent.trail[-1]
        desired = None
        for d in Direction:
            if ((head[0] - prev[0]) % game.board.width, (head[1] - prev[1]) % game.board.height) == d.value:
                desired = d
                break
        if desired:
            for d, can_boost in valid_moves:
                if d == desired:
                    return d, False
    # fallback
    return conservative_policy(game, agent, opponent, valid_moves)

def opportunistic_booster(game, agent, opponent, valid_moves):
    #saves boost for game winning or game changing moments, uses boost only when it can secure victory or avoid imminent death
    if not valid_moves:
        return agent.direction, False
    # if any move leads to large area gain, use boost
    best = None
    best_area = -1
    head = agent.trail[-1]
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        dx, dy = d.value
        nx = (head[0] + dx) % W
        ny = (head[1] + dy) % H
        area = _flood_fill_area(game, (nx, ny))
        if area > best_area:
            best_area = area
            best = (d, can_boost)
    d, can_boost = best
    use_boost = False
    if can_boost and agent.boosts_remaining > 0 and best_area > (W*H)*0.05:
        use_boost = True
    return d, use_boost

def grid_divider(game, agent, opponent, valid_moves):
    #divides board into controlled sections, aims to partition board and limit opponents movement options
    # heuristically prefer moves that increase perimeter of controlled area
    if not valid_moves:
        return agent.direction, False
    best = None
    best_score = -1e9
    head = agent.trail[-1]
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        dx, dy = d.value
        nx = (head[0] + dx) % W
        ny = (head[1] + dy) % H
        # perimeter approx: count empty neighbors adjacent to our trail
        score = 0
        for dd in Direction:
            ex = (nx + dd.value[0]) % W
            ey = (ny + dd.value[1]) % H
            if game.board.get_cell_state((ex, ey)) == 0:
                score += 1
        if score > best_score:
            best_score = score
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.2)

def min_risk_travel(game, agent, opponent, valid_moves):
    #evaluates risk associated with each move, prioritizes moves that minimize potential threats from opponent
    if not valid_moves:
        return agent.direction, False
    best = None
    best_score = 1e9
    head = agent.trail[-1]
    op_head = opponent.trail[-1]
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        dx, dy = d.value
        nx = (head[0] + dx) % W
        ny = (head[1] + dy) % H
        # risk = closeness to opponent + number of occupied neighbors
        dist = np.sqrt(((nx - op_head[0]) % W) ** 2 + ((ny - op_head[1]) % H) ** 2)
        occ_neighbors = 0
        for dd in Direction:
            ex = (nx + dd.value[0]) % W
            ey = (ny + dd.value[1]) % H
            if game.board.get_cell_state((ex, ey)) != 0:
                occ_neighbors += 1
        score = occ_neighbors - 0.5 * dist
        if score < best_score:
            best_score = score
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.15)

def corner_specialist(game, agent, opponent, valid_moves):
    #expertly navigates corners and tight spaces, aims to utilize corners for strategic advantage while avoiding entrapment, focuses on staying near corners
    if not valid_moves:
        return agent.direction, False
    corners = [(0,0),(0,game.board.height-1),(game.board.width-1,0),(game.board.width-1,game.board.height-1)]
    head = agent.trail[-1]
    best = None
    best_score = -1e9
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        dx, dy = d.value
        nx = (head[0] + dx) % W
        ny = (head[1] + dy) % H
        # score proximity to nearest corner but penalize low escape options
        dist_corner = min([np.sqrt(((nx-cx)%W)**2 + ((ny-cy)%H)**2) for cx,cy in corners])
        empties = sum(1 for dd in Direction if game.board.get_cell_state(((nx+dd.value[0])%W, (ny+dd.value[1])%H))==0)
        score = -dist_corner + 0.5*empties
        if score > best_score:
            best_score = score
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.2)

def opponent_trapper(game, agent, opponent, valid_moves):
    #focuses on maximizing the difference in our area vs opponent area, tries to implement traps by predicting opponent behavior
    # combine area_maximizer with cut_off logic
    if not valid_moves:
        return agent.direction, False
    # prefer moves that increase our area while reducing opponent reach
    best = None
    best_score = -1e9
    head = agent.trail[-1]
    op_head = opponent.trail[-1]
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        dx, dy = d.value
        nx = (head[0] + dx) % W
        ny = (head[1] + dy) % H
        my_area = _flood_fill_area(game, (nx, ny))
        op_area = _flood_fill_area(game, op_head)
        score = (my_area - op_area)
        if score > best_score:
            best_score = score
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.25)

def torus_explorer(game, agent, opponent, valid_moves):
    #utilizes toroidal nature of board to create unexpected movement patterns, aims to confuse opponent by wrapping around edges
    if not valid_moves:
        return agent.direction, False
    # prefer moves that use wrap-around (i.e., lead to x==0 or x==W-1 or y==0 or y==H-1) but actually wrapping
    head = agent.trail[-1]
    W, H = game.board.width, game.board.height
    wrap_moves = []
    other_moves = []
    for d, can_boost in valid_moves:
        nx = (head[0] + d.value[0]) % W
        ny = (head[1] + d.value[1]) % H
        if (head[0] + d.value[0]) not in range(0, W) or (head[1] + d.value[1]) not in range(0, H):
            wrap_moves.append((d, can_boost))
        else:
            other_moves.append((d, can_boost))
    if wrap_moves:
        d, can_boost = random.choice(wrap_moves)
        return d, (can_boost and random.random() < 0.3)
    return random_policy(game, agent, opponent, valid_moves)

def collision_avoider(game, agent, opponent, valid_moves):
    #prioritizes moves that maximize distance from opponent and avoid potential collision paths, focuses on survival and longevity, prioritizes maximizing amount of turns, considers spiraling
    return conservative_policy(game, agent, opponent, valid_moves)

def opponent_predictor(game, agent, opponent, valid_moves):
    #cuts off opponent when it predicts their next move, aims to anticipate opponent's strategy and counteract it effectively
    # predict opponent's next move using their valid moves and try to move into that cell
    if not valid_moves:
        return agent.direction, False
    op_head = opponent.trail[-1]
    preds = []
    W, H = game.board.width, game.board.height
    for d in Direction:
        nx = (op_head[0] + d.value[0]) % W
        ny = (op_head[1] + d.value[1]) % H
        if game.board.get_cell_state((nx, ny)) == 0:
            preds.append((nx, ny))
    head = agent.trail[-1]
    for d, can_boost in valid_moves:
        nx = (head[0] + d.value[0]) % W
        ny = (head[1] + d.value[1]) % H
        if (nx, ny) in preds:
            return d, (can_boost and random.random() < 0.4)
    return cut_off_specialist(game, agent, opponent, valid_moves)

def wall_hugger(game, agent, opponent, valid_moves):
    #stays close to walls and edges of the board, aims to minimize exposure to open areas while maximizing control over board perimeter
    if not valid_moves:
        return agent.direction, False
    W, H = game.board.width, game.board.height
    head = agent.trail[-1]
    best = None
    best_score = -1e9
    for d, can_boost in valid_moves:
        nx = (head[0] + d.value[0]) % W
        ny = (head[1] + d.value[1]) % H
        dist_to_wall = min(nx, ny, W-1-nx, H-1-ny)
        score = -dist_to_wall
        if score > best_score:
            best_score = score
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.2)

def spiral_navigator(game, agent, opponent, valid_moves):
    #creates spiral movement patterns to maximize area coverage, aims to systematically cover the board while avoiding opponent's trail
    # prefer turning in one direction consistently to approximate spiral
    if not valid_moves:
        return agent.direction, False
    if len(agent.trail) > 1:
        prev = agent.trail[-2]
        head = agent.trail[-1]
        cur_dir = None
        for d in Direction:
            if ((head[0]-prev[0])%game.board.width, (head[1]-prev[1])%game.board.height) == d.value:
                cur_dir = d
                break
    else:
        cur_dir = None
    order = list(Direction)
    if cur_dir and cur_dir in order:
        idx = order.index(cur_dir)
        candidates = [order[(idx+1)%4], order[idx], order[(idx-1)%4], order[(idx+2)%4]]
        for pref in candidates:
            for d, can_boost in valid_moves:
                if d == pref:
                    return d, (can_boost and random.random() < 0.2)
    return area_maximizer(game, agent, opponent, valid_moves)

def zigzag_master(game, agent, opponent, valid_moves):
    #employs zigzag movement patterns to confuse opponent, aims to create unpredictable paths that are hard to follow or predict
    if not valid_moves:
        return agent.direction, False
    if len(agent.trail) > 2:
        p2 = agent.trail[-3]
        p1 = agent.trail[-2]
        p0 = agent.trail[-1]
        dx1 = (p1[0]-p2[0])%game.board.width
        dy1 = (p1[1]-p2[1])%game.board.height
        dx2 = (p0[0]-p1[0])%game.board.width
        dy2 = (p0[1]-p1[1])%game.board.height
        for d, can_boost in valid_moves:
            if d.value != (dx2, dy2):
                return d, (can_boost and random.random() < 0.3)
    return random_policy(game, agent, opponent, valid_moves)

def snake_charmer(game, agent, opponent, valid_moves):
    #creates winding paths that are difficult for opponent to navigate, aims to create complex trails that limit opponent's movement options
    if not valid_moves:
        return agent.direction, False
    best = None
    best_score = -1e9
    head = agent.trail[-1]
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        nx = (head[0]+d.value[0])%W
        ny = (head[1]+d.value[1])%H
        empties = sum(1 for dd in Direction if game.board.get_cell_state(((nx+dd.value[0])%W, (ny+dd.value[1])%H))==0)
        straight_penalty = 0
        if len(agent.trail)>1:
            prev = agent.trail[-2]
            if ((head[0]-prev[0])%W, (head[1]-prev[1])%H) == d.value:
                straight_penalty = 1
        score = empties - straight_penalty
        if score > best_score:
            best_score = score
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.2)

def perimeter_patroller(game, agent, opponent, valid_moves):
    #regularly patrols the edges of the board to maintain control, aims to secure the perimeter while monitoring opponent's movements
    if not valid_moves:
        return agent.direction, False
    W, H = game.board.width, game.board.height
    head = agent.trail[-1]
    best = None
    best_score = -1e9
    for d, can_boost in valid_moves:
        nx = (head[0]+d.value[0])%W
        ny = (head[1]+d.value[1])%H
        edge_bonus = 1.0 if (nx==0 or ny==0 or nx==W-1 or ny==H-1) else 0.0
        empties = sum(1 for dd in Direction if game.board.get_cell_state(((nx+dd.value[0])%W,(ny+dd.value[1])%H))==0)
        score = edge_bonus*5 + empties
        if score > best_score:
            best_score = score
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.2)

def center_controller(game, agent, opponent, valid_moves):
    #focuses on controlling the center area of the board, aims to dominate the central region to maximize movement options and area control
    if not valid_moves:
        return agent.direction, False
    W, H = game.board.width, game.board.height
    cx, cy = W//2, H//2
    best = None
    best_score = 1e9
    head = agent.trail[-1]
    for d, can_boost in valid_moves:
        nx = (head[0]+d.value[0])%W
        ny = (head[1]+d.value[1])%H
        dist = np.sqrt(((nx-cx)%W)**2 + ((ny-cy)%H)**2)
        if dist < best_score:
            best_score = dist
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.2)

def spiral_trapper(game, agent, opponent, valid_moves):
    #creates spiral patterns that lead opponent into traps, aims to lure opponent into enclosed areas where they can be easily cut off
    if not valid_moves:
        return agent.direction, False
    op_head = opponent.trail[-1]
    candidates = []
    head = agent.trail[-1]
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        nx = (head[0]+d.value[0])%W
        ny = (head[1]+d.value[1])%H
        dist = np.sqrt(((nx-op_head[0])%W)**2 + ((ny-op_head[1])%H)**2)
        area = _flood_fill_area(game,(nx,ny))
        score = -dist + 0.1*area
        candidates.append((score,d,can_boost))
    candidates.sort(reverse=True,key=lambda x:x[0])
    d, can_boost = candidates[0][1], candidates[0][2]
    return d, (can_boost and random.random() < 0.3)

def zigzag_trapper(game, agent, opponent, valid_moves):
    #uses zigzag patterns to lead opponent into confined spaces, aims to create unpredictable paths that funnel opponent into traps
    if not valid_moves:
        return agent.direction, False
    d, use_boost = zigzag_master(game, agent, opponent, valid_moves)
    return d, use_boost

def snake_trapper(game, agent, opponent, valid_moves):
    #creates winding paths that guide opponent into traps, aims to manipulate opponent's movement into areas where they can be easily cut off
    if not valid_moves:
        return agent.direction, False
    best = None
    best_score = -1e9
    head = agent.trail[-1]
    op_head = opponent.trail[-1]
    W, H = game.board.width, game.board.height
    for d, can_boost in valid_moves:
        nx = (head[0]+d.value[0])%W
        ny = (head[1]+d.value[1])%H
        my_area = _flood_fill_area(game,(nx,ny))
        op_area = _flood_fill_area(game,op_head)
        score = my_area - op_area
        if score > best_score:
            best_score = score
            best = (d, can_boost)
    d, can_boost = best
    return d, (can_boost and random.random() < 0.25)

def get_all_styles():
    """Return available opponent style callables in a deterministic order."""
    return [
        area_maximizer,
        random_policy,
        chaser_policy,
        conservative_policy,
        perimeter_claimer,
        escape_artist,
        proximity_hunter,
        cut_off_specialist,
        head_to_head_challenger,
        path_predictor,
        mirror_maneuverer,
        opportunistic_booster,
        grid_divider,
        min_risk_travel,
        corner_specialist,
        opponent_trapper,
        torus_explorer,
        collision_avoider,
        opponent_predictor,
        wall_hugger,
        spiral_navigator,
        zigzag_master,
        snake_charmer,
        perimeter_patroller,
        center_controller,
        spiral_trapper,
        zigzag_trapper,
        snake_trapper,
    ]


__all__ = [
    'area_maximizer',
    'random_policy',
    'chaser_policy',
    'conservative_policy',
    'perimeter_claimer',
    'escape_artist',
    'proximity_hunter',
    'cut_off_specialist',
    'head_to_head_challenger',
    'path_predictor',
    'mirror_maneuverer',
    'opportunistic_booster',
    'grid_divider',
    'min_risk_travel',
    'corner_specialist',
    'opponent_trapper',
    'torus_explorer',
    'collision_avoider',
    'opponent_predictor',
    'wall_hugger',
    'spiral_navigator',
    'zigzag_master',
    'snake_charmer',
    'perimeter_patroller',
    'center_controller',
    'spiral_trapper',
    'zigzag_trapper',
    'snake_trapper',
    'get_all_styles'
]
