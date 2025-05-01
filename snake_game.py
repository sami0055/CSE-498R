import random


class SnakeGame:
    """
    A simple Snake game logic for NEAT training. The snake moves on a grid of given width/height.
    The state vector includes danger indicators (straight/right/left), food direction, and heading.
    """

    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        # Initialize snake in the center, heading right
        cx, cy = self.width // 2, self.height // 2
        self.snake = [[cx-2, cy], [cx-1, cy], [cx, cy]]  # tail..head
        self.direction = 1  # 0=Up,1=Right,2=Down,3=Left
        self.spawn_food()
        self.score = 0
        self.done = False

    def spawn_food(self):
        # Place food at random empty cell
        while True:
            fx = random.randrange(0, self.width)
            fy = random.randrange(0, self.height)
            if [fx, fy] not in self.snake:
                self.food = [fx, fy]
                break

    def get_state(self):
        # Returns 11-dimensional state vector as floats
        head_x, head_y = self.snake[-1]
        # Direction vectors for Up, Right, Down, Left
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        # Compute danger flags
        danger = []
        for turn in (0, 1, -1):  # straight, right, left
            dir_index = (self.direction + turn) % 4
            dx, dy = dirs[dir_index]
            nx, ny = head_x + dx, head_y + dy
            # Check wall collision
            hit_wall = not (0 <= nx < self.width and 0 <= ny < self.height)
            # Check self collision (allowing tail to vacate if not eating)
            will_hit_self = [nx, ny] in self.snake[:-1]
            danger.append(1.0 if (hit_wall or will_hit_self) else 0.0)
        # Food direction flags (up, right, down, left)
        food_dx = self.food[0] - head_x
        food_dy = self.food[1] - head_y
        food_up = 1.0 if food_dy < 0 else 0.0
        food_down = 1.0 if food_dy > 0 else 0.0
        food_right = 1.0 if food_dx > 0 else 0.0
        food_left = 1.0 if food_dx < 0 else 0.0
        # Current heading one-hot
        dir_up = 1.0 if self.direction == 0 else 0.0
        dir_right = 1.0 if self.direction == 1 else 0.0
        dir_down = 1.0 if self.direction == 2 else 0.0
        dir_left = 1.0 if self.direction == 3 else 0.0
        # Compose state vector
        state = danger + [food_up, food_right, food_down,
                          food_left] + [dir_up, dir_right, dir_down, dir_left]
        return state

    def step(self, action):
        """
        Apply action to the game:
        action=0: go straight; 1: turn right; 2: turn left.
        Returns (done, ate) indicating if game ended or food eaten.
        """
        if self.done:
            return True, False

        # Determine new direction
        if action == 1:   # turn right
            self.direction = (self.direction + 1) % 4
        elif action == 2:  # turn left
            self.direction = (self.direction - 1) % 4
        # Move snake
        head_x, head_y = self.snake[-1]
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = dirs[self.direction]
        new_head = [head_x + dx, head_y + dy]
        # Check for wall collision
        if (new_head[0] < 0 or new_head[0] >= self.width or
                new_head[1] < 0 or new_head[1] >= self.height):
            self.done = True
            return True, False
        # Check for self collision
        if new_head in self.snake:
            # If eating, the tail won't move and colliding is fatal
            eating = (new_head == self.food)
            if eating or new_head != self.snake[0]:
                self.done = True
                return True, False
        # Move snake body
        self.snake.append(new_head)
        ate = (new_head == self.food)
        if ate:
            self.score += 1
            self.spawn_food()
        else:
            # Remove tail if not eating
            self.snake.pop(0)
        return False, ate
