import numpy as np
from Box2D import *
from Box2D.b2 import *


# this is the interface to pybox2d
class b2WorldInterface:
    def __init__(self):
        self.world = b2World(gravity=(0.0, 0.0), doSleep=True)
        self.TARGET_FPS = 100
        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.VEL_ITERS, self.POS_ITERS = 10, 10
        self.bodies = []

    def add_bodies(self, new_bodies):
        """ add a single b2Body or list of b2Bodies to the world"""
        if type(new_bodies) == list:
            self.bodies += new_bodies
        else:
            self.bodies.append(new_bodies)

    def step(self, show_display=True, idx=0):
        self.world.Step(self.TIME_STEP, self.VEL_ITERS, self.POS_ITERS)


class end_effector:
    def __init__(self, b2world_interface, init_pos, base, init_angle, hand_shape="rectangle", hand_size=(0.3, 1)):
        world = b2world_interface.world
        self.hand = world.CreateDynamicBody(position=init_pos, angle=init_angle)
        self.hand_shape = hand_shape
        self.hand_size = hand_size
        # forceunit for circle and rect
        if hand_shape == "rectangle":
            rshape = b2PolygonShape(box=hand_size)
            self.forceunit = 30.0
        elif hand_shape == "circle":
            rshape = b2CircleShape(radius=hand_size)
            self.forceunit = 100.0
        elif hand_shape == "polygon":
            rshape = b2PolygonShape(vertices=hand_size)
        else:
            raise Exception("%s is not a correct shape" % hand_shape)

        self.hand.CreateFixture(shape=rshape, density=0.1, friction=0.1)
        self.hand.userData = "hand"

        friction_joint = world.CreateFrictionJoint(bodyA=base, bodyB=self.hand, maxForce=2, maxTorque=2)
        b2world_interface.add_bodies(self.hand)

    def set_pos(self, pos, angle):
        self.hand.position = pos
        self.hand.angle = angle

    def apply_wrench(self, rlvel=(0, 0), ravel=0):

        avel = self.hand.angularVelocity
        delta_avel = ravel - avel
        torque = self.hand.mass * delta_avel * 30.0
        self.hand.ApplyTorque(torque, wake=True)

        lvel = self.hand.linearVelocity
        delta_lvel = b2Vec2(rlvel) - b2Vec2(lvel)
        force = self.hand.mass * delta_lvel * self.forceunit
        self.hand.ApplyForce(force, self.hand.position, wake=True)

    def get_state(self, verbose=False):
        state = (
            list(self.hand.position) + [self.hand.angle] + list(self.hand.linearVelocity) + [self.hand.angularVelocity]
        )
        if verbose:
            print_state = ["%.3f" % x for x in state]
            print
            "position, velocity: (%s), (%s) " % ((", ").join(print_state[:3]), (", ").join(print_state[3:]))

        return state


def create_body(base, b2world_interface, body_shape, body_size, body_friction, body_density, obj_loc):
    world = b2world_interface.world

    link = world.CreateDynamicBody(position=obj_loc)
    if body_shape == "rectangle":
        linkshape = b2PolygonShape(box=body_size)
    elif body_shape == "circle":
        linkshape = b2CircleShape(radius=body_size)
    elif body_shape == "polygon":
        linkshape = b2PolygonShape(vertices=body_size)
    else:
        raise Exception("%s is not a correct shape" % body_shape)

    link.CreateFixture(shape=linkshape, density=body_density, friction=body_friction)
    friction_joint = world.CreateFrictionJoint(bodyA=base, bodyB=link, maxForce=5, maxTorque=2)

    b2world_interface.add_bodies([link])
    return link


def make_base(table_width, table_length, b2world_interface):
    world = b2world_interface.world
    base = world.CreateStaticBody(position=(0, 0), shapes=b2PolygonShape(box=(table_length, table_width)))

    b2world_interface.add_bodies([base])
    return base


def add_obstacles(b2world_interface, obsverts):
    world = b2world_interface.world
    obs = []
    for verts in obsverts:
        tmp = world.CreateStaticBody(position=(0, 0), shapes=b2PolygonShape(vertices=verts))
        tmp.userData = "obs"
        obs.append(tmp)

    # add boundaries
    x, y = sm.wbpolygon.exterior.xy
    minx, maxx, miny, maxy = np.min(x), np.max(x), np.min(y), np.max(y)
    centers = [(0, miny - 1), (0, maxy + 1), (minx - 1, 0), (maxx + 1, 0)]
    boxlen = [(maxx - minx, 0.5), (maxx - minx, 0.5), (0.5, maxy - miny), (0.5, maxy - miny)]
    for (pos, blen) in zip(centers, boxlen):
        tmp = world.CreateStaticBody(position=pos, shapes=b2PolygonShape(box=blen))
        obs.append(tmp)
    b2world_interface.add_bodies(obs)


def run_simulation(
    world,
    body,
    body2,
    robot,
    robot2,
    xvel,
    yvel,
    xvel2,
    yvel2,
    rtor,
    rtor2,
    simulation_steps,
    simulation_steps2,
    noise_std=0.01,
):
    # simulating push with fixed direction pointing from robot location to body location
    desired_vel = np.array([xvel, yvel])
    rvel = b2Vec2(desired_vel[0] + np.random.normal(0, noise_std), desired_vel[1] + np.random.normal(0, noise_std))

    desired_vel2 = np.array([xvel2, yvel2])
    rvel2 = b2Vec2(desired_vel2[0] + np.random.normal(0, noise_std), desired_vel2[1] + np.random.normal(0, noise_std))

    tmax = np.max([simulation_steps, simulation_steps2])
    for t in range(tmax + 100):
        if t < simulation_steps:
            robot.apply_wrench(rvel, rtor)
        if t < simulation_steps2:
            robot2.apply_wrench(rvel2, rtor2)
        world.step()

    return (list(body.position), list(body2.position))


class PushReward:
    def __init__(self, noise_std=0.01):

        # domain of this function
        self.xmin = [-5.0, -5.0, -10.0, -10.0, 2.0, 0.0, -5.0, -5.0, -10.0, -10.0, 2.0, 0.0, -5.0, -5.0]
        self.xmax = [5.0, 5.0, 10.0, 10.0, 30.0, 2.0 * np.pi, 5.0, 5.0, 10.0, 10.0, 30.0, 2.0 * np.pi, 5.0, 5.0]
        self._dx = 14

        # starting xy locations for the two objects
        self.sxy = (0, 2)
        self.sxy2 = (0, -2)
        # goal xy locations for the two objects
        self.gxy = [4, 3.5]
        self.gxy2 = [-4, 3.5]

        # Noise to add
        self.noise_std = noise_std

    def get_range(self):
        return np.array([self.xmin, self.xmax])

    @property
    def f_max(self):
        # maximum value of this function
        return np.linalg.norm(np.array(self.gxy) - np.array(self.sxy)) + np.linalg.norm(
            np.array(self.gxy2) - np.array(self.sxy2)
        )

    @property
    def dx(self):
        # dimension of the input
        return self._dx

    def check_input(self, x):
        x = x.reshape((-1))
        if not x.shape[0] == self.dx or any(x > self.xmax) or any(x < self.xmin):
            raise ValueError("x is wrong dim or out of bounds")
        return x

    def __call__(self, argv, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # returns the reward of pushing two objects with two robots
        # argv = self.check_input(argv)

        rx = float(argv[0])
        ry = float(argv[1])
        xvel = float(argv[2])
        yvel = float(argv[3])
        simu_steps = int(float(argv[4]) * 10)
        init_angle = float(argv[5])
        rx2 = float(argv[6])
        ry2 = float(argv[7])
        xvel2 = float(argv[8])
        yvel2 = float(argv[9])
        simu_steps2 = int(float(argv[10]) * 10)
        init_angle2 = float(argv[11])
        rtor = float(argv[12])
        rtor2 = float(argv[13])

        initial_dist = self.f_max

        world = b2WorldInterface()
        oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size = (
            "circle",
            1,
            0.01,
            0.05,
            0.01,
            "rectangle",
            (1, 0.3),
        )

        base = make_base(500, 500, world)
        body = create_body(base, world, "rectangle", (0.5, 0.5), ofriction, odensity, self.sxy)
        body2 = create_body(base, world, "circle", 1, ofriction, odensity, self.sxy2)

        robot = end_effector(world, (rx, ry), base, init_angle, hand_shape, hand_size)
        robot2 = end_effector(world, (rx2, ry2), base, init_angle2, hand_shape, hand_size)
        (ret1, ret2) = run_simulation(
            world,
            body,
            body2,
            robot,
            robot2,
            xvel,
            yvel,
            xvel2,
            yvel2,
            rtor,
            rtor2,
            simu_steps,
            simu_steps2,
            self.noise_std,
        )

        ret1 = np.linalg.norm(np.array(self.gxy) - ret1)
        ret2 = np.linalg.norm(np.array(self.gxy2) - ret2)
        return initial_dist - ret1 - ret2

    def testmode(self, x, num_seeds=100):
        s0 = 1000000
        testseeds = np.linspace(s0, s0 + num_seeds - 1, num_seeds).astype(int)
        test_f = np.array([self.__call__(x, s) for s in testseeds])

        return test_f.mean()


#######################################################


if __name__ == "__main__":

    import sys

    args = sys.argv

    # Run example code
    if len(args) == 1:
        f = PushReward()
        x = np.random.uniform(f.xmin, f.xmax)
        print("Input = {}".format(x))
        print("Output = {}".format(f.testmode(x)))

    else:
        # run with provided command line arguments
        if len(args) == 15:
            x = np.array(args[1:]).astype(np.float)
            f = PushReward()
            print(f(x))
        elif len(args) == 16:
            x = np.array(args[1:-1]).astype(np.float)
            seed = int(args[-1])
            f = PushReward()
            print(f(x))
        else:
            raise ValueError("provide 0 or 14 floats")
