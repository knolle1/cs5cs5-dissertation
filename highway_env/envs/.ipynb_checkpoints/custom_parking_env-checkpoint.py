from abc import abstractmethod
from typing import Optional

import numpy as np
from gymnasium import Env

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import (
    MultiAgentObservation,
    observation_factory,
)
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle

from line_profiler import profile


class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class CustomParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    # For parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
    # Bug fixed by Mcfly(https://github.com/McflyWZX)
    PARKING_OBS = {
        "observation": {
            "type": "CustomKinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        }
    }

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        self.global_step = 0
        super().__init__(config, render_mode)
        self.observation_type_parking = None

    @classmethod
    @profile
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "CustomKinematicsGoal",
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False,
                },
                "action": {"type": "CustomContinuousAction"},
                "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05], #[1, 0.3, 0, 0, 0.02, 0.02],
                "success_goal_reward": 0.03, #0.01, #0.05, #0.12,
                "collision_reward": -10,#-100, #-5,
                "steering_range": np.deg2rad(45),
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 100,
                "screen_width": 600,
                "screen_height": 300,
                "centering_position": [0.5, 0.5],
                "scaling": 7,
                "controlled_vehicles": 1,
                "vehicles_count": 50,
                "add_walls": True,
                "parking_angles" : [0, 0],
                "fixed_goal" : None,
                "reward_p" : 0.5, #1, #0.5,
                "collision_reward_factor" : 50,
                "env_change_config" : [],
                "env_change_frequency" : 100_000
            }
        )
        return config

    @profile
    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(
            self, self.PARKING_OBS["observation"]
        )

    @profile
    def _info(self, obs, action) -> dict:
        info = super(CustomParkingEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(
                self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
                for agent_obs in obs
            )
        else:
            obs = self.observation_type_parking.observe()
            achieved = obs[:len(self.observation_type_parking.features)]
            goal = obs[len(self.observation_type_parking.features):]
            success = self._is_success(achieved, goal)
            
            test_success = (self.compute_reward(achieved, goal, {})
                > -self.config["success_goal_reward"]
            ) and not self._is_crashed()  and self._is_goal_hit()
        info.update({"is_success": success,
                     "is_crashed" : self._is_crashed(),
                     "achieved" : achieved,
                     "goal" : goal,
                     "reward" : self.compute_reward(achieved, goal, {}),
                     "goal_hit" : self._is_goal_hit(),
                     "test_success" : test_success
                     })
        return info

    @profile
    def _reset(self):
        env_idx = self.global_step // self.config["env_change_frequency"]
        if env_idx in range(len(self.config["env_change_config"])):
            self.configure(self.config["env_change_config"][env_idx])
        print(f"global_step: {self.global_step}, env_idx: {env_idx}, parking angles: {self.config['parking_angles']}")
        self._create_road()
        self._create_vehicles()

    @profile
    def _create_road(self, spots: int = 10) -> None:
        """
        Create a road composed of straight adjacent lanes at angles specified in config.
        Angle of 0 means vertical parking spaces.

        :param spots: number of spots in the parking
        """
        # Parking space parameters
        width = 4.0
        length = 8
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        nodes = [["a", "b"], ["b", "c"]]
        
        net = RoadNetwork()
        
        for row in range(2):
            alpha = (-1) ** row * self.config["parking_angles"][row] * np.pi / 180
            
            if abs(self.config["parking_angles"][row] <= 50):
                delta_x = np.sin(alpha) * length # difference for end point of diagonal lanes
                x_offset = width * (1 - np.cos(alpha)) / np.cos(alpha) / 2 # offset between spaces so they align
                y_offset = 10
                
                spots = int(75 / (width + x_offset))
            
                for k in range(spots):
                    x = (k - spots // 2) * (width + x_offset) + (width) / 2
                    if alpha > 0:
                        x -= delta_x
                    net.add_lane(
                        *nodes[0],
                        StraightLane(
                            [x, (-1) ** row * -y_offset], 
                            [x + delta_x, (-1) ** row * -(y_offset + length)], 
                            width=width, line_types=lt
                            ),
                        )
            else:
                x_offset = 0
                y_offset = 10
                
                spots = int(80 / (length))
                
                for k in range(spots):
                    x = (k - spots // 2) * (length + x_offset) + (spots%2) * length / 2
                    net.add_lane(
                        *nodes[0], 
                        StraightLane(
                            [x, (-1) ** row * -y_offset], 
                            [x + length, (-1) ** row * -y_offset],
                            width=width, line_types=lt))
                       
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    @profile
    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        empty_spots = list(self.road.network.lanes_dict().keys())

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            x0 = (i - self.config["controlled_vehicles"] // 2) * 10 #10
            x, heading = self.np_random.choice([(x0-35, 2*np.pi), (x0+35, np.pi)])
            vehicle = self.action_type.vehicle_class(
                self.road, [x, 0], heading, 0  #2 * np.pi * self.np_random.uniform(), 0 
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            #empty_spots.remove(vehicle.lane_index)

        # Goal
        for vehicle in self.controlled_vehicles:
            if self.config["fixed_goal"] == None:
                lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))] # Pick random goal
            elif type(self.config["fixed_goal"]) == list:
                top_row_idx = np.arange(len(empty_spots)/2)
                bottom_row_idx = np.arange(len(empty_spots)/2) + len(empty_spots)/2
                idx = (top_row_idx, bottom_row_idx)
                goal_set = []
                for row, spot in self.config["fixed_goal"]:
                    goal_set.append(int(idx[row][spot]))
                lane_index = empty_spots[self.np_random.choice(goal_set)] # Pick random goal from specified set
            else:
                lane_index = empty_spots[self.config["fixed_goal"]]
                
            lane = self.road.network.get_lane(lane_index)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)
            empty_spots.remove(lane_index)

        # Other vehicles
        for i in range(self.config["vehicles_count"]):
            if not empty_spots:
                continue
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            lane = self.road.network.get_lane(lane_index)
            obstacle = Obstacle(self.road, lane.position(lane.length / 2, 0), heading=lane.heading)
            obstacle.LENGTH, obstacle.WIDTH = (self.road.vehicles[0].LENGTH, self.road.vehicles[0].WIDTH)
            #obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)
            
            empty_spots.remove(lane_index)
            
        # Walls
        if self.config["add_walls"]:
            width, height = 85, 42
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

    @profile
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                np.array(self.config["reward_weights"]),
            ),
            self.config["reward_p"],
        )

    @profile
    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        #reward = sum(
        #    self.compute_reward(
        #        agent_obs[:len(self.observation_type_parking.features)], # Achieved goal
        #        agent_obs[len(self.observation_type_parking.features):], # Desired goal
        #        {}
        #    )
        #    for agent_obs in obs
        #)
        #reward += self.config["collision_reward"] * sum(
        #    v.crashed for v in self.controlled_vehicles
        #)
        
        reward = sum(
            (1 + self.config["collision_reward_factor"]*int(vehicle.crashed)) * self.compute_reward(
                agent_obs[:len(self.observation_type_parking.features)], # Achieved goal
                agent_obs[len(self.observation_type_parking.features):], # Desired goal
                {}
            ) + self.config["collision_reward"]*int(vehicle.crashed)
            for agent_obs, vehicle in zip(obs, self.controlled_vehicles)
        )
        
        return reward

    @profile
    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        
        return (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -self.config["success_goal_reward"]
        ) and not self._is_crashed() #and self._is_goal_hit()

    @profile
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
        crashed = self._is_crashed()
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs[:len(self.observation_type_parking.features)], # Achieved goal
                             agent_obs[len(self.observation_type_parking.features):], # Desired goal
                             )
            for agent_obs in obs
        )
        return bool(crashed or success)

    @profile
    def _is_truncated(self) -> bool:
        """The episode is truncated if the time is over."""
        return self.time >= self.config["duration"]
    
    @profile
    def _is_crashed(self) -> bool:
        """Check if vehicle crashed."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles)
    
    def _is_goal_hit(self) -> bool:
        """Check if vehicle touched goal."""
        return any(vehicle.goal.hit for vehicle in self.controlled_vehicles)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.global_step += 1
        
        return obs, reward, terminated, truncated, info

    def reset_global_step(self):
        self.global_step = 0


#class ParkingEnvActionRepeat(ParkingEnv):
#    def __init__(self):
#        super().__init__({"policy_frequency": 1, "duration": 20})


#class ParkingEnvParkedVehicles(ParkingEnv):
#    def __init__(self):
#        super().__init__({"vehicles_count": 10})
