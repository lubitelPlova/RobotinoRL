import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame as pg 
import math
#import os
#import pickle
import phys_env
import random
from stable_baselines3 import A2C

class RobotinoWorldEnv(gym.Env):
    metadata = {"render_modes":["human", "rgb_array"], "render_fps": 30}
    field_struct = None
    
    def __init__(self, render_mode = 'human'):
        self.window = 800
        self.field = phys_env.Field()
        self.observation_space = spaces.Box(dtype = np.float32, low = np.array([-800, -800, -180]),\
           high = np.array([800, 800, 180]))
        self.action_space = spaces.Box(dtype = np.float32, low = np.array([-1.2,-1.2,-13]),\
           high = np.array([1.2,1.2,13]))     
        self.pixels = []

        self.traj = [(400,400)]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.field = phys_env.Field(traj = self.traj)
        self.field.get_real_traj()
        self.field.trace = []
        raw_state, reward, terminated, truncated = self.field.get_raw_state()
        info = {}
        return raw_state, info

    def set_traj(self, traj):
       self.traj = traj
       self.field.traj = traj
       #self.field.traj_len = len(traj)
       #self.field.get_real_traj()

    def step(self, action):
        self.field.ustV_vector = action
        # self.field.wheel1.u = action[0]
        # self.field.wheel2.u = action[1]
        # self.field.wheel3.u = action[2]

        self.field.update()
        info = {}
        state, reward, terminated, truncated = self.field.get_raw_state()

        return state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pg.init()
            pg.display.init()
            self.window = pg.display.set_mode((self.field.width, self.field.height))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.Clock()

        if self.field_struct is None and self.render_mode == "human":
             self.field_struct = pg.image.load('poly_colored.png').convert_alpha()
             self.field_struct = pg.transform.scale(self.field_struct, (800,800))

        #pg.display.flip()
        
        self.field.FieldSurface.fill((255,255,255))
        self.field.FieldSurface.blit(self.field_struct,(0,0))

        # pg.draw.line(self.field.FieldSurface, 'Blue', (400, 400), (400 + math.cos((self.field.uangle+90)*math.pi/180)*1000,
        #  400 + math.sin((self.field.uangle+90)*math.pi/180)*1000), 5)
        pg.draw.circle(self.field.FieldSurface, 'Red', (self.field.posX, self.field.posY), 5)
        pg.draw.circle(self.field.FieldSurface, 'Red', (self.field.wheel1.posX, self.field.wheel1.posY), 5)
        pg.draw.circle(self.field.FieldSurface, 'Grey', (self.field.wheel2.posX, self.field.wheel2.posY), 5)
        pg.draw.circle(self.field.FieldSurface, 'Black', (self.field.wheel3.posX, self.field.wheel3.posY), 5)
        pg.draw.line(self.field.FieldSurface, 'Grey', (self.field.posX, self.field.posY), (self.field.posX + self.field.Vx*40,self.field.posY + self.field.Vy*40), 2)
        pg.draw.line(self.field.FieldSurface, 'Black', (self.field.posX, self.field.posY), (self.field.posX + math.cos((self.field.obj_angle+90)*math.pi/180)*40,self.field.posY + math.sin((self.field.obj_angle+90)*math.pi/180)*40), 2)
        # pg.draw.circle(self.field.FieldSurface, 'Green', (400+200*(math.cos((self.field.uangle+90)*math.pi/180)),400+200*(math.sin((self.field.uangle+90)*math.pi/180))), 15)
        
        pg.draw.line(self.field.FieldSurface, 'Blue', (400,400), self.field.traj[0], 4)

        for i in range(1,self.field.traj_len):
            pg.draw.line(self.field.FieldSurface, 'Blue', self.field.traj[i-1], self.field.traj[i], 4)

        # for xy in self.field.real_traj:
        #     pg.draw.circle(self.field.FieldSurface, 'Green', xy, 5)

        for xy in self.field.traj:
            pg.draw.circle(self.field.FieldSurface, 'Red', xy, 10)

        for xy in self.field.trace:
            pg.draw.circle(self.field.FieldSurface, 'Black', xy, 2)

        if self.render_mode=='human':
            self.window.fill((255,255,255))
            self.window.blit(self.field.FieldSurface,  (0, 0))
            pg.event.pump()
            pg.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pg.surfarray.pixels3d(self.field.FieldSurface), axes=(1, 0, 2)))

    def close(self):
        self.field.trace = []
        if self.window is not None:
            self.window = None
            self.clock = None
            pg.display.quit()
            pg.quit()


