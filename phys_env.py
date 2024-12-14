import math
import numpy as np
import pygame as pg

class RobotWheel:
    def __init__(self, u, X, Y):           
            self.u = u
            if self.u > 0:
                self.direction = 1
            else:
                self.direction = -1
            self.loss = 0
            self.posX = X
            self.posY = Y
            self.rect = (X, Y, 1, 1)   #???
            
            self.omega = self.u * self.direction

    def update(self):
        if self.u >= 0:
            self.direction = 1
        else:
            self.direction = -1

        if (abs(self.u)-self.loss>=0):
            self.omega = (abs(self.u) - self.loss) * self.direction
        else:
            self.omega = 0

    def check_bounds(self):
        if self.posX <= 0 or self.posX >= 800:
            return True
        if self.posY <= 0 or self.posY >= 800:
            return True
        return False

class Field:
     trace = []

     color = {
        1:(0,250,0),
        2:(253,233,16),
        3:(253,153,0)
        }

     def __init__(self, uVx = 1, uVy = 0, uangle = 0,  width=800, height=800, 
                  rx = 400, ry = 400, rew_step = -2, rew_timeout = -20, traj = [(400,500),(500,500),(500,600)]):
        #параметры поля
        self.uVx = uVx
        self.uVy = uVy
        self.uabs = 1 #round(math.sqrt(uVx**2 + uVy**2),3)
        self.uangle = uangle
        self.width = width
        self.height = height
        self.FieldSurface = pg.Surface((width, height))
        

        #параметры робота
        self.posX = rx # позиция центра робота 
        self.posY = ry
        self.angle = 0
        self.move_angle = 0
        self.Vx = 0
        self.Vy = 0 # скорости центра робота 
        self.Va = 0
        self.wheel1 = RobotWheel(-0.2, (self.posX + 50*math.cos((self.angle+120-90)*math.pi/180)),
                                (self.posY + 50*math.sin((self.angle+120-90)*math.pi/180)))
        self.wheel2 = RobotWheel(0, (self.posX + 50*math.cos((self.angle-90)*math.pi/180)),
                                (self.posY + 50*math.sin((self.angle-90)*math.pi/180)))
        self.wheel3 = RobotWheel(0, (self.posX + 50*math.cos((self.angle-120-90)*math.pi/180)),
                                (self.posY + 50*math.sin((self.angle-120-90)*math.pi/180)))

        self.wheel_radius = 1
        self.L = 1
        self.ustV_vector = np.array([[self.uVx],[self.uVy],[self.uabs]])
        self.velocities_vector = None #np.array([[Vx],[Vy],[Va]])
        self.move_matrix  = np.array([[(-2/3)*math.cos((self.angle-30-90)*math.pi/180), (2/3)*math.sin((self.angle-90)*math.pi/180), (2/3)*math.cos((self.angle+30-90)*math.pi/180)],
                                      [(-2/3)*math.sin((self.angle-30-90)*math.pi/180), (-2/3)*math.cos((self.angle-90)*math.pi/180), (2/3)*math.sin((self.angle+30-90)*math.pi/180)],
                                      [(3/(3*self.L)),(3/(3*self.L)),(3/(3*self.L))]])
        self.omega_vector = np.array([[self.wheel1.omega],[self.wheel2.omega],[self.wheel3.omega]])
       
        #Для расчёта обратной матрицы кинематики
        self.desirable_omega = np.matmul(np.linalg.inv(self.move_matrix), self.ustV_vector)

        #параметры среды
        self.point_done = False
        self.timeout = False
        self.bound = False
        self.terminated = False
        self.truncated = False
        self.step_count = 0
        self.rew_step = rew_step #награда за шаг
        self.on_point = 0 #награда за достижение точки
        self.rew_point = 1000
        self.rew_finish = 2000
        self.rew_bound = -1000 #награда за выход за пределеы
        self.rew_timeout = rew_timeout #награда за таймаут без достижения точки

        self.pix_array3d = self._get_pixels()
        self.err_angle = 0
        self.obj_angle = 0
        self.dif_angle = 0
        self.traj = traj
        self.traj_len = len(self.traj)
        self.real_traj = traj
        self.real_traj_len = len(self.traj)
        self.uposX = 410
        self.uposY = 410
        self.prevposX = 400
        self.prevposY = 400
        self.point_count = 0
        self.pth = None
        self.diverse = 0 
        self.prange = 0
        self.prevrange = 800*math.sqrt(2)
        self.go_further = 0
        self.on_line = 0
        self.get_path()
        
     def _get_pixels(self): 
         return np.load('field_conf.npy')

     def _get_wheel_loss(self, wheel):
         color = self.pix_array3d[int(wheel.posX)][int(wheel.posY)]
         color_list = color.tolist()
         if color_list == [0, 255, 72]:
             loss = 0.05
         elif color_list == [255, 242, 0]:
             loss = 0.1
         elif color_list == [255, 127, 39]:
             loss = 0.15
         elif color_list == [236, 28, 36]:
             loss = 0.2
         elif color_list == [184, 61, 186]:
             loss = 0.25
         elif color_list == [0, 168, 243]:
             loss = 0.3
         else:
             loss = 0
         return loss
     
     def get_real_traj(self, n = 10):
         points = []
         for j in range(1,n+1):
             x = 400 + (self.traj[0][0] - 400) * j / (n+1)
             y = 400 + (self.traj[0][1] - 400) * j / (n+1)
             points.append((x, y))
         for i in range(self.traj_len-1):
             points.append((self.traj[i][0],self.traj[i][1]))
             for j in range(1, n+1):
                 x = self.traj[i][0] + (self.traj[i+1][0] - self.traj[i][0]) * j / (n+1)
                 y = self.traj[i][1] + (self.traj[i+1][1] - self.traj[i][1]) * j / (n+1)
                 points.append((x, y))
         points.append((self.traj[self.traj_len-1][0],self.traj[self.traj_len-1][1]))
         self.real_traj = points
         self.real_traj_len = len(self.real_traj)

     def get_path(self):
         path = (self.uposY-self.prevposY,-(self.uposX-self.prevposX),-self.prevposX*(self.uposY-self.prevposY)+self.prevposY*(self.uposX-self.prevposX))
         self.pth = path
        
     def path_loss(self):
         d = abs(self.pth[0]*self.posX+self.pth[1]*self.posY+self.pth[2])/math.sqrt(self.pth[0]**2+self.pth[1]**2)
         return d

     def update(self):
        
        self.step_count += 1

        if self.point_count < self.real_traj_len:
            self.uposX = self.real_traj[self.point_count][0]
            self.uposY = self.real_traj[self.point_count][1]
        else:
            self.terminated = 1

        if (self.posX - self.uposX)**2 + (self.posY-self.uposY)**2 <= 25:
            self.prevposX = self.real_traj[self.point_count][0]
            self.prevposX = self.real_traj[self.point_count][1]
            self.point_count += 1
            self.on_point = 1
            #Награда за достижение точки
        else:
            self.on_point = 0


        self.get_path()
        #print(self.pth[0],self.pth[1],self.pth[2])
        self.prange = math.sqrt((self.uposX-self.posX)**2 + (self.uposY-self.posY)**2)
        
        self.desirable_omega = (1/self.wheel_radius) * np.matmul(np.linalg.inv(self.move_matrix), self.ustV_vector)
         
        self.wheel1.u = float(self.desirable_omega[0])
        self.wheel2.u = float(self.desirable_omega[1])
        self.wheel3.u = float(self.desirable_omega[2])
       
        self.wheel1.loss = self._get_wheel_loss(self.wheel1)
        self.wheel2.loss = self._get_wheel_loss(self.wheel2)
        self.wheel3.loss = self._get_wheel_loss(self.wheel3)
       

        self.wheel1.update()
        self.wheel2.update()
        self.wheel3.update()

        self.omega_vector = np.array([[self.wheel1.omega],[self.wheel2.omega],[self.wheel3.omega]])

        self.move_matrix  = np.array([[(-2/3)*math.cos((self.angle-30-90)*math.pi/180), (2/3)*math.sin((self.angle-90)*math.pi/180), (2/3)*math.cos((self.angle+30-90)*math.pi/180)],
                                      [(-2/3)*math.sin((self.angle-30-90)*math.pi/180), (-2/3)*math.cos((self.angle-90)*math.pi/180), (2/3)*math.sin((self.angle+30-90)*math.pi/180)],
                                      [(3*4/(3*self.L)),(3*4/(3*self.L)),(3*4/(3*self.L))]])

        self.velocities_vector = self.wheel_radius * np.matmul(self.move_matrix, self.omega_vector)

        
       
        self.Vx = float(self.velocities_vector[0])
        self.Vy = float(self.velocities_vector[1])
        self.Va = float(self.velocities_vector[2])
 
        self.angle += self.Va
        self.obj_angle = math.atan2(-(self.uposX-self.posX), self.uposY-self.posY)*180/math.pi
        self.move_angle = math.atan2(-self.Vx, self.Vy)*180/math.pi
        self.dif_angle = self.obj_angle - self.move_angle

        if abs(self.dif_angle) <= 180:
            self.err_angle = self.dif_angle
        elif self.dif_angle > 180:
            self.err_angle = self.dif_angle - 360
        elif self.dif_angle < -180:
            self.err_angle = self.dif_angle + 360

        if self.angle > 360:
            self.angle = self.angle % 360 
        elif self.angle <= 0:
            self.angle = 360

        self.posX += self.Vx
        self.posY += self.Vy

        self.trace.append((self.posX,self.posY))

        self.wheel1.posX = self.posX + 50*math.cos((self.angle+60-90)*math.pi/180)
        self.wheel1.posY = self.posY + 50*math.sin((self.angle+60-90)*math.pi/180)
        self.wheel2.posX = self.posX - 50*math.cos((self.angle-90)*math.pi/180)
        self.wheel2.posY = self.posY - 50*math.sin((self.angle-90)*math.pi/180)
        self.wheel3.posX = self.posX + 50*math.cos((self.angle-60-90)*math.pi/180)
        self.wheel3.posY = self.posY + 50*math.sin((self.angle-60-90)*math.pi/180)
       
       
        self.diverse = self.path_loss()
       #self.finrange = math.sqrt((self.finpoint[0]-self.posX)**2+(self.finpoint[1]-self.posY)**2)

        if self.diverse < 10:
            self.on_line = 1
        else:
            self.on_line = 0

        if self.prevrange - self.prange >= 1 :
            self.go_further = 1
            self.prevrange = self.prange
        else:
            self.go_further = 0

        if self.on_point == 1:
            self.prevrange = 2000
       
        if self.point_count == self.real_traj_len:
            self.terminated = True
            #self.trace = []
        if self.wheel1.check_bounds() or self.wheel2.check_bounds() or self.wheel3.check_bounds():
            self.truncated = True
            #self.trace=[]
        if self.step_count >= 1000:
            self.truncated = True
            #self.trace=[]
      
     def get_raw_state(self):
         reward =  (10*(180-abs(self.err_angle))/180) - 5 + self.go_further
         if self.on_point:
             reward = 1000 #self.rew_point
         if self.terminated:
             reward = 10000 #self.rew_finish
         if self.truncated:
             reward = 0
         #if self.terminated or self.truncated:
         #    done = True
         #else:
         #    done = False 
             ###
             ### УБРАТЬ КООРДИНАТЫ ЦЕНТРА, убрать угловые скорости скорости, добавить координату цели
           
         state = np.array([self.uposX - self.posX, self.uposY-self.posY, self.err_angle], dtype = np.float32) # self.posX, self.posY, self.uposX, self.uposY

         return state, reward, self.terminated, self.truncated
