import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class image:
     def __init__(self, detN=100, detz=100, dety = 50, detx = 50,
                  boxl=10, boxz=80, boxy =15, boxx = 15):

        self.mu_a = 0.1
        self.z_prime = 10

        self.detNx = detN
        self.detNy = detN
        self.detector = np.zeros((detN,detN))
        self.detz = detz # detector distance from x-ray origin
        self.dety = dety # vertical size of detector
        self.detx = detx # horizontal size of detector

        self.boxz = boxz # box distance from x-ray origin
        self.boxy = boxy # vertical size of box
        self.boxx = boxx # horizontal size of box
        self.boxl = boxl # thickness of box

        self.shadowx = self.boxx * self.detz/self.boxz
        self.shadowy = self.boxy * self.detz/self.boxz

        self.small_shadowx = self.boxx * self.detz/(self.boxz+self.boxl)
        self.small_shadowy = self.boxy * self.detz/(self.boxz+self.boxl)

        self.deltax = self.detx / (self.detNx-1)
        self.deltay = self.dety / (self.detNy-1)

        eps = 1e-10
        x = np.arange(-0.5*self.detx,0.5*self.detx+eps, self.deltax)
        y = np.arange(-0.5*self.dety,0.5*self.dety+eps, self.deltay)
        X, Y = np.meshgrid(x, y)

        self.X = X
        self.Y = Y

     def outer_value(self):
        return 1*np.cos(self.theta(self.X,self.Y))**3


     def inner_value(self):
        return 1*np.cos(self.theta(self.X,self.Y))**3*\
              np.exp(-self.mu_a*self.boxl/np.cos(self.theta(self.X,self.Y)))

     def edge_value(self):
        return 1/1000*np.cos(self.theta(self.X,self.Y))**3*\
              np.exp(-self.mu_a*(self.z_prime-(self.boxz-self.boxl/2)/\
                     np.cos(self.theta(self.X,self.Y))))

     def project(self):
        # shadow
        A1 = self.X<=0.5*self.shadowx
        A2 = self.X>=-0.5*self.shadowx
        B1 = self.Y<=0.5*self.shadowy
        B2 = self.Y>=-0.5*self.shadowy
        outer = (A1*A2*B1*B2)==False
        self.detector = outer*(1*np.cos(self.theta(self.X,self.Y))**3)

        # small_shadow
        A1 = self.X<=0.5*self.small_shadowx
        A2 = self.X>=-0.5*self.small_shadowx
        B1 = self.Y<=0.5*self.small_shadowy
        B2 = self.Y>=-0.5*self.small_shadowy
        inner = A1*A2*B1*B2
        edge = (outer.astype(int) ^ inner.astype(int))==False


        self.detector = inner.astype(int)*self.inner_value()\
                        + edge.astype(int)*self.edge_value()\
                        + outer.astype(int)*self.outer_value()

        return self.detector

     def theta(self,x,y):
        return np.arctan(np.sqrt(x**2+y**2)/self.detz)

     def XY(self,x,y):
        return np.arctan(np.sqrt(x**2+y**2)/self.detz)


def main():
  obj = image()
  field = obj.project()
  assert(field.shape==(obj.detNx,obj.detNy))

  fig, ax = plt.subplots()
  CS = ax.contourf(obj.X, obj.Y, obj.detector)
  #ax.clabel(CS, inline=True, fontsize=10)
  #ax.set_title('Simplest default with labels')
  cbar = fig.colorbar(CS)
  cbar.ax.set_ylabel('verbosity coefficient')
  print('here3')
  plt.show()
  print('here4')
main()
