import re
import numpy as np
from numpy.core.fromnumeric import product
import torch
import torch_scatter 
import math

class Sphere:
    def __init__(self, dim=2):
        self.dim = dim

    def _is_in_unit_sphere(self, x):
        norm_2 = torch.norm(x, dim=-1)
        return ~(torch.abs(norm_2 - 1) > 1e-7).prod().bool()

    def _ensure_in_unit_sphere(self, x):
        assert self._is_in_unit_sphere(x), 'One of the given vector is not on the unit sphere'

    def _is_in_tangent_space(self, center, v):
        '''
        inputs:
            center: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            if_in_tangence: bool
        '''
        self._ensure_in_unit_sphere(center)
        product = torch.matmul(v, center[:,:,None])
        product[torch.isnan(product)] = 0.0
        return (torch.abs(torch.matmul(v, center[:,:,None])) <= 1e-7).prod().bool()

    def _ensure_in_tangent_space(self, center, v):
        assert self._is_in_tangent_space(center, v), 'One of the given vector is not on the tangent space'

    def _is_in_ctangent_space(self, center, v):
        '''
        inputs:
            center: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            if_in_tangence: bool
        '''
        self._ensure_in_unit_sphere(center)
        v_minus = v[:,:,:-1]
        center_minus = center[:,:-1]
        product = torch.matmul(v_minus, center_minus[:,:,None])
        product[torch.isnan(product)] = 0.0
        return (torch.abs(torch.matmul(v_minus, center_minus[:,:,None])) <= 1e-7).prod().bool()

    def _ensure_in_ctangent_space(self, center, v):
        assert self._is_in_ctangent_space(center, v), 'One of the given vector is not on the cylindrical-tangent space'

    def geo_distance(self, u, v):
        '''
        inputs:
            u: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            induced_distance(u,v): (N, M)
        '''
        assert u.shape[1] == v.shape[2] == self.dim + 1, 'Dimension is not identical.'
        self._ensure_in_unit_sphere(u)
        self._ensure_in_unit_sphere(v)
        return torch.arccos(torch.matmul(v, u[:,:,None]))

    def tangent_space_projector(self, x, v):
        '''
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            project_x(v): (N, M, self.dim + 1)
        '''
        assert x.shape[1] == v.shape[2], 'Dimension is not identical.'

        x_normalized = torch.divide(x, torch.norm(x, dim=-1, keepdim=True))
        v_normalized = torch.divide(v, torch.norm(v, dim=-1, keepdim=True))
        v_on_x_norm = torch.matmul(v_normalized, x_normalized[:,:,None]) #N, M, 1
        v_on_x = v_on_x_norm * x_normalized[:,None,:] #N,M,dim
        p_x = v_normalized - v_on_x #N,M,dim
        return p_x

    def exp_map(self, x, v):
        '''
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1) which is on the tangent space of x
        outputs:
            exp_x(v): (N, M, self.dim + 1)
        '''
        assert x.shape[1] == v.shape[2] == self.dim + 1, 'Dimension is not identical.'
        self._ensure_in_unit_sphere(x)
        self._ensure_in_tangent_space(x, v)

        v_norm = torch.norm(v, dim=-1)[:,:,None]  # N,M, 1
        return torch.cos(v_norm) * x[:,None,:] + torch.sin(v_norm) * torch.divide(v, v_norm)

    def log_map(self, x, v):
        '''
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1) # v is on the sphere
        outputs:
            log_x(v): (N, M, self.dim + 1)
        '''
        assert x.shape[1] == v.shape[2] == self.dim + 1, 'Dimension is not identical.'
        self._ensure_in_unit_sphere(x)
        self._ensure_in_unit_sphere(v)

        p_x = self.tangent_space_projector(x, v-x[:,None,:]) #N,M,d
        p_x_norm = torch.norm(p_x, dim=-1)[:,:,None] #N,M,1
        distance = self.geo_distance(x, v) #N,M,1
        log_xv = torch.divide(distance * p_x, p_x_norm)
        log_xv[torch.isnan(log_xv)] = 0.0  # map itself to the origin

        return log_xv

    def horizon_map(self, x, v):
        '''
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1) # v is on the sphere
        outputs:
            H_x(v): (N, M, self.dim + 1)
        '''
        assert x.shape[1] == v.shape[2] == self.dim + 1, 'Dimension is not identical.'
        self._ensure_in_unit_sphere(x)
        self._ensure_in_unit_sphere(v)

        x_minus = x[:,:-1]
        v_minus = v[:,:,:-1]
        p_x_minus = self.tangent_space_projector(x_minus, v_minus - x_minus[:,None,:])
        p_x = torch.cat([p_x_minus, v[:,:,[-1]]- x[:,None,[-1]]], dim=-1)
        p_x_norm = torch.norm(p_x, dim=-1)[:,:,None] 
        distance = self.geo_distance(x, v)
        H_xv = torch.divide(distance * p_x, p_x_norm)
        H_xv[torch.isnan(H_xv)] = 0.0  # map itself to the origin

        return H_xv
    
    def cart3d_to_ctangent_local2d(self, x, v):
        '''
        inputs:
            x: (N, 3)
            v: (N, M, 3) # v is on the ctangent space of x
        outputs:
            \Pi_x(v): (N, M, 2)
        '''
        assert x.shape[1] == v.shape[2] == 3, 'the method can only used for 2d sphere, so the input should be in R^3.'
        self._ensure_in_ctangent_space(x, v)
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        lat, lon = self.xyz2latlon(x1, x2, x3)

        v_temp = v.sum(dim=-1, keepdim=True)
        idx_zero = (v_temp == 0)

        e_phi = torch.stack([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1)
        v_phi = torch.matmul(v, e_phi[:,:,None])
        v_phi[idx_zero] = 0
        v_z = v[:,:,[-1]]
        v_z[idx_zero] = 0
        return torch.cat([v_phi, v_z], dim=-1)

    def cart3d_to_tangent_local2d(self, x, v):
        '''
        inputs:
            x: (N, 3)
            v: (N, M, 3) # v is on the tangent space of x
        outputs:
            \Pi_x(v): (N, M, 2)
        '''
        assert x.shape[1] == v.shape[2] == 3, 'the method can only used for 2d sphere, so the input should be in R^3.'
        self._ensure_in_tangent_space(x, v)
        
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        lat, lon = self.xyz2latlon(x1, x2, x3)
        e_theta = torch.stack([torch.sin(lat)*torch.cos(lon), torch.sin(lat)*torch.sin(lon), torch.cos(lat)], dim=-1) #N,3
        e_phi = torch.stack([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1) #N,3
        
        v_temp = v.sum(dim=-1, keepdim=True)
        idx_zero = (v_temp == 0)

        v_theta = torch.matmul(v-x[:,None,:], e_theta[:,:,None]) #N,M,1
        v_theta[idx_zero] = 0
        v_phi = torch.matmul(v-x[:,None,:], e_phi[:,:,None]) #N,M,1
        v_phi[idx_zero] = 0
        return torch.cat([v_theta, v_phi], dim=-1)

    @classmethod
    def latlon2xyz(self, lat, lon, is_input_degree=True):
        if is_input_degree == True:
            lat = lat*math.pi/180
            lon = lon*math.pi/180 
        x= torch.cos(lat)*torch.cos(lon)
        y= torch.cos(lat)*torch.sin(lon)
        z= torch.sin(lat)
        return x, y, z

    @classmethod
    def xyz2latlon(self, x, y, z):
        lat = torch.atan2(z, torch.norm(torch.stack([x,y], dim=-1), dim=-1))
        lon = torch.atan2(y, x)
        return lat, lon


if __name__ == '__main__':
    sphere_2d = Sphere(2)
    x = torch.rand((100, 3))
    x_norm = torch.norm(x, dim=-1)
    x = torch.divide(x, x_norm[:,None])
    
    y = torch.rand((200, 3))
    y_norm = torch.norm(y, dim=-1)
    y = torch.divide(y, y_norm[:,None])
    p = sphere_2d.tangent_space_projector(x, y)
    v = log = sphere_2d.log_map(x, y)
    sphere_2d.cart3d_to_local2d(x, v)
    print('finished')
