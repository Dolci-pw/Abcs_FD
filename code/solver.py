#==============================================================================
# Python Imports
#==============================================================================
import numpy as np
import math  as mt
from   pyrevolve import Revolver
#==============================================================================

#==============================================================================
# Devito Imports
#==============================================================================
from   devito import *
from   examples.seismic import RickerSource
from   examples.seismic import Receiver
from   examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator

#==============================================================================

#==============================================================================
class solverABCs():
    
    def solvedamp(self,rec,src,vp,geramdamp,u,grid,setup,system,save=False,**kwargs):    
        
    
        nptx = setup.nptx
        nptz = setup.nptz

        D0   = geramdamp
        damp = Function(name="damp",grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)
        damp.data[:,:] = D0

        (x,z)   = grid.dimensions     
        t       = grid.stepping_dim
        dt      = grid.stepping_dim.spacing
       
        subds = ['d1','d2','d3']
        
        
        pde0 = Eq(u.dt2 - u.laplace*vp*vp)

        if(system=='forward'):
            
            pde1 = Eq(u.dt2 - u.laplace*vp*vp + vp*vp*damp*u.dtc)

            stencil0 =  Eq(u.forward, solve(pde0,u.forward),subdomain = grid.subdomains['d0'])
            stencil1 = [Eq(u.forward, solve(pde1,u.forward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]

            src_term = src.inject(field=u.forward,expr=src*dt**2*vp**2)
            rec_term = rec.interpolate(expr=u)
            bc = [Eq(u[t+1,0,z],0.),Eq(u[t+1,nptx-1,z],0.),Eq(u[t+1,x,nptz-1],0.)]
            bc1 = [Eq(u[t+1,x,-k],u[t+1,x,k]) for k in range(1,int(setup.sou/2)+1)]
                  
            if(save):
            
                usave = kwargs.get('usave')
                op    = Operator([stencil0, stencil1] + src_term + bc + bc1 + rec_term + [Eq(usave,u.forward)],subs=grid.spacing_map)
            
            else:
            
                op = Operator([stencil0, stencil1] + src_term + bc + bc1 + rec_term,subs=grid.spacing_map)
            
        elif(system=='adjoint'):
            
            pde1 = Eq(u.dt2 - u.laplace*vp*vp + vp*vp*damp*u.dtc.T)

            stencil0 =  Eq(u.backward, solve(pde0,u.backward),subdomain = grid.subdomains['d0'])
            stencil1 = [Eq(u.backward, solve(pde1,u.backward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]

            bc  = [Eq(u[t-1,0,z],0.),Eq(u[t-1,nptx-1,z],0.),Eq(u[t-1,x,nptz-1],0.)]
            bc1 = [Eq(u[t-1,x,-k],u[t-1,x,k]) for k in range(1,int(setup.sou/2)+1)]            
            src_term = src.interpolate(expr=u)
            rec_term = rec.inject(field=u.backward, expr=rec* dt**2*vp**2)
            op  = Operator([stencil0, stencil1] + rec_term + bc + bc1 + src_term,subs=grid.spacing_map)  

        elif(system=='gradient'):    
             
            grad  = kwargs.get('grad')
            usave = kwargs.get('usave')

            pde1 = Eq(u.dt2 - u.laplace*vp*vp + vp*vp*damp*u.dtc.T)

            stencil0 =  Eq(u.backward, solve(pde0,u.backward),subdomain = grid.subdomains['d0'])
            stencil1 = [Eq(u.backward, solve(pde1,u.backward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]

            bc  = [Eq(u[t-1,0,z],0.),Eq(u[t-1,nptx-1,z],0.),Eq(u[t-1,x,nptz-1],0.)]
            bc1 = [Eq(u[t-1,x,-k],u[t-1,x,k]) for k in range(1,int(setup.sou/2)+1)]
            src_term = src.interpolate(expr=u)
            rec_term = rec.inject(field=u.backward, expr=rec* dt**2*vp**2)
            grad_update = Eq(grad, grad - usave * u.dt2)
            op  = Operator([stencil0, stencil1] + bc + bc1 + rec_term + [grad_update],subs=grid.spacing_map)             
       
        else:
        
            assert "Invalid option"

        return op 

    def solvepml(self,rec,src,vp,geramdamp,vector,grid,setup,system,save=False,**kwargs):   

        nptx = setup.nptx
        nptz = setup.nptz
        u       = vector[0]
        phi1    = vector[1]
        phi2    = vector[2]
        (x,z)   = grid.dimensions     
        (hx,hz) = grid.spacing_map  
        t       = grid.stepping_dim
        dt      = grid.stepping_dim.spacing
        
        subds = ['d1','d2','d3']

        D01, D02, D11, D12 = geramdamp
        
        dampx0 = Function(name="dampx0", grid=grid,space_order=setup.sou,staggered=NODE ,dtype=np.float64)
        dampz0 = Function(name="dampz0", grid=grid,space_order=setup.sou,staggered=NODE ,dtype=np.float64)
        dampx0.data[:,:] = D01
        dampz0.data[:,:] = D02
        
        dampx1 = Function(name="dampx1", grid=grid,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
        dampz1 = Function(name="dampz1", grid=grid,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
        dampx1.data[0:nptx-1,0:nptz-1] = D11
        dampz1.data[0:nptx-1,0:nptz-1] = D12
        dampx1.data[nptx-1,0:nptz-1]   = dampx1.data[nptx-2,0:nptz-1]
        dampx1.data[0:nptx,nptz-1]     = dampx1.data[0:nptx,nptz-2]
        dampz1.data[nptx-1,0:nptz-1]   = dampz1.data[nptx-2,0:nptz-1]
        dampz1.data[0:nptx,nptz-1]     = dampz1.data[0:nptx,nptz-2]

        pde01   = Eq(u.dt2-u.laplace*vp[0]*vp[0]) 
                                                     
        if(system=='forward'):
            
            pde02a  = u.dt2   + (dampx0+dampz0)*u.dtc + (dampx0*dampz0)*u - u.laplace*vp[0]*vp[0] 
            pde02b  = - (0.5/hx)*(phi1[t,x,z-1]+phi1[t,x,z]-phi1[t,x-1,z-1]-phi1[t,x-1,z])
            pde02c  = - (0.5/hz)*(phi2[t,x-1,z]+phi2[t,x,z]-phi2[t,x-1,z-1]-phi2[t,x,z-1])
            pde02   = Eq(pde02a + pde02b + pde02c)

            pde10 = phi1.dt + dampx1*0.5*(phi1.forward+phi1)
            a1    = u[t+1,x+1,z] + u[t+1,x+1,z+1] - u[t+1,x,z] - u[t+1,x,z+1] 
            a2    = u[t,x+1,z]   + u[t,x+1,z+1]   - u[t,x,z]   - u[t,x,z+1] 
            pde11 = -(dampz1-dampx1)*0.5*(0.5/hx)*(a1+a2)*vp[1]*vp[1]
            pde1  = Eq(pde10+pde11)
                                                                
            pde20 = phi2.dt + dampz1*0.5*(phi2.forward+phi2) 
            b1    = u[t+1,x,z+1] + u[t+1,x+1,z+1] - u[t+1,x,z] - u[t+1,x+1,z] 
            b2    = u[t,x,z+1]   + u[t,x+1,z+1]   - u[t,x,z]   - u[t,x+1,z] 
            pde21 = -(dampx1-dampz1)*0.5*(0.5/hz)*(b1+b2)*vp[1]*vp[1]
            pde2  = Eq(pde20+pde21)

            stencil01 =  Eq(u.forward,solve(pde01,u.forward) ,subdomain = grid.subdomains['d0'])

            stencil02 = [Eq(u.forward,solve(pde02, u.forward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil1  = [Eq(phi1.forward, solve(pde1,phi1.forward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil2  = [Eq(phi2.forward, solve(pde2,phi2.forward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]

            bc  = [Eq(u[t+1,0,z],0.),Eq(u[t+1,nptx-1,z],0.),Eq(u[t+1,x,nptz-1],0.)]
            bc1 = [Eq(u[t+1,x,-k],u[t+1,x,k]) for k in range(1,int(setup.sou/2)+1)]            
            src_term = src.inject(field=u.forward,expr=src*dt**2*vp[0]**2)
            rec_term = rec.interpolate(expr=u)

            if(save):
                
                usave = kwargs.get('usave')
                op    = Operator([stencil01,stencil02] + src_term + bc + bc1 + [stencil1,stencil2] + rec_term + [Eq(usave,u.forward)],subs=grid.spacing_map)
            
            else:
            
                op = Operator([stencil01,stencil02] + src_term + bc + bc1 + [stencil1,stencil2] + rec_term,subs=grid.spacing_map)

        elif(system=='adjoint'):
            
            pde02a  = u.dt2   + (dampx0+dampz0)*u.dtc.T + (dampx0*dampz0)*u - u.laplace*vp[0]*vp[0] 
            pde02b  = - (0.5/hx)*(phi1[t,x,z-1]+phi1[t,x,z]-phi1[t,x-1,z-1]-phi1[t,x-1,z])
            pde02c  = - (0.5/hz)*(phi2[t,x-1,z]+phi2[t,x,z]-phi2[t,x-1,z-1]-phi2[t,x,z-1])
            pde02   = Eq(pde02a + pde02b + pde02c)

            pde10 = phi1.dt.T + dampx1*0.5*(phi1.backward+phi1)
            a1    = u[t-1,x+1,z] + u[t-1,x+1,z+1] - u[t-1,x,z] - u[t-1,x,z+1] 
            a2    = u[t,x+1,z]   + u[t,x+1,z+1]   - u[t,x,z]   - u[t,x,z+1] 
            pde11 = -(dampz1-dampx1)*0.5*(0.5/hx)*(a1+a2)*vp[1]*vp[1]
            pde1  = Eq(pde10+pde11)
                                                                
            pde20 = phi2.dt.T + dampz1*0.5*(phi2.backward+phi2) 
            b1    = u[t-1,x,z+1] + u[t-1,x+1,z+1] - u[t-1,x,z] - u[t-1,x+1,z] 
            b2    = u[t,x,z+1]   + u[t,x+1,z+1]   - u[t,x,z]   - u[t,x+1,z] 
            pde21 = -(dampx1-dampz1)*0.5*(0.5/hz)*(b1+b2)*vp[1]*vp[1]
            pde2  = Eq(pde20+pde21)

            stencil01 =  Eq(u.backward,solve(pde01,u.backward) ,subdomain = grid.subdomains['d0'])
            stencil02 = [Eq(u.backward,solve(pde02, u.backward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil1  = [Eq(phi1.backward, solve(pde1,phi1.backward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil2  = [Eq(phi2.backward, solve(pde2,phi2.backward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]

            bc  = [Eq(u[t-1,0,z],0.),Eq(u[t-1,nptx-1,z],0.),Eq(u[t-1,x,nptz-1],0.)]
            bc1 = [Eq(u[t-1,x,-k],u[t-1,x,k]) for k in range(1,int(setup.sou/2)+1)]            
            source_a  = src.interpolate(expr=u)
            receivers = rec.inject(field=u.backward, expr=rec*dt**2*vp[0]**2)
            op = Operator([stencil01,stencil02] + receivers + bc + bc1 + [stencil1, stencil2] + source_a,subs=grid.spacing_map)

        elif(system=='gradient'):    
                
            grad  = kwargs.get('grad')
            usave = kwargs.get('usave')

            pde02a  = u.dt2   + (dampx0+dampz0)*u.dtc.T + (dampx0*dampz0)*u - u.laplace*vp[0]*vp[0] 
            pde02b  = - (0.5/hx)*(phi1[t,x,z-1]+phi1[t,x,z]-phi1[t,x-1,z-1]-phi1[t,x-1,z])
            pde02c  = - (0.5/hz)*(phi2[t,x-1,z]+phi2[t,x,z]-phi2[t,x-1,z-1]-phi2[t,x,z-1])
            pde02   = Eq(pde02a + pde02b + pde02c)

            pde10 = phi1.dt.T + dampx1*0.5*(phi1.backward+phi1)
            a1    = u[t-1,x+1,z] + u[t-1,x+1,z+1] - u[t-1,x,z] - u[t-1,x,z+1] 
            a2    = u[t,x+1,z]   + u[t,x+1,z+1]   - u[t,x,z]   - u[t,x,z+1] 
            pde11 = -(dampz1-dampx1)*0.5*(0.5/hx)*(a1+a2)*vp[1]*vp[1]
            pde1  = Eq(pde10+pde11)
                                                                
            pde20 = phi2.dt.T + dampz1*0.5*(phi2.backward+phi2) 
            b1    = u[t-1,x,z+1] + u[t-1,x+1,z+1] - u[t-1,x,z] - u[t-1,x+1,z] 
            b2    = u[t,x,z+1]   + u[t,x+1,z+1]   - u[t,x,z]   - u[t,x+1,z] 
            pde21 = -(dampx1-dampz1)*0.5*(0.5/hz)*(b1+b2)*vp[1]*vp[1]
            pde2  = Eq(pde20+pde21)

            stencil01 =  Eq(u.backward,solve(pde01,u.backward) ,subdomain = grid.subdomains['d0'])
            stencil02 = [Eq(u.backward,solve(pde02, u.backward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil1 = [Eq(phi1.backward, solve(pde1,phi1.backward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil2 = [Eq(phi2.backward, solve(pde2,phi2.backward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]

            bc  = [Eq(u[t-1,0,z],0.),Eq(u[t-1,nptx-1,z],0.),Eq(u[t-1,x,nptz-1],0.)]
            bc1 = [Eq(u[t-1,x,-k],u[t-1,x,k]) for k in range(1,int(setup.sou/2)+1)]          
            source_a = src.interpolate(expr=u)
            receivers   = rec.inject(field=u.backward, expr=rec*dt**2*vp[0]**2)
            grad_update = Eq(grad, grad - usave * u.dt2)
            op = Operator([stencil01,stencil02] + bc + bc1 + [stencil1, stencil2] + receivers + [grad_update],subs=grid.spacing_map)

        else:
            
            assert "Invalid option"
        
        return op

    def solvehabcA1(self,rec,src,vp,gerapesos,u,grid,system,save=False,**kwargs):

        (hx,hz) = grid.spacing_map 
        (x, z)  = grid.dimensions     
        t       = grid.stepping_dim
        dt      = grid.stepping_dim.spacing
    
        Mpesosx,Mpesosz = gerapesos
        
        pesosx = Function(name="pesosx",grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)
        pesosx.data[:,:] = Mpesosx[:,:]

        pesosz = Function(name="pesosz",grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)
        pesosz.data[:,:] = Mpesosz[:,:]

        u1  = Function(name="u1"   ,grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)
        u2  = Function(name="u2"   ,grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)
        u3  = Function(name="u3"   ,grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)

        pde0 = Eq(u.dt2 - u.laplace*vp*vp)
        
        if(system=='forward'):
            
            stencil01 = [Eq(u1,u.forward),Eq(u2,u),Eq(u3,u.forward)]
            stencil02 = [Eq(u3,u.forward)]  
            stencil0  = Eq(u.forward, solve(pde0,u.forward))
            
            # Região B_{1}
            aux1      = ((-vp[x,z]*dt+hx)*u2[x,z] + (vp[x,z]*dt+hx)*u2[x+1,z] + (vp[x,z]*dt-hx)*u3[x+1,z])/(vp[x,z]*dt+hx)
            pde1      = (1-pesosx[x,z])*u3[x,z] + pesosx[x,z]*aux1
            stencil1  = Eq(u.forward,pde1,subdomain = grid.subdomains['d1'])

            # Região B_{3}
            aux2      = ((-vp[x,z]*dt+hx)*u2[x,z] + (vp[x,z]*dt+hx)*u2[x-1,z] + (vp[x,z]*dt-hx)*u3[x-1,z])/(vp[x,z]*dt+hx)
            pde2      = (1-pesosx[x,z])*u3[x,z] + pesosx[x,z]*aux2
            stencil2  = Eq(u.forward,pde2,subdomain = grid.subdomains['d2'])

            # Região B_{2}
            aux3      = ((-vp[x,z]*dt+hz)*u2[x,z] + (vp[x,z]*dt+hz)*u2[x,z-1] + (vp[x,z]*dt-hz)*u3[x,z-1])/(vp[x,z]*dt+hz)
            pde3      = (1-pesosz[x,z])*u3[x,z] + pesosz[x,z]*aux3
            stencil3  = Eq(u.forward,pde3,subdomain = grid.subdomains['d3'])

            bc  = []
            bc1 = [Eq(u[t+1,x,-k],u[t+1,x,k]) for k in range(1,int(setup.sou/2)+1)]            
                
            src_term = src.inject(field=u.forward,expr=src*dt**2*vp**2)
            rec_term = rec.interpolate(expr=u)
            
            if(save):
                
                usave = kwargs.get('usave')
                op    = Operator([stencil0] + src_term + [stencil01,stencil3,stencil02,stencil2,stencil1] + bc + bc1 + rec_term + [Eq(usave,u.forward)],subs=grid.spacing_map)
            
            else:
            
                op = Operator([stencil0] + src_term + [stencil01,stencil3,stencil02,stencil2,stencil1] + bc + bc1 + rec_term,subs=grid.spacing_map)

        elif(system=='adjoint'):
            
            stencil01 = [Eq(u1,u.backward),Eq(u2,u),Eq(u3,u.backward)]
            stencil02 = [Eq(u3,u.backward)]  
            stencil0  = Eq(u.backward, solve(pde0,u.backward))
            
            # Região B_{1}
            aux1      = ((-vp[x,z]*dt+hx)*u2[x,z] + (vp[x,z]*dt+hx)*u2[x+1,z] + (vp[x,z]*dt-hx)*u3[x+1,z])/(vp[x,z]*dt+hx)
            pde1      = (1-pesosx[x,z])*u3[x,z] + pesosx[x,z]*aux1
            stencil1  = Eq(u.backward,pde1,subdomain = grid.subdomains['d1'])

            # Região B_{3}
            aux2      = ((-vp[x,z]*dt+hx)*u2[x,z] + (vp[x,z]*dt+hx)*u2[x-1,z] + (vp[x,z]*dt-hx)*u3[x-1,z])/(vp[x,z]*dt+hx)
            pde2      = (1-pesosx[x,z])*u3[x,z] + pesosx[x,z]*aux2
            stencil2  = Eq(u.backward,pde2,subdomain = grid.subdomains['d2'])

            # Região B_{2}
            aux3      = ((-vp[x,z]*dt+hz)*u2[x,z] + (vp[x,z]*dt+hz)*u2[x,z-1] + (vp[x,z]*dt-hz)*u3[x,z-1])/(vp[x,z]*dt+hz)
            pde3      = (1-pesosz[x,z])*u3[x,z] + pesosz[x,z]*aux3
            stencil3  = Eq(u.backward,pde3,subdomain = grid.subdomains['d3'])

            receivers = rec.inject(field=u.backward, expr=rec*dt**2*vp**2)
            source_a = src.interpolate(expr=u)
            
            bc  = []
            bc1 = [Eq(u[t-1,x,-k],u[t-1,x,k]) for k in range(1,int(setup.sou/2)+1)]            
            op  = Operator([stencil0] + receivers + [stencil01,stencil3,stencil02,stencil2,stencil1] + bc + bc1 + source_a,subs=grid.spacing_map)
            
        elif(system=='gradient'):    
             
            grad  = kwargs.get('grad')
            usave = kwargs.get('usave')

            stencil01 = [Eq(u1,u.backward),Eq(u2,u),Eq(u3,u.backward)]
            stencil02 = [Eq(u3,u.backward)]  
            stencil0  = Eq(u.backward, solve(pde0,u.backward))
            
            # Região B_{1}
            aux1      = ((-vp[x,z]*dt+hx)*u2[x,z] + (vp[x,z]*dt+hx)*u2[x+1,z] + (vp[x,z]*dt-hx)*u3[x+1,z])/(vp[x,z]*dt+hx)
            pde1      = (1-pesosx[x,z])*u3[x,z] + pesosx[x,z]*aux1
            stencil1  = Eq(u.backward,pde1,subdomain = grid.subdomains['d1'])

            # Região B_{3}
            aux2      = ((-vp[x,z]*dt+hx)*u2[x,z] + (vp[x,z]*dt+hx)*u2[x-1,z] + (vp[x,z]*dt-hx)*u3[x-1,z])/(vp[x,z]*dt+hx)
            pde2      = (1-pesosx[x,z])*u3[x,z] + pesosx[x,z]*aux2
            stencil2  = Eq(u.backward,pde2,subdomain = grid.subdomains['d2'])

            # Região B_{2}
            aux3      = ((-vp[x,z]*dt+hz)*u2[x,z] + (vp[x,z]*dt+hz)*u2[x,z-1] + (vp[x,z]*dt-hz)*u3[x,z-1])/(vp[x,z]*dt+hz)
            pde3      = (1-pesosz[x,z])*u3[x,z] + pesosz[x,z]*aux3
            stencil3  = Eq(u.backward,pde3,subdomain = grid.subdomains['d3'])

            receivers = rec.inject(field=u.backward, expr=rec*dt**2*vp**2)
            source_a  = src.interpolate(expr=u)
            bc  = []
            bc1 = [Eq(u[t-1,x,-k],u[t-1,x,k]) for k in range(1,int(setup.sou/2)+1)]            
            grad_update = Eq(grad, grad - usave * u.dt2)
            op  = Operator([stencil0] + bc + bc1 +[stencil01,stencil3,stencil02,stencil2,stencil1] + receivers + [grad_update],subs=grid.spacing_map)
             
        else:
            
            assert "Invalid option"

        return op

    def solvecpml(self,rec,src,vp,geradamp,vector,grid, setup,system,save=False,**kwargs):   
        
        u     = vector[0]
        phi1  = vector[1]
        phi2  = vector[2]
        zeta1 = vector[3]
        zeta2 = vector[4]
  
        nptx = setup.nptx
        nptz = setup.nptz
        (x,z)   = grid.dimensions     
        (hx,hz) = grid.spacing_map  
        t       = grid.stepping_dim
        dt      = grid.stepping_dim.spacing
        
        subds = ['d1','d2','d3']

        D01, D02 = geradamp[0]
        A1C, A2C, B1C, B2C, alpha1v, alpha2v = geradamp[1]
        
        dampx0 = Function(name="dampx0", grid=grid,space_order=setup.sou,staggered=NODE ,dtype=np.float64)
        dampz0 = Function(name="dampz0", grid=grid,space_order=setup.sou,staggered=NODE ,dtype=np.float64)
        dampx0.data[:,:] = D01
        dampz0.data[:,:] = D02
               
        
        alpha1 = Function(name="alpha1", grid=grid,space_order=2,staggered=NODE ,dtype=np.float64)
        alpha2 = Function(name="alpha2", grid=grid,space_order=2,staggered=NODE ,dtype=np.float64)
        alpha1.data[:,:] = alpha1v
        alpha2.data[:,:] = alpha2v
        a1w = Function(name="a1w", grid=grid,space_order=2,staggered=NODE ,dtype=np.float64)
        a1w.data[:,:] = A1C

        a2w = Function(name="a2w", grid=grid,space_order=2,staggered=NODE ,dtype=np.float64)
        a2w.data[:,:] = A2C

        b1w = Function(name="b1w", grid=grid,space_order=2,staggered=NODE ,dtype=np.float64)
        b1w.data[:,:] = B1C

        b2w = Function(name="b2w", grid=grid,space_order=2,staggered=NODE ,dtype=np.float64)
        b2w.data[:,:] = B2C

        
        pde01   = Eq(u.dt2-u.laplace*vp*vp) 
                                                     
        if(system=='forward'):
            
            pde02  = u.dt2 + vp*vp*(-u.laplace -zeta1 -zeta2 -(1/hx)*(phi1[t,x,z]-phi1[t,x-1,z]) -(1/hz)*(phi2[t,x,z]-phi2[t,x,z-1]))

            pde1 = a1w*phi1  + b1w*(1/hx)*(1/2)*(u[t+1,x+1,z]-u[t+1,x,z]+u[t,x+1,z]-u[t,x,z])
            pde2 = a2w*phi2  + b2w*(1/hz)*(1/2)*(u[t+1,x,z+1]-u[t+1,x,z]+u[t,x,z+1]-u[t,x,z])

            pde3 = a1w*zeta1 + b1w*((1/hx)*(phi1[t+1,x,z]-phi1[t+1,x-1,z]) + u.forward.dx2)
            pde4 = a2w*zeta2 + b2w*((1/hz)*(phi2[t+1,x,z]-phi2[t+1,x,z-1]) + u.forward.dy2)

            stencil01 =  Eq(u.forward,solve(pde01,u.forward) ,subdomain = grid.subdomains['d0'])
            stencil02 = [Eq(u.forward,solve(pde02, u.forward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil1 = [Eq(phi1.forward,  pde1,subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil2 = [Eq(phi2.forward,  pde2,subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil3 = [Eq(zeta1.forward, pde3,subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil4 = [Eq(zeta2.forward, pde4,subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            #==============================================================================

            bc = [Eq(u[t+1,0,z],0.),Eq(u[t+1,nptx-1,z],0.),Eq(u[t+1,x,nptz-1],0.)]
            bc1 = [Eq(u[t+1,x,-k],u[t+1,x,k]) for k in range(1,int(setup.sou/2)+1)]  
            bczeta  = [Eq(zeta1[t+1,0,z],zeta1[t+1,1,z]),Eq(zeta1[t+1,nptx-1,z],zeta1[t+1,nptx-2,z])]
            bczeta += [Eq(zeta1[t+1,x,0],zeta1[t+1,x,1]),Eq(zeta1[t+1,x,nptz-1],zeta1[t+1,x,nptz-2])]
            bczeta += [Eq(zeta2[t+1,0,z],zeta2[t+1,1,z]),Eq(zeta2[t+1,nptx-1,z],zeta2[t+1,nptx-2,z])]
            bczeta += [Eq(zeta2[t+1,x,0],zeta2[t+1,x,1]),Eq(zeta2[t+1,x,nptz-1],zeta2[t+1,x,nptz-2])]
                    
            src_term = src.inject(field=u.forward,expr=src*dt**2*vp**2)
            rec_term = rec.interpolate(expr=u)

            if(save):
                
                usave = kwargs.get('usave')
                op = Operator([stencil01,stencil02] + src_term + bc + bc1 + [stencil1,stencil2,stencil3,stencil4] + bczeta + rec_term + [Eq(usave,u.forward)],subs=grid.spacing_map)           
            
            else:

                op = Operator([stencil01,stencil02] + src_term + bc + bc1 + [stencil1,stencil2,stencil3,stencil4] + bczeta + rec_term,subs=grid.spacing_map)

        elif(system=='gradient'):  
            grad  = kwargs.get('grad')
            usave = kwargs.get('usave')  
                
            pde02  = u.dt2 + vp*vp*(-u.laplace -zeta1 -zeta2 -(1/hx)*(phi1[t,x,z]-phi1[t,x-1,z]) -(1/hz)*(phi2[t,x,z]-phi2[t,x,z-1]))

            pde1 = a1w*phi1  + b1w*(1/hx)*(1/2)*(u[t-1,x+1,z]-u[t-1,x,z]+u[t,x+1,z]-u[t,x,z])
            pde2 = a2w*phi2  + b2w*(1/hz)*(1/2)*(u[t-1,x,z+1]-u[t-1,x,z]+u[t,x,z+1]-u[t,x,z])

            pde3 = a1w*zeta1 + b1w*((1/hx)*(phi1[t-1,x,z]-phi1[t-1,x-1,z]) + u.forward.dx2)
            pde4 = a2w*zeta2 + b2w*((1/hz)*(phi2[t-1,x,z]-phi2[t-1,x,z-1]) + u.forward.dy2)

            stencil01 =  Eq(u.backward,solve(pde01,u.backward) ,subdomain = grid.subdomains['d0'])
            stencil02 = [Eq(u.backward,solve(pde02, u.backward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil1 = [Eq(phi1.backward,  pde1,subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil2 = [Eq(phi2.backward,  pde2,subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil3 = [Eq(zeta1.backward, pde3,subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            stencil4 = [Eq(zeta2.backward, pde4,subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]
            
            bc = [Eq(u[t-1,0,z],0.),Eq(u[t-1,nptx-1,z],0.),Eq(u[t-1,x,nptz-1],0.)]
            bc1 = [Eq(u[t-1,x,-k],u[t-1,x,k]) for k in range(1,int(setup.sou/2)+1)]  
            bczeta  = [Eq(zeta1[t-1,0,z],zeta1[t-1,1,z]),Eq(zeta1[t-1,nptx-1,z],zeta1[t-1,nptx-2,z])]
            bczeta += [Eq(zeta1[t-1,x,0],zeta1[t-1,x,1]),Eq(zeta1[t-1,x,nptz-1],zeta1[t-1,x,nptz-2])]
            bczeta += [Eq(zeta2[t-1,0,z],zeta2[t-1,1,z]),Eq(zeta2[t-1,nptx-1,z],zeta2[t-1,nptx-2,z])]
            bczeta += [Eq(zeta2[t-1,x,0],zeta2[t-1,x,1]),Eq(zeta2[t-1,x,nptz-1],zeta2[t-1,x,nptz-2])]
                    
            receivers = rec.inject(field=u.backward, expr=rec*dt**2*vp**2)
            grad_update = Eq(grad, grad - usave * u.dt2)

            op = Operator([stencil01,stencil02]  + bc + bc1 + [stencil1,stencil2,stencil3,stencil4] + bczeta + receivers + [grad_update],subs=grid.spacing_map)           
  
        return op
    
class FWISolver():

    def __init__(self,set_time,setup,setting,grid,utils,v0):  

        self.dt0, self.nt, self.time_range = set_time  #time discretization
        self.setting = setting
        self.grid    = grid
        self.setup   = setup

        self.abc = setting["Abcs"]
        #==============================================================================
        # Solver Settigns
        #==============================================================================          
        if(self.abc=='damping'):
        
            self.g        = utils.geramdamp(self.setup,v0,self.abc)
            self.solv     = solverABCs.solvedamp

        elif(self.abc=='pml'):
            self.g        = utils.geramdamp(self.setup,v0,self.abc)
            self.solv     = solverABCs.solvepml
        
        elif(self.abc=='cpml'):
            
            g1       = utils.geramdamp(self.setup,v0,self.abc)
            g2       = utils.gerapesoscpml(self.setup,v0,g1,self.dt0)
            self.solv     = solverABCs.solvecpml
            self.g        = [g1, g2]
              
        elif(self.abc=='habc-a1'):
            habcw    = setting["habcw"]
            self.g        = utils.gerapesos(self.setup,habcw)
            self.solv     = solverABCs.solvehabcA1
    
    #==============================================================================
    # FWI Function
    #==============================================================================    
    def apply(self,sn,grad,sd,vp,vp_guess):
        setting = self.setting
        nt      = self.nt
        setup   = self.setup
        grid    = self.grid
        solv    = self.solv
        dt0     = self.dt0
        g       = self.g
        abc     = self.abc
        (x, z)  = grid.dimensions

        if not setting["checkpointing"]:
            # Saves Parameters    
            nsnaps = int(nt/setting["jump"])
            factor  = mt.ceil(nt/nsnaps) + 1
            time_subsampled = ConditionalDimension('t_sub', parent=grid.time_dim, factor=factor)
            usave = TimeFunction(name='usave', grid=grid, time_order=2, space_order=2,save=nsnaps, time_dim=time_subsampled)

        # Receivers Parameters
        nrec = setting["rec_n"] #receivers numbers
        rec  = Receiver(name='rec',grid=grid,npoint=nrec,time_range=self.time_range,staggered=NODE,dtype=np.float64)
        rec.coordinates.data[:, 0] = np.linspace(setup.x0pml,setup.x1pml,nrec)
        rec.coordinates.data[:, 1] = setting["recposition_z"]

        recg  = Receiver(name='recg',grid=grid,npoint=nrec,time_range=self.time_range,staggered=NODE,dtype=np.float64)
        recg.coordinates.data[:, 0] = np.linspace(setup.x0pml,setup.x1pml,nrec)
        recg.coordinates.data[:, 1] = setting["recposition_z"]

        residual  = Receiver(name='residual',grid=grid,npoint=nrec,time_range=self.time_range,staggered=NODE,dtype=np.float64)
        residual.coordinates.data[:, 0] = np.linspace(setup.x0pml,setup.x1pml,nrec)
        residual.coordinates.data[:, 1] = setting["recposition_z"]

        # Source Prameters
        src = RickerSource(name='src',grid=grid,f0=setting["f0"],npoint=1,time_range=self.time_range,staggered=NODE,dtype=np.float64)

        # The shots start at the position 2*sd in the physical domain
        xposf = setting["x0"] + 2*sd + sd*sn  
        src.coordinates.data[:, 0] = xposf
        src.coordinates.data[:, 1] = setting["shotposition_z"] 

        
        u    = TimeFunction(name="u",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64) 
        v    = TimeFunction(name="v",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64) 
        u0   = TimeFunction(name="u0",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64) 
        

        if abc=='pml' or abc=='cpml':
                
            phi1 = TimeFunction(name="phi1",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
            phi2 = TimeFunction(name="phi2",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
            
            phi1_adj = TimeFunction(name="phi1_adj",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
            phi2_adj = TimeFunction(name="phi2_adj",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)

            phi10 = TimeFunction(name="phi10",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
            phi20 = TimeFunction(name="phi20",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)

            if abc=='cpml':
                
                zeta1 = TimeFunction(name="zeta1",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
                zeta2 = TimeFunction(name="zeta2",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
                
                zeta1_adj = TimeFunction(name="zeta1_adj",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
                zeta2_adj = TimeFunction(name="zeta2_adj",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
    
                vector      = [u,phi1,phi2,zeta1,zeta2]      
                vector_adj  = [v,phi1_adj, phi2_adj,zeta1_adj,zeta2_adj]

                zeta10   = TimeFunction(name="zeta10",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
                zeta20   = TimeFunction(name="zeta20",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
                vector0  = [u0,phi10,phi20,zeta10,zeta20]

            if abc=='pml':
                vector     = [u,phi1, phi2]
                vector_adj = [v,phi1_adj, phi2_adj]
                vector0    = [u0,phi10, phi20]
            
        else:
            vector0    = u0
            vector     = u
            vector_adj = v


        # Forward solver using true model
        op_fw = solv(self,rec,src,vp,g,vector0,grid,setup,system='forward')
        op_fw( dt=dt0)

        if setting["checkpointing"]:
            cp = DevitoCheckpoint([u])
            n_checkpoints = 10

            # Forward solver -- Wrapper
            op_fw_guess = solv(self,recg, src, vp_guess,g,vector,grid,setup,system='forward') 
            
            # Adjoint-based gradient solver -- Wrapper
            op_bw = solv(self,residual, src, vp_guess,g,vector_adj,grid,setup,system='gradient',grad=grad,usave=u)

            wrap_fw  = CheckpointOperator(op_fw_guess,dt=dt0)

            wrap_rev = CheckpointOperator(op_bw,dt=dt0)

            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt-1)

            # Forward solver
            wrp.apply_forward()

            # Difference betwen true and guess model
            residual.data[:]=rec.data[:]-recg.data[:] # residual used as a forcing in the adjoint eq.

            # Backward solver
            wrp.apply_reverse()

        else:
           
            # Forward solver -- Wrapper
            op_fw_guess = solv(self,recg, src, vp_guess,g,vector,grid,setup,system='forward',save=True, usave=usave) 
            
            # Adjoint-based gradient solver -- Wrapper
            op_bw = solv(self,residual, src, vp_guess,g,vector_adj,grid,setup,system='gradient',grad=grad,usave=usave)
        
            op_fw_guess(dt=dt0)

            # Difference betwen true and guess model
            residual.data[:]=rec.data[:]-recg.data[:] # residual used as a forcing in the adjoint eq.
        
            op_bw(dt=dt0)


        J = 0.5*np.linalg.norm(residual.data.flatten())**2
        
        return J
    #==============================================================================    
    