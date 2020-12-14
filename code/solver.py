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
#==============================================================================

#==============================================================================
# Damping Solver 
#==============================================================================

#==============================================================================    
    def solvedamp(rec,src,vp,geramdamp,u,grid,setup,system,save=False,**kwargs):    
#==============================================================================
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

            vsave = kwargs.get('vsave')
            pde1 = Eq(u.dt2 - u.laplace*vp*vp + vp*vp*damp*u.dtc.T)

            stencil0 =  Eq(u.backward, solve(pde0,u.backward),subdomain = grid.subdomains['d0'])
            stencil1 = [Eq(u.backward, solve(pde1,u.backward),subdomain = grid.subdomains[subds[i]]) for i in range(0,len(subds))]

            bc  = [Eq(u[t-1,0,z],0.),Eq(u[t-1,nptx-1,z],0.),Eq(u[t-1,x,nptz-1],0.)]
            bc1 = [Eq(u[t-1,x,-k],u[t-1,x,k]) for k in range(1,int(setup.sou/2)+1)]
            src_term = src.interpolate(expr=u)
            rec_term = rec.inject(field=u.backward, expr=rec* dt**2*vp**2)

            op  = Operator([stencil0, stencil1] + bc + bc1 + rec_term + [Eq(vsave,u.backward)],subs=grid.spacing_map)             
       
        else:
        
            assert "Invalid option"

        return op 
#==============================================================================

#==============================================================================
# PML Solver 
#==============================================================================

#==============================================================================
    def solvepml(rec,src,vp,geramdamp,vector,grid,setup,system,save=False,**kwargs):   

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
                
            vsave = kwargs.get('vsave')

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
           
            op = Operator([stencil01,stencil02] + bc + bc1 + [stencil1, stencil2] + receivers + [Eq(vsave,u.backward)],subs=grid.spacing_map)

        else:
            
            assert "Invalid option"
        
        return op
#==============================================================================

#==============================================================================
# HABC Solver 
#==============================================================================

#==============================================================================
    def solvehabc(rec,src,vp,gerapesos,u,grid,setup,system,save=False,**kwargs):
        
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
        
        if(system=='forward' and setup.Abcs=='habc-a1'):
            
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

        if(system=='forward' and setup.Abcs=='Higdon'):
            stencil0  = Eq(u.forward, solve(pde0,u.forward))
            stencil01 = [Eq(u1,u.backward),Eq(u2,u),Eq(u3,u.forward)]
            stencil02 = [Eq(u3,u.forward)]
            
            alpha1 = 0.0
            alpha2 = np.pi/4
            a1 = 0.5
            b1 = 0.5
            a2 = 0.5
            b2 = 0.5

            # Região B_{1}
            gama111 = np.cos(alpha1)*(1-a1)*(1/dt)
            gama121 = np.cos(alpha1)*(a1)*(1/dt)
            gama131 = np.cos(alpha1)*(1-b1)*(1/hx)*vp[x,z]
            gama141 = np.cos(alpha1)*(b1)*(1/hx)*vp[x,z]
                    
            gama211 = np.cos(alpha2)*(1-a2)*(1/dt)
            gama221 = np.cos(alpha2)*(a2)*(1/dt)
            gama231 = np.cos(alpha2)*(1-b2)*(1/hx)*vp[x,z]
            gama241 = np.cos(alpha2)*(b2)*(1/hx)*vp[x,z]
            
            c111 =  gama111 + gama131
            c121 = -gama111 + gama141
            c131 =  gama121 - gama131
            c141 = -gama121 - gama141
                    
            c211 =  gama211 + gama231
            c221 = -gama211 + gama241
            c231 =  gama221 - gama231
            c241 = -gama221 - gama241

            aux1      = ( u2[x,z]*(-c111*c221-c121*c211) + u3[x+1,z]*(-c111*c231-c131*c211) + u2[x+1,z]*(-c111*c241-c121*c231-c141*c211-c131*c221) 
                      + u1[x,z]*(-c121*c221) + u1[x+1,z]*(-c121*c241-c141*c221) + u3[x+2,z]*(-c131*c231) +u2[x+2,z]*(-c131*c241-c141*c231)
                      + u1[x+2,z]*(-c141*c241))/(c111*c211)
            pde1      = (1-pesosx[x,z])*u3[x,z] + pesosx[x,z]*aux1
            stencil1  = Eq(u.forward,pde1,subdomain = grid.subdomains['d1'])

            # Região B_{3}
            gama112 = np.cos(alpha1)*(1-a1)*(1/dt)
            gama122 = np.cos(alpha1)*(a1)*(1/dt)
            gama132 = np.cos(alpha1)*(1-b1)*(1/hx)*vp[x,z]
            gama142 = np.cos(alpha1)*(b1)*(1/hx)*vp[x,z]
                    
            gama212 = np.cos(alpha2)*(1-a2)*(1/dt)
            gama222 = np.cos(alpha2)*(a2)*(1/dt)
            gama232 = np.cos(alpha2)*(1-b2)*(1/hx)*vp[x,z]
            gama242 = np.cos(alpha2)*(b2)*(1/hx)*vp[x,z]
                    
            c112 =  gama112 + gama132
            c122 = -gama112 + gama142
            c132 =  gama122 - gama132
            c142 = -gama122 - gama142

            c212 =  gama212 + gama232
            c222 = -gama212 + gama242
            c232 =  gama222 - gama232
            c242 = -gama222 - gama242

            aux2      = ( u2[x,z]*(-c112*c222-c122*c212) + u3[x-1,z]*(-c112*c232-c132*c212) + u2[x-1,z]*(-c112*c242-c122*c232-c142*c212-c132*c222) 
                      + u1[x,z]*(-c122*c222) + u1[x-1,z]*(-c122*c242-c142*c222) + u3[x-2,z]*(-c132*c232) +u2[x-2,z]*(-c132*c242-c142*c232)
                      + u1[x-2,z]*(-c142*c242))/(c112*c212)
            pde2      = (1-pesosx[x,z])*u3[x,z] + pesosx[x,z]*aux2
            stencil2  = Eq(u.forward,pde2,subdomain = grid.subdomains['d2'])

            # Região B_{2}
            gama113 = np.cos(alpha1)*(1-a1)*(1/dt)
            gama123 = np.cos(alpha1)*(a1)*(1/dt)
            gama133 = np.cos(alpha1)*(1-b1)*(1/hz)*vp[x,z]
            gama143 = np.cos(alpha1)*(b1)*(1/hz)*vp[x,z]

            gama213 = np.cos(alpha2)*(1-a2)*(1/dt)
            gama223 = np.cos(alpha2)*(a2)*(1/dt)
            gama233 = np.cos(alpha2)*(1-b2)*(1/hz)*vp[x,z]
            gama243 = np.cos(alpha2)*(b2)*(1/hz)*vp[x,z]
                    
            c113 =  gama113 + gama133
            c123 = -gama113 + gama143
            c133 =  gama123 - gama133
            c143 = -gama123 - gama143
                    
            c213 =  gama213 + gama233
            c223 = -gama213 + gama243
            c233 =  gama223 - gama233
            c243 = -gama223 - gama243

            aux3      = ( u2[x,z]*(-c113*c223-c123*c213) + u3[x,z-1]*(-c113*c233-c133*c213) + u2[x,z-1]*(-c113*c243-c123*c233-c143*c213-c133*c223) 
                        + u1[x,z]*(-c123*c223) + u1[x,z-1]*(-c123*c243-c143*c223) + u3[x,z-2]*(-c133*c233) +u2[x,z-2]*(-c133*c243-c143*c233)
                        + u1[x,z-2]*(-c143*c243))/(c113*c213)
            pde3      = (1-pesosz[x,z])*u3[x,z] + pesosz[x,z]*aux3
            stencil3  = Eq(u.forward,pde3,subdomain = grid.subdomains['d3'])


            bc  = []
            bc1 = [Eq(u[t+1,x,-k],u[t+1,x,k]) for k in range(1,int(setup.sou/2)+1)]            
                
            src_term = src.inject(field=u.forward,expr=src*dt**2*vp**2)
            rec_term = rec.interpolate(expr=u)
            
            if(save):
                
                usave = kwargs.get('usave')
                op = Operator([stencil0] + src_term + [stencil01,stencil3,stencil02,stencil2,stencil1] + bc + bc1 + rec_term + [Eq(usave,u.forward)],subs=grid.spacing_map)
            
            else:
            
                op = Operator([stencil0] + src_term + [stencil01,stencil3,stencil02,stencil2,stencil1] + bc + bc1 + rec_term,subs=grid.spacing_map)

        if(system=='adjoint' and setup.Abcs=='habc-a1'):    
             
            vsave = kwargs.get('vsave')

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
            
            op  = Operator([stencil0] + bc + bc1 +[stencil01,stencil3,stencil02,stencil2,stencil1] + receivers + [Eq(vsave,u.backward)],subs=grid.spacing_map)

        if(system=='adjoint' and setup.Abcs=='Higdon'):
            vsave = kwargs.get('vsave')

            stencil01 = [Eq(u1,u.backward),Eq(u2,u),Eq(u3,u.backward)]
            stencil02 = [Eq(u3,u.backward)]  
            stencil0  = Eq(u.backward, solve(pde0,u.backward))
            alpha1 = 0.0
            alpha2 = np.pi/4
            coef1  =  np.cos(alpha1)*np.cos(alpha2)
            coef2  =  np.cos(alpha1) + np.cos(alpha2)

            # Região B_{1}
            cte11 =  coef1*(1/(2*dt**2)) + coef2*(1/(2*dt*hx))*vp[x,z]
            cte21 = -coef1*(1/(2*dt**2)) + coef2*(1/(2*dt*hx))*vp[x,z] - (1/(2*hz**2))*vp[x,z]*vp[x,z] 
            cte31 = -coef1*(1/(2*dt**2)) - coef2*(1/(2*dt*hx))*vp[x,z]
            cte41 =  coef1*(1/(dt**2))
            cte51 =  (1/(4*hz**2))*vp[x,z]*vp[x,z]

            aux1      = (cte21*(u3[x+1,z] + u1[x,z]) + cte31*u1[x+1,z] + cte41*(u2[x,z]+u2[x+1,z]) + cte51*(u3[x+1,z+1] + u3[x+1,z-1] + u1[x,z+1] + u1[x,z-1]))/cte11
            pde1      = (1-pesosx[x,z])*u3[x,z] + pesosx[x,z]*aux1
            stencil1  = Eq(u.backward,pde1,subdomain = grid.subdomains['d1'])
            
            # Região B_{3}
            cte12 =  coef1*(1/(2*dt**2)) + coef2*(1/(2*dt*hx))*vp[x,z]
            cte22 = -coef1*(1/(2*dt**2)) + coef2*(1/(2*dt*hx))*vp[x,z] - (1/(2*hz**2))*vp[x,z]*vp[x,z] 
            cte32 = -coef1*(1/(2*dt**2)) - coef2*(1/(2*dt*hx))*vp[x,z]
            cte42 =  coef1*(1/(dt**2))
            cte52 =  (1/(4*hz**2))*vp[x,z]*vp[x,z]
  
            aux2      = (cte22*(u3[x-1,z] + u1[x,z]) + cte32*u1[x-1,z] + cte42*(u2[x,z]+u2[x-1,z]) + cte52*(u3[x-1,z+1] + u3[x-1,z-1] + u1[x,z+1] + u1[x,z-1]))/cte12
            pde2      = (1-pesosx[x,z])*u3[x,z] + pesosx[x,z]*aux2
            stencil2  = Eq(u.backward,pde2,subdomain = grid.subdomains['d2'])

            # Região B_{2}
            cte13 =  coef1*(1/(2*dt**2)) + coef2*(1/(2*dt*hz))*vp[x,z]
            cte23 = -coef1*(1/(2*dt**2)) + coef2*(1/(2*dt*hz))*vp[x,z] - (1/(2*hx**2))*vp[x,z]*vp[x,z] 
            cte33 = -coef1*(1/(2*dt**2)) - coef2*(1/(2*dt*hz))*vp[x,z]
            cte43 =  coef1*(1/(dt**2))
            cte53 =  (1/(4*hx**2))*vp[x,z]*vp[x,z]

            aux3      = (cte23*(u3[x,z-1] + u1[x,z]) + cte33*u1[x,z-1] + cte43*(u2[x,z]+u2[x,z-1]) + cte53*(u3[x+1,z-1] + u3[x-1,z-1] + u1[x+1,z] + u1[x-1,z]))/cte13
            pde3      = (1-pesosz[x,z])*u3[x,z] + pesosz[x,z]*aux3
            stencil3  = Eq(u.backward,pde3,subdomain = grid.subdomains['d3'])

            receivers = rec.inject(field=u.backward, expr=rec*dt**2*vp**2)

            bc  = []
            bc1 = [Eq(u[t-1,x,-k],u[t-1,x,k]) for k in range(1,int(setup.sou/2)+1)]            
            
            op  = Operator([stencil0] + bc + bc1 +[stencil01,stencil3,stencil02,stencil2,stencil1] + receivers + [Eq(vsave,u.backward)],subs=grid.spacing_map)

        else:
            
            assert "Invalid option"

        return op
#==============================================================================

#==============================================================================
# CPML Solver 
#==============================================================================

#==============================================================================
    def solvecpml(rec,src,vp,geradamp,vector,grid, setup,system,save=False,**kwargs):   
        
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

        elif(system=='adjoint'):  
            vsave = kwargs.get('vsave')  
                
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
            
            op = Operator([stencil01,stencil02]  + bc + bc1 + [stencil1,stencil2,stencil3,stencil4] + bczeta + receivers + [Eq(vsave,u.backward)],subs=grid.spacing_map)           
  
        return op
#==============================================================================
    
#==============================================================================
class FWISolver():
#==============================================================================

#==============================================================================
    def __init__(self,set_time,setup,setting,grid,utils,v0,vp):  

        self.dt0, self.nt, self.time_range = set_time  #time discretization
        self.setting  = setting
        self.grid     = grid
        self.setup    = setup
        self.rec_true = []
        self.utils    = utils
        self.vp       = vp
        self.vp_g     = 0

        self.abc = setting["Abcs"]
#==============================================================================
        
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
              
        elif(self.abc=='habc-a1' or self.abc=='Higdon'):
            habcw    = setting["habcw"]
            self.g        = utils.gerapesos(self.setup,habcw)
            self.solv     = solverABCs.solvehabc
#==============================================================================

#==============================================================================
    def vp_guess(self, m0):
        
        grid    = self.grid
        (x, z)  = grid.dimensions
        setting = self.setting
        setup   = self.setup
        
        if(setting["Abcs"]=='pml'):
    
            vp_guess0  = Function(name="vp_guess0",grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)
            vp_guess0.data[:,:] = m0.reshape((setup.nptx,setup.nptz))
            v1loc = self.utils.gerav1m0(setup,vp_guess0.data)
            vp_guess1  = Function(name="vp_guess1",grid=grid,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
            vp_guess1.data[0:setup.nptx-1,0:setup.nptz-1]  = v1loc
            vp_guess1.data[setup.nptx-1,0:setup.nptz-1]    = vp_guess1.data[setup.nptx-2,0:setup.nptz-1]
            vp_guess1.data[0:setup.nptx,setup.nptz-1]      = vp_guess1.data[0:setup.nptx,setup.nptz-2]        
            vp_guess = [vp_guess0,vp_guess1]
        
        else:
        
            vp_guess  = Function(name="vp_guess",grid=grid,space_order=setup.sou,staggered=NODE,dtype=np.float64)
            vp_guess.data[:,:] = m0.reshape((setup.nptx,setup.nptz))

        self.vp_g = vp_guess
#==============================================================================

#==============================================================================
# True model solver function
#==============================================================================    

#==============================================================================
    def forward_true(self,sn):

        setting = self.setting
        nt      = self.nt
        setup   = self.setup
        grid    = self.grid
        solv    = self.solv
        dt0     = self.dt0
        g       = self.g
        abc     = self.abc
        (x, z)  = grid.dimensions
        nrec   = setting["rec_n"] #receivers numbers
        rec_0  = setting["pos_rec_0"]
        rec_n  = setting["pos_rec_n"]
        rec    = Receiver(name='rec',grid=grid,npoint=nrec,time_range=self.time_range,staggered=NODE,dtype=np.float64)
        #rec.coordinates.data[:, 0] = np.linspace(setup.x0pml+100,setup.x1pml-100,nrec)
        rec_0 = setting["pos_rec_0"]
        rec_n = setting["pos_rec_n"]
        rec.coordinates.data[:, 0] = np.linspace(rec_0,rec_n,nrec)
        rec.coordinates.data[:, 1] = setting["recposition_z"] 

        # Source Prameters
        src = RickerSource(name='src',grid=grid,f0=setting["f0"],npoint=1,time_range=self.time_range,staggered=NODE,dtype=np.float64)

        # The shots start at the position sd in the physical domain
        xposf = setting["position_src"] + setting["x0"] 
        src.coordinates.data[:, 0] = xposf
        src.coordinates.data[:, 1] = setting["shotposition_z"]
        
        u = TimeFunction(name="u",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64) 

        if(setting["Abcs"]=='pml' or setting["Abcs"]=='cpml'):
            
            phi1 = TimeFunction(name="phi1",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
            phi2 = TimeFunction(name="phi2",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
            
            if(setting["Abcs"]=='cpml'):
                
                zeta1 = TimeFunction(name="zeta1",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
                zeta2 = TimeFunction(name="zeta2",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
                vector  = [u,phi1, phi2,zeta1,zeta2]
            
            if(setting["Abcs"]=='pml'):
                
                vector  = [u,phi1, phi2]
        
        else:
        
            vector = u   
        nsnaps = setting["snapshots"]
        factor  = mt.ceil(nt/nsnaps) + 1
        time_subsampled = ConditionalDimension('t_sub', parent=grid.time_dim, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, time_order=2, space_order=2,save=nsnaps, time_dim=time_subsampled)
      
        op_fw = solv(rec,src,self.vp,g,vector,grid,setup,system='forward', save=True, usave=usave)
        op_fw(dt=dt0)
        
        return rec, usave.data
#==============================================================================
# FWI Function
#==============================================================================    

#==============================================================================
    def apply(self,sn):
        rec     = self.rec_true[sn].data[:]
        setting = self.setting
        nt      = self.nt
        setup   = self.setup
        grid    = self.grid
        solv    = self.solv
        dt0     = self.dt0
        g       = self.g
        abc     = self.abc
        (x, z)  = grid.dimensions
        vp_guess = self.vp_g
        
        if not setting["checkpointing"]:
            # Saves Parameters    
            nsnaps = setting["snapshots"]
            factor  = mt.ceil(nt/nsnaps) + 1
            time_subsampled = ConditionalDimension('t_sub', parent=grid.time_dim, factor=factor)
            usave = TimeFunction(name='usave', grid=grid, time_order=2, space_order=2,save=nsnaps, time_dim=time_subsampled)
            vsave = TimeFunction(name='vsave', grid=grid, time_order=2, space_order=2,save=nsnaps, time_dim=time_subsampled)

        # Receivers Parameters
        nrec = setting["rec_n"] #receivers numbers
        rec_0  = setting["pos_rec_0"]
        rec_n  = setting["pos_rec_n"]
        recg  = Receiver(name='recg',grid=grid,npoint=nrec,time_range=self.time_range,staggered=NODE,dtype=np.float64)
        #recg.coordinates.data[:, 0] = np.linspace(setup.x0pml+100,setup.x1pml-100,nrec)
        rec_0 = setting["pos_rec_0"]
        rec_n = setting["pos_rec_n"]
        recg.coordinates.data[:, 0] = np.linspace(rec_0,rec_n,nrec)
        recg.coordinates.data[:, 1] = setting["recposition_z"]

        residual  = Receiver(name='residual',grid=grid,npoint=nrec,time_range=self.time_range,staggered=NODE,dtype=np.float64)
        #residual.coordinates.data[:, 0] = np.linspace(setup.x0pml+100,setup.x1pml-100,nrec)
        rec_0 = setting["pos_rec_0"]
        rec_n = setting["pos_rec_n"]
        residual.coordinates.data[:, 0] = np.linspace(rec_0,rec_n,nrec)
        residual.coordinates.data[:, 1] = setting["recposition_z"]

        # Source Prameters
        src = RickerSource(name='src',grid=grid,f0=setting["f0"],npoint=1,time_range=self.time_range,staggered=NODE,dtype=np.float64)

        # The shots start at the position 2*sd in the physical domain
        xposf = setting["position_src"] + setting["x0"] 
        src.coordinates.data[:, 0] = xposf
        src.coordinates.data[:, 1] = setting["shotposition_z"] 

        u    = TimeFunction(name="u",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64) 
        v    = TimeFunction(name="v",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64) 
   
        if(abc=='pml' or abc=='cpml'):
                
            phi1 = TimeFunction(name="phi1",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
            phi2 = TimeFunction(name="phi2",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
            
            phi1_adj = TimeFunction(name="phi1_adj",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)
            phi2_adj = TimeFunction(name="phi2_adj",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=(x,z),dtype=np.float64)

            if(abc=='cpml'):
                
                zeta1 = TimeFunction(name="zeta1",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
                zeta2 = TimeFunction(name="zeta2",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
                
                zeta1_adj = TimeFunction(name="zeta1_adj",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
                zeta2_adj = TimeFunction(name="zeta2_adj",grid=grid,time_order=setup.tou,space_order=setup.sou,staggered=NODE,dtype=np.float64)
    
                vector      = [u,phi1,phi2,zeta1,zeta2]      
                vector_adj  = [v,phi1_adj, phi2_adj,zeta1_adj,zeta2_adj]
               
            if(abc=='pml'):
                vector     = [u,phi1, phi2]
                vector_adj = [v,phi1_adj, phi2_adj]
            
        else:
           
            vector     = u
            vector_adj = v

        if(setting["checkpointing"]):
            
            cp = DevitoCheckpoint([u])
            n_checkpoints = setting["n_checkpointing"]

            # Forward solver -- Wrapper
            op_fw_guess = solv(recg,src,vp_guess,g,vector,grid,setup,system='forward') 
            
            # Adjoint-based gradient solver -- Wrapper
            op_bw = solv(residual, src, vp_guess,g,vector_adj,grid,setup,system='gradient',grad=grad,usave=u)

            wrap_fw  = CheckpointOperator(op_fw_guess,dt=dt0)

            wrap_rev = CheckpointOperator(op_bw,dt=dt0)

            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt-1)

            # Forward solver
            wrp.apply_forward()

            # Difference betwen true and guess model
            residual.data[:] = rec-recg.data[:]   # residual used as a forcing in the adjoint eq.

            # Backward solver
            wrp.apply_reverse()

        else:
           
            # Forward solver -- Wrapper
            op_fw_guess = solv(recg, src, vp_guess,g,vector,grid,setup,system='forward') 
            
            # Adjoint-based gradient solver -- Wrapper
            op_bw = solv(residual, src, vp_guess,g,vector_adj,grid,setup,system='adjoint',vsave=vsave)

            op_fw_guess(dt=dt0)

            # Difference betwen true and guess model
            residual.data[:]=rec-recg.data[:] # residual used as a forcing in the adjoint eq.
            
            op_bw(dt=dt0)

        # J = 0.5*np.linalg.norm(residual.data.flatten())**2
        # grad.data[0:setup.npmlx,:] = 0.
        # grad.data[-setup.npmlx:setup.nptx,:] = 0.
        # grad.data[:,-setup.npmlz:setup.nptz] = 0.

        return vsave, recg.data
#==============================================================================    