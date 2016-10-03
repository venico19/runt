
import mimpy.mesh.mesh as mesh
import mimpy.mfd.mfd as mfd
import numpy as np

import pickle

from input_example import *


#set the mesh as an instance of the HexMesh class
res_mesh = pickle.load(open("mesh"))


for cell_index in range(res_mesh.get_number_of_cells()):
    domain = res_mesh.get_cell_domain(cell_index)
    if domain == 0:
        res_mesh.set_cell_k(cell_index, K*np.eye(3)/mu)
    else:
        res_mesh.set_cell_k(cell_index, K_f*np.eye(3)/mu)

def find_flux(pressure):

    res_mfd = mfd.MFD()
    res_mfd.set_compute_diagonality(True)
    res_mfd.set_m_e_construction_method(1)
    
    #Connect the MFD instance to the new mesh. 
    res_mfd.set_mesh(res_mesh)

    res_mfd.apply_neumann_from_function(0, lambda p:np.zeros(3))
    res_mfd.apply_neumann_from_function(1, lambda p:np.zeros(3))
    res_mfd.apply_neumann_from_function(2, lambda p:np.zeros(3))

    ## Apply pressure boundary condnitions to the 9 wells. 

    ##   3  10   9
    ##   4  11   8
    ##   5   6   7

    w_to_b = {1:3, 
              2:10,
              3:9, 
              4:4,
              5:11,
              6:8, 
              7:5, 
              8:6, 
              9:7}

    for well in range(1, 10):
        if well_type[well-1] == 0:
            res_mfd.apply_neumann_from_function(w_to_b[well], lambda p:np.zeros(3))
        elif well_type[well-1] == 1:
            res_mfd.apply_dirichlet_from_function(w_to_b[well], lambda p:101325.)
        elif well_type[well-1] == 2:
            res_mfd.apply_dirichlet_from_function(w_to_b[well], lambda p:pressure)


#    res_mfd.apply_neumann_from_function(w_to_b[1], lambda p:np.zeros(3))
#    res_mfd.apply_neumann_from_function(w_to_b[3], lambda p:np.zeros(3))
#    res_mfd.apply_neumann_from_function(w_to_b[7], lambda p:np.zeros(3))
#    res_mfd.apply_neumann_from_function(w_to_b[9], lambda p:np.zeros(3))
#
#
#    res_mfd.apply_dirichlet_from_function(w_to_b[5], lambda p:pressure)
#    res_mfd.apply_dirichlet_from_function(w_to_b[2], lambda p:101325.)
#    res_mfd.apply_dirichlet_from_function(w_to_b[4], lambda p:101325.)
#    res_mfd.apply_dirichlet_from_function(w_to_b[6], lambda p:101325.)
#    res_mfd.apply_dirichlet_from_function(w_to_b[8], lambda p:101325.)

    #Build the LHS and RHS. 
    res_mfd.build_lhs()
    res_mfd.build_rhs()

    #Solve the linear system. 
    res_mfd.solve()

    #Output the solution in the vtk format. It will be saved in 
    #the file "hexmes_example_1.vtk". 
    res_mesh.output_vtk_mesh("hexmesh_example_1", 
                             [res_mfd.get_pressure_solution(),], 
                             ["MFDPressure"])
    
    fluxes = np.zeros(9)
    for well in range(1, 10):
        bm = w_to_b[well]
        total_flux = 0.
        for (face_index, orientation) in res_mesh.get_boundary_faces_by_marker(bm):
            area = res_mesh.get_face_area(face_index)
            total_flux += res_mfd.get_velocity_solution_by_index(face_index)*orientation*area
            
        fluxes[well-1]+= total_flux
    return fluxes

#pressures = np.arange(105805.6, 106676.8, (106676.8-105805.6)/10.)

fluxes = np.zeros(9)
for pres in pressures:    
    print>>output_file, pres

    fluxes += find_flux(pres)

    injection = 0
    for well in range(1, 10):
        if well_type[well-1] == 2:
            injection += fluxes[well-1]

    ratio = np.zeros(9)
    for well in range(1, 10):
        ratio[well-1] = - fluxes[well-1]/injection
        print>>output_file, well, fluxes[well-1], ratio[well-1]

print>>output_file, ' '
print>>output_file, fluxes*total_time*1.e6/len(pressures)

flag_file = open('flag','w')
print>>flag_file, 'done'
    
