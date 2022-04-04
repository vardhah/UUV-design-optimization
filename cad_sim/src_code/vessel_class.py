import os
import sys

# https://wiki.freecadweb.org/Embedding_FreeCAD
freecad_libs = [
    '/usr/local/lib/FreeCAD.so',
    '/usr/lib/freecad-python3/lib/FreeCAD.so',
]
for lib in freecad_libs:
    if os.path.exists(lib):
        path = os.path.dirname(lib)
        if path not in sys.path:
            sys.path.append(path)
        break
else:
    raise ValueError("FreeCAD library was not found!")

import FreeCAD                              # noqa
from FreeCAD import Units                   # noqa

femtools_libs = [
    '/usr/local/Mod/Fem/femtools',
    '/usr/share/freecad/Mod/Fem/femtools',
]
for lib in femtools_libs:
    if os.path.exists(lib):
        path = os.path.dirname(lib)
        if path not in sys.path:
            sys.path.append(path)
        path = os.path.abspath(os.path.join(lib, '..', '..'))
        if path not in sys.path:
            sys.path.append(path)
        path = os.path.abspath(os.path.join(lib, '..', '..', '..', 'Ext'))
        if path not in sys.path:
            sys.path.append(path)
        break
else:
    raise ValueError("femtools library was not found!")

from femtools.ccxtools import FemToolsCcx   # noqa
from femmesh.gmshtools import GmshTools     # noqa




import Mesh

class GliderVessel(object):
    """
    The base class to work with parametric pressure vessel models.
    """

    def __init__(self, filename: str, debug=True):
        """
        Creates a pressure vessel analysis class that can be used to run
        multiple simulations for the given design template by changing its
        parameters.
        """
        self.filename = filename
        self.debug = debug

        print("Opening:", filename)
        self.doc = FreeCAD.open(filename)
        self.exp_index= None 
       
       
        self.sketch_params = []
        obj = self.doc.getObject('Sketch')
        obj1 = self.doc.getObject('Sketch001') 
        
     
       
        print('***Parametric properties are:***')
        print('Sketch is:')
        for c in obj.Constraints:
            if c.Name:
                self.sketch_params.append(str(c.Name))
                print(str(c.Name))
        print('Sketch001 is:')
        for c in obj1.Constraints:
            if c.Name:
                self.sketch_params.append(str(c.Name))
                print(str(c.Name))
                
        
    def set_prop(self, f_dia,tail_l,fair_body):
       f_dia=  str(f_dia)+' m'
       t_len= str(tail_l)+' m'
       f_len= str(fair_body)+' m'
       print('f dia:',f_dia,'t_len:',t_len,'f_len:',f_len)
       try:
        sheet = self.doc.getObjectsByLabel('Parameters')[0]
        print('sheet is:',sheet)
        sheet.set("fairing_dia", f_dia)
        sheet.set("tail_len", t_len)
        sheet.set("fairing_len", f_len)
        
        sheet.recompute()
        self.doc.recompute()
        
       except: 
        print('failed')
    
    def set_fins(self, r_height,r_len,r_thickness):
       rud_h= str(r_height)+' m'
       rud_l=  str(r_len)+' m'
       rud_t= str(r_thickness)+' mm'
       print('rudd_h:',rud_h,'rudd_l:',rud_l,'rudd_t:',rud_t)
       try:
        sheet = self.doc.getObjectsByLabel('Parameters')[0]
        sheet.set("rudder_len", rud_l)
        sheet.set("rudder_height", rud_h)
        sheet.set("rudder_thickness", rud_t)
        
        sheet.recompute()
        self.doc.recompute()
        
       except: 
        print('failed')
    
    def get_rudder_details(self) -> float:
        """
        Returns the wing thickness for current design.
        """
        sheet = self.doc.getObjectsByLabel('Parameters')[0]
        r_l=sheet.get("rudder_len")
        r_h=sheet.get("rudder_height")
        r_t=sheet.get("rudder_thickness")
        print('Rudder height is:',r_h,'Rudder lenght is:',r_l,'Rudder thickness is:',r_t)
        
        
        
    def print_info(self):
        
        self.recompute()
        """
        Prints out all relevant information from the design template
        and the output of design analysis.
        """
        names = [obj.Name for obj in self.doc.Objects]
        print("Object names:", ", ".join(names))

        self.obj1 = self.doc.getObject('Sketch001')


        print("------Body properties:-------")
        print("  body_area = {:.6f} m^2".format(self.get_body_area()))
        print("  body_volume = {:.9f} m^3".format(self.get_body_volume()))
        print("  Fairing_len = ".format(self.get_fairing_len()))
        print("  Fairing_dia = ".format(self.get_fairing_dia()))
        print("  Tail_len = ".format(self.get_tail_len()))
        
      
    
    def recompute(self):
        """
        Recompute the design after setting all parametric values of design
        """
        self.clean()
        self.doc.recompute()
        
    
    def create_stl(self,exp_index):   
        """
        Generate stl file from the current design
        """
        try:
         __objs__=self.doc.getObjectsByLabel("CFDModel")
         stl_name= "./stl_repo/rudder_study_uuv"+str(exp_index)+".stl"
         Mesh.export(__objs__,stl_name)
         del __objs__    
        except:
         print("An error occurred while creating stl file") 
    
    
    def get_exp_index(self) -> int:
        """
        Returns the experiment index of current design.
        """
        print('self index is:',self.exp_index)
        return self.exp_index
   
    def set_exp_index(self,exp_ind) -> int:
        """
        set the experiment index of current design
        """
        self.exp_index=exp_ind

    

   
    def set_fairing_dia(self,dia):
        """
        set the wing location of current experiment
        """
        #print('In wing location setting')
        obj1 = self.doc.getObject('Sketch001') 
        return obj1.setDatum('fairing_dia', Units.Quantity(dia * 1e3, Units.Unit('mm')))
    
    
    def get_fairing_dia(self) -> float:
        """
        Returns the wing thickness for current design.
        """
        obj_spz = self.doc.getObject('Sketch001')
        return obj_spz.getDatum('fairing_dia').getValueAs('mm') * 1e-3
   

    def set_tail_len(self,t_len) -> float:
        """
        set the wing's thickness of current design
        """
        obj1 = self.doc.getObject('Sketch001') 
        return obj1.setDatum('tail_len', Units.Quantity(t_len * 1e3, Units.Unit('mm')))
    
    
    def get_tail_len(self) -> float:
        """
        Returns the wing thickness for current design.
        """
        obj_spz = self.doc.getObject('Sketch001')
        return obj_spz.getDatum('tail_len').getValueAs('mm') * 1e-3
   
   
    def set_fairing_len(self,f_len) -> float:
        """
        set the wing's chord of current design
        """
        obj1 = self.doc.getObject('Sketch001') 
        return obj1.setDatum('fairing_len', Units.Quantity(f_len * 1e3, Units.Unit('mm')))
     
    
    def get_fairing_len(self) -> float:
        """
        Returns the wing thickness for current design.
        """
        obj_spz = self.doc.getObject('Sketch001')
        return obj_spz.getDatum('fairing_len').getValueAs('mm') * 1e-3
   


    def get_body_area(self):
        """
        Returns the body volume in square meters.
        """
        obj = self.doc.getObject('Body')
        return obj.Shape.Area * 1e-6

    def get_body_volume(self):
        """
        Returns the body volume in cubic meters.
        """
        obj = self.doc.getObject('Body')
        return obj.Shape.Volume * 1e-9

   
    def get_outer_area(self):
        obj = self.doc.getObject('Body')
        return obj.Shape.OuterShell.Area * 1e-6

    def get_outer_volume(self):
        obj = self.doc.getObject('Body')
        return obj.Shape.OuterShell.Volume * 1e-9

    def get_inner_area(self):
        obj = self.doc.getObject('Body')
        return self.get_body_area() - self.get_outer_area()

    def get_inner_volume(self):
        obj = self.doc.getObject('Body')
        return self.get_outer_volume() - self.get_body_volume()

    def clean(self):
        """
        Removes all temporary artifacts from the model.
        """
        if self.doc.getObject('CCX_Results'):
            self.doc.removeObject('CCX_Results')
        if self.doc.getObject('ResultMesh'):
            self.doc.removeObject('ResultMesh')
        if self.doc.getObject('ccx_dat_file'):
            self.doc.removeObject('ccx_dat_file')


   
    






if __name__ == '__main__':
    run()
