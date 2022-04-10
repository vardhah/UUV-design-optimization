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
        
        
     
        """
        print('***Parametric properties are:***')
        print('Sketch is:')
        for c in obj.Constraints:
            if c.Name:
                self.sketch_params.append(str(c.Name))
                print(str(c.Name))
        """
                
        
    def set_tail(self,tail_l,tail_r):
      try:
       obj = self.doc.getObject('Sketch') 
       obj.setDatum('tail_len', Units.Quantity(tail_l , Units.Unit('mm')))
       obj.setDatum('tail_rad', Units.Quantity(tail_r , Units.Unit('mm')))
       self.doc.recompute()
      except: 
       print('failed in setting tail dimensions') 
    
    def set_y_loc(self,first,second,third,fourth,fifth):
      try: 
       obj = self.doc.getObject('Sketch') 
       obj.setDatum('first_y', Units.Quantity(first , Units.Unit('mm')))
       obj.setDatum('second_y', Units.Quantity(second , Units.Unit('mm')))
       obj.setDatum('third_y', Units.Quantity(third , Units.Unit('mm')))
       obj.setDatum('fourth_y', Units.Quantity(fourth , Units.Unit('mm')))
       obj.setDatum('fifth_y', Units.Quantity(fifth , Units.Unit('mm')))
       self.doc.recompute()
      except: 
       print('failed in setting y location') 


    def set_fairing_len(self, fairing_l):
      try:
       obj = self.doc.getObject('Sketch') 
       obj.setDatum('fairing_len', Units.Quantity(fairing_l , Units.Unit('mm')))
       self.doc.recompute()
      except: 
        print('failed in setting length')
    
    def set_fairing_rad(self, fairing_r):
      try: 
       obj = self.doc.getObject('Sketch') 
       obj.setDatum('radius', Units.Quantity(fairing_r , Units.Unit('mm')))
       self.doc.recompute()
      except: 
        print('failed in setting radius')
        
        
    def print_info(self):
        
        self.recompute()
        obj_spz = self.doc.getObject('Sketch')
        fairing_len=obj_spz.getDatum('fairing_len').getValueAs('mm') 
        fairing_rad=obj_spz.getDatum('radius').getValueAs('mm') 
        tail_len=obj_spz.getDatum('tail_len').getValueAs('mm') 
        tail_rad=obj_spz.getDatum('tail_rad').getValueAs('mm') 
        
        first_y=obj_spz.getDatum('first_y').getValueAs('mm') 
        second_y=obj_spz.getDatum('second_y').getValueAs('mm') 
        third_y=obj_spz.getDatum('third_y').getValueAs('mm') 
        fourth_y=obj_spz.getDatum('fourth_y').getValueAs('mm') 
        fifth_y=obj_spz.getDatum('fifth_y').getValueAs('mm') 
        


        
        print("------Body properties:-------")
        print("  body_area = {:.6f} m^2".format(self.get_outer_volume()))
        print("  body_volume = {:.9f} m^3".format(self.get_outer_volume()))
        print("  Fairing_len = ",fairing_len,'fairing_rad = ',fairing_rad ,'.')
        print("  Tail_len = ",tail_len,'Tail_rad = ',tail_rad ,'.')
        print("  First_y = ",first_y,'second_y = ',second_y ,'; Third_y:',third_y,'"Fourth Y:',fourth_y,'Fifth_Y:',fifth_y)
        print("------------------------------")
      
    
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
         __objs__=self.doc.getObject("Body")
         #print(__objs__.Name, self.doc.Name)
         stl_name= u"./stl_repo/swordfish"+str(exp_index)+".stl"
         Mesh.export([__objs__], stl_name)
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
