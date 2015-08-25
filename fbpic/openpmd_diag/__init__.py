"""
This file imports the objects that allow to produce
output in OpenPMD format.

Usage
-----
In the Warp main script, initialize a
FieldDiagnostic and a ParticleDiagnostic :
    from openpmd_diag import FieldDiagnostic, ParticleDiagnostic
    diag1 = FieldDiagnostic( period=50, top=top, w3d=w3d, em=em )
    diag2 = ParticleDiagnostic( period=50, top=top, w3d=w3d,
                     species={"electrons":elec} )
    
Then pass the method diag.write to installafterstep :
    installafterstep( diag1.write )
    installafterstep( diag2.write )
"""

from field_diag import FieldDiagnostic
from particle_diag import ParticleDiagnostic
