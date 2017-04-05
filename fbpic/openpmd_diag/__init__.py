"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It imports the objects that allow to produce output in openPMD format.

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

from .field_diag import FieldDiagnostic
from .particle_diag import ParticleDiagnostic
from .boosted_field_diag import BoostedFieldDiagnostic
from .boosted_particle_diag import BoostedParticleDiagnostic
from .checkpoint_restart import set_periodic_checkpoint, \
     restart_from_checkpoint

__all__ = ['FieldDiagnostic', 'ParticleDiagnostic',
	'BoostedFieldDiagnostic', 'BoostedParticleDiagnostic',
    'set_periodic_checkpoint', 'restart_from_checkpoint']
