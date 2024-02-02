Betatron radiation
===================

This plugin allows to calculate on-fly the synchrotron radiation assuming a 
simplified description valid for the relativistic particles and the strong-wiggler regime. 
These assumptions can be formulated as the conditions on the Lorentz factor of the emitting 
particle :math:`\gamma_p\gg 1`, and the angular excursion of it's oscillations is 
:math:`\sigma_{\theta}\gg 1/\gamma_p`. This is a typical case for the laser-plasma betatron 
or non-linear Compton source.

For the assumptions above, radiation emitted by an electron during time :math:`\Delta t` can be described by the dimensionless spectral density distribution as:

.. math::
	\frac{\partial E_{rad} }{ \partial \hbar \omega_{ph}} = \frac{P_{rad}\; \Delta t  }{ \hbar\omega_c }  \; S\left(\frac{\omega}{\omega_c}\right) \,,

where the instantaneous power follows from the relativistic Larmor formula: 

.. math::

	P_{rad} = \frac{e^2 \gamma^6}{6 \pi\epsilon_0 c} \; [|\dot{\boldsymbol \beta}|^2 - (\boldsymbol \beta\times\dot{\boldsymbol \beta})^2] \,.

The spectral profile is approximated with the standard function of :math:`\xi=\omega/\omega_c`:

.. math::

	S(x) = \frac{9\sqrt{3}}{8\pi} \; x\; \int_x^{\infty} K_{5/3}(\xi)\; d\xi'\,,

where critical frequency is calculated from with `curvature radius along trajectory <https://en.wikipedia.org/wiki/Radius_of_curvature#Derivation>`_: 

.. math::

	\omega_c = \frac{3}{2}\;\gamma^3\; \cfrac{c }{ \rho} = \frac{3}{2}\;\gamma^3 \frac{\sqrt{ \left|\dot{\boldsymbol \beta}\right|^2 - \left(\boldsymbol \beta \cdot  \dot{\boldsymbol\beta} \right)^2 / |\boldsymbol \beta|^2} } {|\boldsymbol \beta|^2} \,,

and motion quantities are calculated from the motion equations with fields:

.. math::

	\boldsymbol\beta = \mathbf{u} /  \gamma  \,,\;\;  \partial_t \mathbf{u} = - \frac{e}{m c} ( \boldsymbol E + \boldsymbol\beta \times c \mathbf{B} )\,, 

	\partial_t \gamma = - \frac{e}{m c} \; \boldsymbol\beta \cdot \mathbf{E} \,,\;\;   \partial_t \boldsymbol\beta = \cfrac{1}{\gamma} \;(\partial_t \mathbf{u} - \boldsymbol\beta \; \partial_t \gamma)\,.

Radiation is emitted in the direction along electron propagation 
:math:`\boldsymbol u/|\boldsymbol u|` into the angular profile approximated as 
:math:`\propto (1-\beta\cos\theta)^{-5}`. The field is calculated in the discrete 
3D spectral-angular domain :math:`(\theta_x, \theta_y, \hbar\omega)`, and the particles are deposited 
using fast bilinear interpolations, so to account for the changing angular shapes and emission 
directions we add to emission direction a random angular spread 
:math:`\delta\theta \sim \mathcal  N(2^{-3/2} \gamma^{-1})`, and with time averages to 
the proper profile.

Optionally the plugin can deduce the photon momentum from the emitting electron in order to 
simulate the classical radiation reaction.

Plugin is activated for the chosen particle specie (electron) with the following method:

.. automethod:: fbpic.particles.Particles.activate_synchrotron

The calculated radiation data in the dimensionless units 
(:math:`\partial^2 E_{rad} / \partial \hbar \omega_{ph} \partial \theta`) can be added as a 
standard openPMD diagnostics:

.. autoclass:: fbpic.openpmd_diag.SRDiagnostic