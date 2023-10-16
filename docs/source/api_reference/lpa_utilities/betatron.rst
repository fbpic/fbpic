Betatron radiation
===================

This plugin allows to calculate on-fly the synchrotron radiation assuming a simplified description valid for the relativistic  particles and the strong-wiggler regime. These assumptions can be formulated as the conditions on the Lorentz factor of the emitting particle :math:`\gamma_p\gg 1`, and the angular excursion of it's oscillations is :math:`\sigma_{\theta}\gg 1/\gamma_p`. This is a typical case for the laser-plasma betatron source.

At each time step :math:`\Delta t` each particle emits energy into a small angle along it's propagation, for which the dimensionless spectral density is calculated as:

.. math::
	\frac{\partial E_{rad} }{ \partial \hbar \omega_{ph}} = \frac{P_{rad}\; \Delta t  }{ \hbar\omega_c }  \; S\left(\frac{\omega}{\omega_c}\right)
where:

- instantaneous power follows from the relativistic Larmor formula: 

.. math::

	P_{rad} = \frac{e^2 \gamma^6}{6 \pi\epsilon_0 c} \; [|\dot{\boldsymbol \beta}|^2 - (\boldsymbol \beta\times\dot{\boldsymbol \beta})^2]

- spectral profile is given by the standard function of :math:`\xi=\omega/\omega_c`

.. math::

	S(x) = \frac{9\sqrt{3}}{8\pi} \; x\; \int_x^{\infty} K_{5/3}(\xi)\; d\xi

- critical frequency is calculated from with `curvature radius along trajectory <https://en.wikipedia.org/wiki/Radius_of_curvature#Derivation>`_: 

.. math::

	\omega_c = \frac{3}{2}\;\gamma^3\; \cfrac{c }{ \rho} = \frac{3}{2}\;\gamma^3 \frac{\sqrt{ \left|\dot{\boldsymbol \beta}\right|^2 - \left(\boldsymbol \beta \cdot  \dot{\boldsymbol\beta} \right)^2 / |\boldsymbol \beta|^2} } {|\boldsymbol \beta|^2} 

- motion quantities are calculated from the motion equations with fields:

.. math::

	\boldsymbol\beta = \mathbf{u} /  \gamma  \,,\;\;  \partial_t \mathbf{u} = - \frac{e}{m c} ( \boldsymbol E + \boldsymbol\beta \times c \mathbf{B} )\,, 

	\partial_t \gamma = - \frac{e}{m c} \; \boldsymbol\beta \cdot \mathbf{E} \,,\;\;   \partial_t \boldsymbol\beta = \cfrac{1}{\gamma} \;(\partial_t \mathbf{u} - \boldsymbol\beta \; \partial_t \gamma)

Field is calculated in 3D spectral-angular domain :math:`(\theta_x, \theta_y, \hbar\omega)`. We consider that particle emits energy along its propagation :math:`\boldsymbol u/|\boldsymbol u|` into an angular profile :math:`\propto (1-\beta\cos\theta)^{-5}`, which fits well by a Gaussian distribution with :math:`\sigma_\theta \simeq 2^{-3/2} \gamma^{-1}`. Since such shapes and emission directions constantly change it is problematic to adjust numerical deposition to the :math:`(\theta_x, \theta_y)`-plane, which also has to be atomic. To account for angular spread of emission and still use simple 1st order 2D deposition, we add corresponding random spread :math:`\delta\theta \sim \mathcal  N(2^{-3/2} \gamma^{-1})` to the emission direction, so with many time it averages to the proper profile.

Calculation proceeds as following:

- get :math:`\gamma=\sqrt{1+|\boldsymbol u|^2}`, :math:`\theta_x=\arctan\left(\frac{u_x}{u_z}\right) + \delta\theta_x`, :math:`\theta_y=\arctan\left(\frac{u_y}{u_z}\right)  + \delta\theta_y`, where :math:`\delta\theta` are random numbers from normal distribution :math:`\mathcal N(2^{-3/2} \gamma^{-1})`

- discard non-realtivistic particles :math:`\gamma<\gamma_\text{cutoff}`, and those outside of angular ROI :math:`(\theta_x, \theta_y)`

- calculate :math:`P_{rad}` and :math:`\omega_c` from motion quantities

- discard low :math:`\omega_c` that does not resolve on :math:`\omega` axis (empirically, :math:`\omega_c<4 \mathrm{d}\omega`)

- calculate :math:`\frac{\partial E_{rad} }{ \partial \hbar \omega_{ph}}` on the given :math:`\hbar\omega` axis. To evaluate :math:`S(x)` efficiently on a constantly changing grid :math:`\omega/\omega_c`, we store in memory sampling of this function on the large range  and fine grid (:math:`x\in(0, 20)`, :math:`N_x\sim 10^3` ), and use it for the linear interpolations.

- depose :math:`\cfrac{\partial E_{rad} }{ \partial \Omega \partial \hbar \omega_{ph}}` onto :math:`(\theta_x, \theta_y)`-grid, where :math:`\partial/ \partial \Omega` means :math:`1/ (\Delta\theta_x \Delta\theta_y )`

- optionally calcuate photon momentum and exctract it from the electron's