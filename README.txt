

        LQR simulation for discrete time cart-pole problem


    Problem statement

Given the state of the cart-pole system at time k, x^k, and the control input u^k, compute
the state of the cart-pole system at time k+1, x^(k+1).

x^(k+1) = f(x^k, u^k)
x^(0) = x_ini
x^(H) = x_fin



1) Clearly define an optimal control problem for pole balancing


1.a) Define the state of the system
    x^(k) = [x, theta, x_dot, theta_dot]^T
where:
    x goes from


1.b) Define the control input
    u^(k) = [F]^T
where F is the (unbounded) force applied to the cart


1.c) Define the transition function
It's the discretized version of the continuous-time dynamics of the cart-pole system


1.d) Define the dynamics of the system
    x^(k+1) = f(x^k, u^k)

The equations of motion for the cart-pole system are:

    x'' = (F + m_p * l * (theta'^2 * sin(theta) - theta'' * cos(theta))) / M
    theta'' = (g * sin(theta) - cos(theta) * (F + m_p * l * (theta')^2) / M - mu_p * theta'/(m_p * l)) / (l * (4/3 - m_p * cos(theta)^2/M))

where:
    * x is the position of the cart
    * theta is the angle of the pole with respect to the vertical
    * l is the length of the pole
    * m_c is the mass of the cart
    * m_p is the mass of the pole
    * mu_p is the friction coefficient of the pole
    * M = m_c + m_p
    * g is the acceleration due to gravity
    * F is the force applied to the cart (control input)


1.e) Define the initial condition
    x^(0) = x_ini


1.f) Define the terminal condition
    x^(H) = x_fin

The task requires to achieve:
    theta = 0
    theta_dot = 0
    x_dot = 0


1.g) Define the objective function
The optimal control problem is to find the control input u^k that minimizes the cost function
    J = sum_{k=0}^{H-1} (x^k)^T Q x^k + (u^k)^T R u^k + (x^H)^T P x^H

where:
    * Q is the state cost matrix (4x4)
    * R is the control effort cost matrix (1x1)
    * P is the terminal state cost matrix (4x4)
    * H is the time horizon
    * x^H is the final state of the system
    * x^k is the state of the system at time k
    * u^k is the control input at time k


1.h) Define the constraints
H is the finite-time horizon



2) Is the LQR applicable in its simple form to the system?

The LQR is applicable to the cart-pole system because it's a linear system. The equations of motion
for the cart-pole system are nonlinear, but they can be linearized around the equilibrium point
(theta = 0, x = 0, theta_dot = 0, x_dot = 0) to obtain a linear system. The linearized system can be
written as

    x^(k+1) = A x^k + B u^k

where:
    A is the state transition matrix
    B is the control input matrix

Reachability: the system is reachable if the controllability matrix is full rank



3) Simulation: Given a sequence of control inputs [u^(1), ...,  u^(H-1)], simulate the
behavior of the discretized cart and pole system.

delta_t = 0.1 s

    v' = (F + m_p * l * (omega^2 * sin(theta) - omega' * cos(theta))) / M
    omega' = (g * sin(theta) - cos(theta) * (F + m_p * l * omega^2) / M - mu_p * omega/(m_p * l)) / (l * (4/3 - m_p * cos(theta)^2/M))

where:
    v = speed of the cart x'
    omega = angular velocity of the pole theta'
