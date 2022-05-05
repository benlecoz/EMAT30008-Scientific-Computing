import matplotlib.pyplot as plt
import numpy as np
from ODE_solver import solve_ode, input_test
from scipy.optimize import fsolve


def phase_condition(ODE, u0, *args):

    x0, t = u0[:-1], u0[-1]
    phase_con = ODE(x0, t, *args)[0]

    return x0, t, phase_con


def shooting(ODE):

    def conds(u0, pc, *args):
        x0, t, phase_con = pc(ODE, u0, *args)

        sol, sol_time = solve_ode(ODE, x0, 0, t, 'RK4', 0.01, True, *args)

        period_con = []

        for i in range(len(x0)):
            period_con.append(x0[i] - sol[-1, i])

        full_conds = np.r_[np.array(period_con), phase_con]

        return full_conds

    return conds


def shooting_orbit(ODE, u0, pc, system, *args):

    shooting_solution = fsolve(shooting(ODE), u0, (pc, *args), full_output=True)

    convergence = shooting_solution[3]

    if convergence == 'The solution converged.':
        print(convergence + f' The ODE ran was {ODE.__name__}.\n')
        x0, t = shooting_solution[0][:-1], shooting_solution[0][-1]
    else:
        raise ValueError("The shooting algorithm could not converge to a solution, please try again with different values.")

    sol, sol_time = solve_ode(ODE, x0, 0, t, 'RK4', 0.01, True, *args)

    def plot_sol(ax):

        for i in range(sol.shape[1]):
            ax.plot(sol_time, sol[:, i], label = 'S' + str(i))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()

    if system:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(sol[:, 0], sol[:, 1], label = 'Periodic Orbit')
        ax1.set_xlabel('S0')
        ax1.set_ylabel('S1')
        plt.legend()
        plot_sol(ax2)
    else:
        plot_sol(plt)

    plt.show()


def main():

    def predator_prey(X, t, args):
        """
        Function for predator-prey equation
            Parameters:
                X:      initial conditions
                t:      t value
                args:   any additional arguments that ODE expects

            Returns:
                Array of dxdt, dydt
        """
        x, y = X
        a, b, d = args[0], args[1], args[2]

        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - y / x)

        dXdt = np.array([dxdt, dydt])

        return dXdt

    """
    We simulate the predator-prey equations for a = 1, d = 0.1, and choosing two b values on either side of 0.26. 
    The random values chosen for b1 and b2 are slightly distanced from 0.26, to ensure that the behaviours of the 
    equations is clear to see. 
    """

    b1 = np.round(np.random.uniform(0.1, 0.22), 2)
    b2 = np.round(np.random.uniform(0.3, 0.5), 2)

    pred_prey_sol1, pred_prey_time1 = solve_ode(predator_prey, [0.2, 0.2], 0, 120, 'RK4', 0.01, True, [1, b1, 0.1])
    pred_prey_sol2, pred_prey_time2 = solve_ode(predator_prey, [0.2, 0.2], 0, 120, 'RK4', 0.01, True, [1, b2, 0.1])

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(pred_prey_time1, pred_prey_sol1, label = 'Predator Prey equation with b = ' + str(b1))
    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2.plot(pred_prey_time2, pred_prey_sol2, label = 'Predator Prey equation with b = ' + str(b2))
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.show()

    """
    Plotting this long term, we see that when b < 0.26, the predator prey solutions start 
    oscillating periodically, whereas when b > 0.26, the solutions converge.
    """

    """
    We now want to isolate a periodic orbit, in order to test our shooting code later on. We have seen above that 
    the predator prey solutions only oscillate periodically when b < 0.26, so we choose a low b value to ensure that 
    we see periodicity.  
    """

    pred_prey_sol3, pred_prey_time3 = solve_ode(predator_prey, [1.2, 1.2], 0, 100, 'RK4', 0.01, True, [1, 0.1, 0.1])

    """
    When plotted, we see that the predator prey equations seem to be periodic once the value of t starts getting larger.
    Moreover, the period of these equations seems to be approximately t = 34s. To visualise this, we plot the predator
    prey equations up to t = 100s to ensure we see periodicity, and then next to it plot the isolated orbit with period 
    t = 34s.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(pred_prey_time3, pred_prey_sol3, label = 'Predator Prey equation with b = 0.1')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')

    ax2.plot(pred_prey_time3[5000:8400], pred_prey_sol3[5000:8400, :], label = 'Predator Prey equation with b = 0.1')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')

    plt.show()

    """
    As mentioned above, it is visually confirmed that the period of the predator prey equations are in fact t = 34s, but 
    we will now develop and run the shooting root-finding problem with the same initial conditions in order to either confirm or deny this.
    """

    """
    We run the shooting root-finding code with the same initial conditions as the plots above
    """
    pc = phase_condition
    pp_args = [1, 0.1, 0.1]
    pp_u0 = np.array([1.2, 1.2, 6])

    shooting_orbit(predator_prey, pp_u0, pc, True, pp_args)

    """
    Our code finds the solution required, as the first graph shows a complete orbit when plotting the second solution of 
    the predator prey equations against the first solution. When looking at the right hand side graph, this shows us a 
    periodicity of t = 34s, which is what we had predicted in the previous section, showing that our shooting 
    root-finding algorithm works
    """

    """
    We then run the root-finding code on the Hopf bifurcation normal form, which we find the period for, as evidenced 
    by the circular plot of the second equation against the first, and given the periodic form of both solutions against 
    time. Given the second plot, the period seems to be approximately t = 6.25s 
    """

    def Hopf_bif(U, t, args):
        beta = args[0]
        sigma = args[1]

        u1, u2 = U
        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
        dudt = np.array([du1dt, du2dt])

        return dudt
    
    hopf_args = [1, -1]
    hopf_u0 = [1.2, 1.2, 8]
    shooting_orbit(Hopf_bif, hopf_u0, pc, True, hopf_args)

    """
    This same process can be repeated for the extended Hopf bifurcation equations, where we also manage to isolate
    periodicity, as well as finding an approximate period of t = 6.25s. 
    """

    def Hopf_ext(U, t, args):
        beta = args[0]
        sigma = args[1]

        u1, u2, u3 = U
        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
        du3dt = - u3
        dudt = np.array([du1dt, du2dt, du3dt])

        return dudt

    ext_args = [1, -1]
    ext_u0 = [1, 1, 1, 8]
    shooting_orbit(Hopf_ext, ext_u0, pc, True, ext_args)


if __name__ == '__main__':
    main()

