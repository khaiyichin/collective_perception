from sim_modules import MultiAgentSimData
import viz_modules as vm
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize multi-agent simulation data.")
    parser.add_argument("DATA", type=str, help="Filename to the data file.")
    args = parser.parse_args()

    # Load the data
    data = MultiAgentSimData.load(args.DATA)

    # Plot data
    fig_x_hat, ax_x_hat = plt.subplots(2)
    fig_x_hat.set_size_inches(8,6)
    
    fig_x_bar, ax_x_bar = plt.subplots(2)
    fig_x_bar.set_size_inches(8,6)

    fig_x, ax_x = plt.subplots(2)
    fig_x.set_size_inches(8,6)

    abscissa_values_x_hat = list(range(data.num_obs))
    abscissa_values_x_bar = list(range(0, data.num_obs, data.comms_period))
    abscissa_values_x = list(range(0, data.num_obs, data.comms_period))

    # Plot mean trajectories
    for dfr in data.dfr_range:
        for sp in data.sp_range:
            for n in range(data.num_exp):

                sim_obj = data.get_sim_obj(dfr, sp)

                # Plot time evolution of local estimates and confidences
                x_hat_bounds = vm.compute_std_bounds(sim_obj.x_hat_sample_mean[n], sim_obj.x_hat_sample_std[n])
                alpha_bounds = vm.compute_std_bounds(sim_obj.alpha_sample_mean[n], sim_obj.alpha_sample_std[n])

                ax_x_hat[0].plot(sim_obj.x_hat_sample_mean[n], label="Exp {}".format(n))
                ax_x_hat[0].fill_between(abscissa_values_x_hat, x_hat_bounds[0], x_hat_bounds[1], alpha=0.2)

                ax_x_hat[1].plot(sim_obj.alpha_sample_mean[n], label="Exp {}".format(n))
                ax_x_hat[1].fill_between(abscissa_values_x_hat, alpha_bounds[0], alpha_bounds[1], alpha=0.2)

                # Plot time evolution of social estimates and confidences
                x_bar_bounds = vm.compute_std_bounds(sim_obj.x_bar_sample_mean[n], sim_obj.x_bar_sample_std[n])
                rho_bounds = vm.compute_std_bounds(sim_obj.rho_sample_mean[n], sim_obj.rho_sample_std[n])

                ax_x_bar[0].plot(sim_obj.x_bar_sample_mean[n], label="Exp {}".format(n))
                ax_x_bar[0].fill_between(abscissa_values_x_bar, x_bar_bounds[0], x_bar_bounds[1], alpha=0.2)

                ax_x_bar[1].plot(sim_obj.rho_sample_mean[n], label="Exp {}".format(n))
                ax_x_bar[1].fill_between(abscissa_values_x_bar, rho_bounds[0], rho_bounds[1], alpha=0.2)

                # Plot time evolution of informed estimates and confidences
                x_bounds = vm.compute_std_bounds(sim_obj.x_sample_mean[n], sim_obj.x_sample_std[n])
                gamma_bounds = vm.compute_std_bounds(sim_obj.gamma_sample_mean[n], sim_obj.gamma_sample_std[n])

                ax_x[0].plot(sim_obj.x_sample_mean[n], label="Exp {}".format(n))
                ax_x[0].fill_between(abscissa_values_x, x_bounds[0], x_bounds[1], alpha=0.2)

                ax_x[1].plot(sim_obj.gamma_sample_mean[n], label="Exp {}".format(n))
                ax_x[1].fill_between(abscissa_values_x, gamma_bounds[0], gamma_bounds[1], alpha=0.2)

    # Set axis properties
    ax_x_hat[0].set_title("Average of {0} agents' local values with 1\u03c3 bounds".format(data.num_agents))
    ax_x_hat[0].set_xticklabels([])
    ax_x_hat[0].set_ylabel("Local estimates")
    ax_x_hat[0].set_ylim(0, 1.0)
    ax_x_hat[1].set_ylabel("Local confidences")
    ax_x_hat[1].set_xlabel("Observations")
    ax_x_hat[1].set_yscale("log")

    ax_x_bar[0].set_title("Average of {0} agents' social values with 1\u03c3 bounds".format(data.num_agents))
    ax_x_bar[0].set_ylabel("Social estimates")
    ax_x_bar[0].set_xticklabels([])
    ax_x_bar[0].set_ylim(0, 1.0)
    ax_x_bar[1].set_ylabel("Social confidences")
    ax_x_bar[1].set_xlabel("Observations")
    ax_x_bar[1].set_yscale("log")

    ax_x[0].set_title("Average of {0} agents' informed values with 1\u03c3 bounds".format(data.num_agents))
    ax_x[0].set_ylabel("Informed estimates")
    ax_x[0].set_xticklabels([])
    ax_x[0].set_ylim(0, 1.0)
    ax_x[1].set_ylabel("Informed confidences")
    ax_x[1].set_xlabel("Observations")
    ax_x[1].set_yscale("log")
    
    # Turn on grid lines
    vm.activate_subplot_grid_lines(ax_x_hat)
    vm.activate_subplot_grid_lines(ax_x_bar)
    vm.activate_subplot_grid_lines(ax_x)

    # Adjust legend location and plot sizes 
    vm.adjust_subplot_legend_and_axis(fig_x_hat, ax_x_hat)
    vm.adjust_subplot_legend_and_axis(fig_x_bar, ax_x_bar)
    vm.adjust_subplot_legend_and_axis(fig_x, ax_x)

    plt.show()