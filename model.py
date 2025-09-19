import numpy as np
import astropy.units as u
import time
from plotter import plot_packet_trajectories, plot_radius_vs_time, plot_mu_vs_radius

# Physical Constants
C = 2.99792458e10  # Speed of light in cm/s


class PhotonPacket:
    """Represents a single Monte Carlo photon packet."""

    def __init__(self, r, mu, nu, energy):
        self.r, self.mu, self.nu, self.energy = r, mu, nu, energy
        self.nu_cmf, self.energy_cmf = nu, energy
        self.status, self.current_shell_id = "In Flight", 0


class EjectaModel:
    """Represents the 1D spherically symmetric ejecta of a supernova."""

    def __init__(self, v_inner, v_outer, num_shells, time_since_explosion):
        time_cgs = time_since_explosion.to(u.s).value
        v_inner_cgs, v_outer_cgs = (
            v_inner.to(u.cm / u.s).value,
            v_outer.to(u.cm / u.s).value,
        )
        self.time_since_explosion, self.num_shells = time_cgs, num_shells
        self.velocity_boundaries = np.linspace(v_inner_cgs, v_outer_cgs, num_shells + 1)
        self.velocity = (
            self.velocity_boundaries[:-1] + self.velocity_boundaries[1:]
        ) / 2.0
        self.radius_boundaries = self.velocity_boundaries * time_cgs
        self.temperature = np.ones(num_shells) * 10000.0
        self.density = np.ones(num_shells) * 1e-13


def lab_to_cmf(packet, ejecta_model):
    """Transforms packet properties from lab frame to co-moving frame."""
    shell_id = packet.current_shell_id
    if shell_id >= ejecta_model.num_shells:
        shell_id = ejecta_model.num_shells - 1
    beta = ejecta_model.velocity[shell_id] / C
    doppler_factor = 1.0 - packet.mu * beta
    packet.nu_cmf = packet.nu * doppler_factor
    packet.energy_cmf = packet.energy * doppler_factor


def cmf_to_lab(packet, ejecta_model):
    """Transforms packet properties from co-moving frame to lab frame."""
    shell_id = packet.current_shell_id
    if shell_id >= ejecta_model.num_shells:
        shell_id = ejecta_model.num_shells - 1
    beta = ejecta_model.velocity[shell_id] / C
    mu_cmf = 2 * np.random.random() - 1
    doppler_factor = 1.0 / (1.0 + mu_cmf * beta)
    packet.mu = (mu_cmf + beta) * doppler_factor
    if packet.mu > 1.0:
        packet.mu = 1.0
    if packet.mu < -1.0:
        packet.mu = -1.0
    packet.nu = packet.nu_cmf * doppler_factor
    packet.energy = packet.energy_cmf * doppler_factor


def get_distance_to_boundary(packet, ejecta_model):
    """Calculates the distance to the next shell boundary."""
    r, mu = packet.r, packet.mu
    shell_id = packet.current_shell_id
    if mu > 0:
        if shell_id >= ejecta_model.num_shells - 1:
            return np.inf
        R_shell = ejecta_model.radius_boundaries[shell_id + 1]
    else:
        if shell_id == 0:
            return np.inf
        R_shell = ejecta_model.radius_boundaries[shell_id]
    sqrt_arg = (r * mu) ** 2 - r**2 + R_shell**2
    return -r * mu + np.sqrt(sqrt_arg) if sqrt_arg >= 0 else np.inf


if __name__ == "__main__":
    # --- CHOOSE YOUR TEST ---
    # TEST_MODE = "SINGLE_PACKET_DEEP_DIVE"
    # TEST_MODE = "MULTI_PACKET_OPTICALLY_THIN"
    TEST_MODE = "MULTI_PACKET_OPTICALLY_THICK"

    start_time = time.time()
    ejecta = EjectaModel(
        v_inner=11000 * u.km / u.s,
        v_outer=20000 * u.km / u.s,
        num_shells=20,
        time_since_explosion=7 * u.day,
    )

    # --- MULTI-PACKET TRAJECTORY VISUALIZATION ---
    if "MULTI_PACKET" in TEST_MODE:
        print(f"--- Running Test: {TEST_MODE} ---")
        if TEST_MODE == "MULTI_PACKET_OPTICALLY_THIN":
            opacity_per_gram = 0.2
            num_packets = 5
        elif TEST_MODE == "MULTI_PACKET_OPTICALLY_THICK":
            opacity_per_gram = 20.0
            num_packets = 5

        packets, packet_trajectories = [], [[] for _ in range(num_packets)]
        for i in range(num_packets):
            packets.append(
                PhotonPacket(
                    r=ejecta.radius_boundaries[0],
                    mu=np.sqrt(np.random.random()),
                    nu=3e15,
                    energy=1.0,
                )
            )
            packet_trajectories[i].append((packets[-1].r, packets[-1].mu))

        packets_in_flight = len(packets)
        outer_boundary_r = ejecta.radius_boundaries[-1]

        while packets_in_flight > 0:
            for i, packet in enumerate(packets):
                if packet.status == "In Flight":
                    distance_boundary = get_distance_to_boundary(packet, ejecta)
                    lab_to_cmf(packet, ejecta)
                    shell_id = packet.current_shell_id
                    if shell_id >= ejecta.num_shells:
                        shell_id = ejecta.num_shells - 1
                    opacity_cmf = opacity_per_gram * ejecta.density[shell_id]
                    tau_boundary = opacity_cmf * distance_boundary
                    tau_event = -np.log(np.random.random() + 1e-16)

                    if tau_event < tau_boundary:
                        distance = tau_event / opacity_cmf
                        cmf_to_lab(packet, ejecta)
                    else:
                        distance = distance_boundary
                        if packet.mu > 0:
                            packet.current_shell_id += 1
                        else:
                            packet.current_shell_id -= 1

                    r_old, mu_old = packet.r, packet.mu
                    packet.r = np.sqrt(
                        r_old**2 + distance**2 + 2 * r_old * distance * mu_old
                    )
                    packet.mu = (mu_old * r_old + distance) / packet.r

                    if packet.r >= outer_boundary_r:
                        packet.status = "Escaped"
                        packets_in_flight -= 1
                    packet_trajectories[i].append((packet.r, packet.mu))

        plot_packet_trajectories(packet_trajectories, ejecta)

    # --- SINGLE PACKET DEEP-DIVE ANALYSIS ---
    elif TEST_MODE == "SINGLE_PACKET_DEEP_DIVE":
        print(f"--- Running Test: {TEST_MODE} ---")
        opacity_per_gram = 20.0
        test_packet = PhotonPacket(
            r=ejecta.radius_boundaries[0],
            mu=np.sqrt(np.random.random()),
            nu=3e15,
            energy=1.0,
        )
        packet_history = [(test_packet.r, test_packet.mu)]
        outer_boundary_r = ejecta.radius_boundaries[-1]
        step_count = 0

        while test_packet.status == "In Flight":
            distance_boundary = get_distance_to_boundary(test_packet, ejecta)
            lab_to_cmf(test_packet, ejecta)
            shell_id = test_packet.current_shell_id
            if shell_id >= ejecta.num_shells:
                shell_id = ejecta.num_shells - 1
            opacity_cmf = opacity_per_gram * ejecta.density[shell_id]
            tau_boundary = opacity_cmf * distance_boundary
            tau_event = -np.log(np.random.random() + 1e-16)

            if tau_event < tau_boundary:
                distance = tau_event / opacity_cmf
                cmf_to_lab(test_packet, ejecta)
            else:
                distance = distance_boundary
                if test_packet.mu > 0:
                    test_packet.current_shell_id += 1
                else:
                    test_packet.current_shell_id -= 1

            r_old, mu_old = test_packet.r, test_packet.mu
            test_packet.r = np.sqrt(
                r_old**2 + distance**2 + 2 * r_old * distance * mu_old
            )
            test_packet.mu = (mu_old * r_old + distance) / test_packet.r

            if test_packet.r >= outer_boundary_r:
                test_packet.status = "Escaped"
            packet_history.append((test_packet.r, test_packet.mu))
            step_count += 1

        print(f"Packet escaped after {step_count} steps.")
        plot_radius_vs_time(packet_history)
        plot_mu_vs_radius(packet_history)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
