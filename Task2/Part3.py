from Part3_mtlm import MTLM
from Part2_functions import plot_complex
import matplotlib.pyplot as plt
import numpy as np

# switch between 1khz static freq. (a) and 10Hz-10MHz sweep analysis (b)
khz_analysis = True


# frequency analysis @ 1 khz & 2km -------------------------------------------------------------------------------------
def freq_analysis(freq: float, wire_l: float, u_in: np.array, plots: bool):

    # create and calculate MTML
    mtml = MTLM(freq,  wire_l, 1, u_in)
    A = mtml.A()
    u_in, i_in, u_out, i_out = mtml.solve(A)

    if plots:

        p_in, p_out, rplp = mtml.power(u_in, i_in, u_out, i_out)

        # print
        print(str(' power consumption @' + str(freq) + ' Hz ').center(100, '#'))
        print(' i_in : ', i_in)
        print(' u_out : ', u_out)
        print(' input power [W]:     ', p_in.real)
        print(' load power [W]:      ', p_out.real)
        print(' -----------------------------------------------')
        print(' power loss [W]:      ', p_in.real - p_out.real, '   =>', rplp, '%')

        # plot
        plt.subplot(1, 2, 1)
        plot_complex(u_in, 'Input voltage @1kHz')
        plt.subplot(1, 2, 2)
        plot_complex(u_out, 'Output voltage @1kHz')

        # plot
        plt.figure()
        plt.subplot(1, 2, 1)
        plot_complex(i_in, 'Input current @1kHz')
        plt.subplot(1, 2, 2)
        plot_complex(i_out, 'Output current @1kHz')
        plt.show()

    return


# frequency sweep from 10 - 10MHz @ 2km --------------------------------------------------------------------------------
def freq_sweep(N: int, spaced_freq: np.array, wire_l: float, R_out: float, u_in: np.array):

    print(str(' frequency sweep - 10Hz to 10MHz ').center(100, '#'))
    # create log spaced freq vector between 10Hz and 10MHz with N values:
    freq_p_out = np.zeros(N)
    loop_count = 0

    for f_ind, f in enumerate(spaced_freq):
        loop_count += 1

        f_mtlm = MTLM(f,  wire_l, R_out, u_in)
        f_A = f_mtlm.A()
        f_u_in, f_i_in, f_u_out, f_i_out = f_mtlm.solve(f_A)
        _, f_p_out, _ = f_mtlm.power(f_u_in, f_i_in, f_u_out, f_i_out)
        freq_p_out[f_ind] = f_p_out.real

    plt.loglog(spaced_freq, freq_p_out)
    # plt.plot(spaced_freq, freq_p_out)
    # plt.xscale("log")
    plt.title('P_out over rising frequency')
    plt.xlabel('input frequency in Hz')
    plt.ylabel('(real) output power in W')
    plt.show()

    return


def main():
    # input voltage
    phase2 = (2 / 3) * np.pi
    phase3 = (4 / 3) * np.pi
    u_in = np.array([100, 80 * (np.cos(phase2) + 1j * np.sin(phase2)), 60 * (np.cos(phase3) + 1j * np.sin(phase3))])

    if khz_analysis:

        # set input
        freq = 1e3                      # [Hz] excitation
        wire_l = 2e3                    # [m] wire length

        freq_analysis(freq, wire_l, u_in, True)

    else:   # frequency sweep

        # set input
        N = 500  # number of sweep points
        spaced_freq = np.logspace(1, 6, num=N)
        wire_l = 2e3  # [m] wire length
        R_out = 1  # [R] load resistance

        freq_sweep(N, spaced_freq, wire_l, R_out, u_in)


if __name__ == '__main__':
    main()
