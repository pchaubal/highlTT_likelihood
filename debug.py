import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_fgmodels( tsz, ksz, cib_p, cib_c, radio, tsz_cib):
    # make a plot with 3 columns and 2 rows
    fig = plt.figure(figsize=(10,8))
    plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlepad'] = -14 
    gs = fig.add_gridspec(2, 3, hspace=0, wspace=0)
    (ax1, ax2, ax3), (ax4, ax5, ax6) = gs.subplots(sharex='col', sharey='row')
    # 90x90
    ax1.set_title('90x90')
    ax1.plot(tsz[0], color='blue', label='tsz')
    ax1.plot(ksz, color='blue', linestyle='dashed', label='ksz')
    ax1.plot(cib_p[0], color='orange', label='cib')
    ax1.plot(cib_c[0], color='orange', linestyle='dashed', label='cib clustering')
    ax1.plot(radio[0], color='lime', label='radio')
    ax1.plot(tsz_cib[0], color='violet', label='tsz_cib')
    #
    # # 150x150
    ax2.set_title('150x150')
    ax2.plot(tsz[3], color='blue', label='tsz')
    ax2.plot(ksz, color='blue', linestyle='dashed', label='ksz')
    ax2.plot(cib_p[3], color='orange', label='cib')
    ax2.plot(cib_c[3], color='orange', linestyle='dashed', label='cib clustering')
    ax2.plot(radio[3], color='lime', label='radio')
    ax2.plot(tsz_cib[3], color='violet', label='tsz_cib')
    #
    # # 220x220
    ax3.set_title('220x220')
    ax3.plot(tsz[5], color='blue', label='tsz')
    ax3.plot(ksz, color='blue', linestyle='dashed', label='ksz')
    ax3.plot(cib_p[5], color='orange', label='cib')
    ax3.plot(cib_c[5], color='orange', linestyle='dashed', label='cib clustering')
    ax3.plot(radio[5], color='lime', label='radio')
    ax3.plot(tsz_cib[5], color='violet', label='tsz_cib')
    #
    # #90x150
    ax4.set_title('90x150')
    ax4.plot(tsz[1], color='blue', label='tsz')
    ax4.plot(ksz, color='blue', linestyle='dashed', label='ksz')
    ax4.plot(cib_p[1], color='orange', label='cib')
    ax4.plot(cib_c[1], color='orange', linestyle='dashed', label='cib clustering')
    ax4.plot(radio[1], color='lime', label='radio')
    ax4.plot(tsz_cib[1], color='violet', label='tsz_cib')
    #
    # #90x220
    ax5.set_title('90x220')
    ax5.plot(tsz[2], color='blue', label='tsz')
    ax5.plot(ksz, color='blue', linestyle='dashed', label='ksz')
    ax5.plot(cib_p[2], color='orange', label='cib')
    ax5.plot(cib_c[2], color='orange', linestyle='dashed', label='cib clustering')
    ax5.plot(radio[2], color='lime', label='radio')
    ax5.plot(tsz_cib[2], color='violet', label='tsz_cib')
    #
    # #150x220
    ax6.set_title('150x220')
    ax6.plot(tsz[4], color='blue', label='tsz')
    ax6.plot(ksz, color='blue', linestyle='dashed', label='ksz')
    ax6.plot(cib_p[4], color='orange', label='cib')
    ax6.plot(cib_c[4], color='orange', linestyle='dashed', label='cib clustering')
    ax6.plot(radio[4], color='lime', label='radio')
    ax6.plot(tsz_cib[4], color='violet', label='tsz_cib')
    #
    for ax in fig.get_axes():
        ax.label_outer()
        ax.set_xlim(2000, 11000)
        ax.set_ylim(0.4, 1200)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())

    plt.subplots_adjust(bottom=0.2)
    plt.legend(loc='lower center', bbox_to_anchor=(-0.5, -0.55),
          ncol=3, fancybox=True, shadow=True)
    plt.savefig('foregrounds.png')
