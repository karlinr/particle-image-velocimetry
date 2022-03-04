import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rc('text', usetex=True)
plt.rcParams.update({"pgf.texsystem": "xelatex", "font.family": "serif", "text.usetex": True, 'pgf.rcfonts': False})


binsize = 60
files = os.listdir("../../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(np.min(phases), np.max(phases), binsize)
indices = np.digitize(phases, bins)
b = 45

filestopiv = np.array(files)[indices == b]
piv = PIV("", 60, 16, 16, 0.52, "5pointgaussian", True)
piv.add_video(["../../data/zebrafish/phase/" + str(f) for f in filestopiv])
piv.get_spaced_coordinates()
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
piv.plot_flow_field()
piv.set(24, 16, 16)
piv.do_pass()
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
piv.plot_flow_field()
print(b + np.pi / binsize)

piv.begin_draw()
plt.figure(figsize = (3.2, 2.6))
piv.draw_intensity()
piv.draw_flow_field()
plt.tight_layout()
plt.savefig('piv_zebrafish_flowfield.pgf', transparent = True)
piv.end_draw()
