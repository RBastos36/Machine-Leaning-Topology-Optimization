from topopt_cholmod_cantilever_beam_diagonal_load import topopt
import numpy as np
import time


nelx = 180
nely = 60
rmin = 5.4
penal = 3.0
ft = 0  # ft==0 -> sens, ft==1 -> dens

num_volfrac = 7
num_load_position = 11
num_load_magnitude = 11

volfrac = np.linspace(0.6, 0.8, num=num_volfrac)  # 7 evenly spaced values between 0.2 and 0.8
load_position = np.linspace(0.6, 1, num=num_load_position)  # 11 evenly spaced values between 0 and 1
load_magnitude_vertical = np.linspace(0, 100, num=num_load_magnitude)  # 10 evenly spaced values between 1 and 100
load_magnitude_horizontal = np.linspace(0, 100, num=num_load_magnitude)  # 10 evenly spaced values between 1 and 100

start = time.time()

for i, vf in enumerate(volfrac):
    for j, lp in enumerate(load_position):
        for k, lmv in enumerate(load_magnitude_vertical):
            for l, lmh in enumerate(load_magnitude_horizontal):
                if ((lmh == 0 and lmv == 0) or (vf == 0.6 and (lmv == 0 or lmv == 10 or lmv == 20 or lmv == 30)) or
                        (vf == 0.6 and lmv == 40 and (lmh == 0 or lmh == 10 or lmh == 20 or lmh == 30 or lmh == 40))):
                    print("Skipped dataset iteration.......................................................")
                    continue

                else:
                    try:
                        load_config = {
                            'position': lp.item(),
                            'horizontal_magnitude': lmh.item(),
                            'vertical_magnitude': lmv.item()
                        }

                        # Call the topology optimization function
                        xPhys, obj = topopt(nelx, nely, vf, penal, rmin, ft, load_config)

                    except ZeroDivisionError:
                        print(ZeroDivisionError)

        # Print progress
        print(f"Processed volfrac={vf.item()}, load_position={lp.item()}, 'horizontal_magnitude': {lmh.item()},"
              f"'vertical_magnitude': {lmv.item()}, Objective={obj}")

end = time.time()

runtime = end - start

print("\n" * 3)
print("-----------------------------------------------")
print("-----------------------------------------------")
print(f"Total runtime: {runtime}")

with open("runtime.txt", "w") as file:
    file.write(f"Runtime: {runtime} seconds")

print(f"Saved runtime successfully!")
print(f"Terminating...")
print("-----------------------------------------------")
print("-----------------------------------------------")