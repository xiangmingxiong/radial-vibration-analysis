# radial-vibration-analysis
Determine complex material constants of piezoelectric disks in radial vibrations

Usage:
1. Set the mass, radius, and thickness of the piezoelectric disk in the main program radial_vibration_analysis.m.
2. Provide experimental impedance measurements around the first and second resonance frequencies in the input files impedance_measurements_1.txt and impedance_measurements_2.txt, respectively. The data of frequency f (Hz) are in Column 1. The modulus of complex impedance |Z| (Ohm) and the argument of complex impedance theta (deg) are in Columns 2 and 3.
3. Run the main program radial_vibration_analysis.m in MATLAB. The optimal material constants and the minimum average relative error between the measured and best fitting admittance and impedance will be returned. The measured and best fitting admittance and impedance will be saved in the output file fitting_results.txt and plotted in a new figure window.
