clear all;

% Input parameters
mass = 14.8993e-3; % Mass (kg)
a = 0.5 * 35e-3; % Radius (m)
d = 2e-3; % Thickness (m)
rho = mass / (pi * a^2 * d); % Density (kg/m^3)

% Input files contain the impedance measurements in Fig. 1 of 
% X. Li, R. Sriramdas, Y. Yan, M. Sanghadasa, S. Priya.
% A new method for evaluation of the complex material coefficients
% of piezoelectric ceramics in the radial vibration modes.
% IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control,
% 2021, 68(11): 3446-3460.
inputfile_1 = 'impedance_measurements_1.csv';
inputfile_2 = 'impedance_measurements_2.csv';

% Output file contains the measured and fitted admittances and impedances.
outputfile = 'fitting_results.txt';

N_run = 100; % Number of runs with random initial material constants
N_max_iter = 100; % Number of maximum iterations

% Scaling parameters for 
% Re(s11E), Im(s11E), Re(sigma), Im(sigma), 
% Re(eps33T), Im(eps33T), Re(d31), and Im(d31)
C = [1e-11; 1e-13; 1; 0.01; 1e-8; 1e-10; 1e-10; 1e-12];

% Lower and upper bounds of random initial values of 
% Re(s11E), Im(s11E), Re(sigma), Im(sigma), 
% Re(eps33T), Im(eps33T), Re(d31), and Im(d31)
y_min = [1.0e-11; -1e-12; 0.2; -0.01; 1e-8; -1e-9; -3e-10; -1e-11];
y_max = [2.0e-11;  1e-12; 0.4;  0.01; 3e-8;  1e-9; -1e-10;  1e-11];

% Read experimental impedance data from input files.
exp_results_1 = importdata(inputfile_1);
exp_results_2 = importdata(inputfile_2);

% Number of frequency data points
N = size(exp_results_1.data, 1) + size(exp_results_2.data, 1); 

f = [exp_results_1.data(:, 1); exp_results_2.data(:, 1)]; % Frequency (Hz)

% Impedance (experiment) (Ohm)
Z_exp = [exp_results_1.data(:, 2) ...
    .* exp(1i * pi * exp_results_1.data(:, 3) / 180); ...
    exp_results_2.data(:, 2) ...
    .* exp(1i * pi * exp_results_2.data(:, 3) / 180)];

Y_exp = Z_exp .^ (-1); % Admittance (experiment) (S)

% Initialize the minimum average relative error and the corresponding
% material constants and the number of iterations of each run.
E_run_min = 1e10 * ones(N_run, 1);
s11E_run_opt = zeros(N_run, 1);
s12E_run_opt = zeros(N_run, 1);
sigma_run_opt = zeros(N_run, 1);
d31_run_opt = zeros(N_run, 1);
eps33T_run_opt = zeros(N_run, 1);
iter_run = zeros(N_run, 1);
ispossible_run = false(N_run, 1);

for r = 1 : N_run
    % Generate a set of pseudo random initial material constants
    % between the lower and upper bounds for each run.
    rng(r);
    y = y_min + rand(8,1) .* (y_max - y_min);
    
    s11E = y(1) + 1i * y(2);
    sigma = y(3) + 1i * y(4);
    eps33T = y(5) + 1i * y(6);
    d31 = y(7) + 1i * y(8);
    
    % Perform the scaling so that each component of x is around 1.
    x = y ./ C;
    
    % Calculate the admittance (Y_mod) and impedance (Z_mod) 
    % in the one-dimensional vibration model and their
    % derivatives and Hessian matrices (with respect to y).
    [Y_mod, Z_mod, dY_mod_dy, d2Y_mod_dy2, dZ_mod_dy, d2Z_mod_dy2] ...
        = one_dimensional_model(f, rho, a, d, s11E, sigma, eps33T, d31);
    
    % Calculate the average relative error (E) of the admittance and 
    % impedance in the one-dimensional vibration model and its
    % derivative and Hessian matrix (with respect to y).
    [E, dE_dy, d2E_dy2] = average_relative_error(Y_exp, Z_exp, ...
        Y_mod, Z_mod, dY_mod_dy, dZ_mod_dy, d2Y_mod_dy2, d2Z_mod_dy2);
    
    % Save the initial average relative error and material constants
    % in case the average relative error is already a local minimum.
    E_run_min(r) = E;
    s11E_run_opt(r) = s11E;
    sigma_run_opt(r) = sigma;
    d31_run_opt(r) = -sign(real(d31)) * d31;
    eps33T_run_opt(r) = eps33T;
    iter_run(r) = 0;
            
    for iter = 1 : N_max_iter
        % Derivative and Hessian matrix of E with respect to x
        dE_dx = C .* dE_dy;
        d2E_dx2 = (C * C') .* d2E_dy2;
        
        mu = max(-2*min(eig(d2E_dx2)), 0.001);
        dx = -(d2E_dx2 + mu * eye(8)) \ dE_dx;
        
        x0 = x;
        E_alpha_opt = 1e10;
        
        % Calculate the optimal alpha (alpha_opt) among 0.01, 0.1, and 1,
        % such that E_alpha = E(x0 + alpha * dx) is the minimum, 
        % which is denoted by E_alpha_opt.
        for m = -2 : 1
            alpha = 10^m;
            x = x0 + alpha * dx;
            
            s11E = C(1) * x(1) + 1i * C(2) * x(2);
            sigma = C(3) * x(3) + 1i * C(4) * x(4);
            eps33T = C(5) * x(5) + 1i * C(6) * x(6);
            d31 = C(7) * x(7) + 1i * C(8) * x(8);
            
            % Calculate the admittance (Y_mod) and impedance (Z_mod)
            % in the one-dimensional vibration model.
            [Y_mod, Z_mod] = one_dimensional_model(f, rho, a, d, ...
                s11E, sigma, eps33T, d31);

            % Calculate the average relative error (E) of the admittance
            % and impedance in the one-dimensional vibration model.
            E_alpha = average_relative_error(Y_exp, Z_exp, Y_mod, Z_mod);

            if E_alpha < E_alpha_opt
                E_alpha_opt = E_alpha;
                alpha_opt = alpha;
            end
        end
        
        if max(abs(alpha_opt * dx)) < 1e-8
            break; % Local minimum E is found.
        else
            x = x0 + alpha_opt * dx; % Update the material constants.
        end
        
        s11E = C(1) * x(1) + 1i * C(2) * x(2);
        sigma = C(3) * x(3) + 1i * C(4) * x(4);
        eps33T = C(5) * x(5) + 1i * C(6) * x(6);
        d31 = C(7) * x(7) + 1i * C(8) * x(8);
        
        % Calculate the admittance (Y_mod) and impedance (Z_mod)
        % in the one-dimensional vibration model and their
        % derivatives and Hessian matrices (with respect to y).
        [Y_mod, Z_mod, dY_mod_dy, d2Y_mod_dy2, dZ_mod_dy, d2Z_mod_dy2] ...
            = one_dimensional_model(f, rho, a, d, s11E, sigma, eps33T, d31);
        
        % Calculate the average relative error (E) of the admittance and
        % impedance in the one-dimensional vibration model and its
        % derivative and Hessian matrix (with respect to y).
        [E, dE_dy, d2E_dy2] = average_relative_error(Y_exp, Z_exp, ...
            Y_mod, Z_mod, dY_mod_dy, dZ_mod_dy, d2Y_mod_dy2, d2Z_mod_dy2);

        if E < E_run_min(r)
            E_run_min(r) = E;
            s11E_run_opt(r) = s11E;
            sigma_run_opt(r) = sigma;
            d31_run_opt(r) = -sign(real(d31)) * d31;
            eps33T_run_opt(r) = eps33T;
            iter_run(r) = iter;
        end
    end

    % For a material to be passive, the imaginary parts of the material
    % constants must satisfy the following conditions.
    % (1) imag(s11E) <= 0;
    % (2) |imag(s12E)| <= -imag(s11E);
    % (3) imag(eps33T) <= 0;
    % (4) imag(s11E) * imag(eps33T) >= imag(d31)^2.
    % Please refer to
    % R. Holland. Representation of dielectric, elastic, and piezoelectric
    % losses by complex coefficients. IEEE Transactions on Sonics and
    % Ultrasonics, 1967, SU-14: 18-20.
    s12E_run_opt(r) = -s11E_run_opt(r) * sigma_run_opt(r);
    if imag(s11E_run_opt(r)) > 0 ...
            || -imag(s11E_run_opt(r)) < abs(imag(s12E_run_opt(r))) ...
            || imag(eps33T_run_opt(r)) > 0 ...
            || imag(s11E_run_opt(r)) * imag(eps33T_run_opt(r)) ...
            < imag(d31_run_opt(r))^2
        ispossible_run(r) = false; 
    else
        ispossible_run(r) = true;
    end

    if mod(r, 10) == 1
        fprintf('\n%4s %7s %6s %9s %9s %11s %11s %11s %11s %9s %9s %s\n', ...
            'Run', 'Error', 'NIter', 'Re(s11E)', 'Im(s11E)', ...
            'Re(sigma)', 'Im(sigma)', 'Re(eps33T)', 'Im(eps33T)', ...
            'Re(d31)', 'Im(d31)', 'IsPossible');
    end

    if E_run_min(r) < 1
        fprintf('%4d %6.2f%% %6d % 9.2e % 9.2e % 11.2e % 11.2e % 11.2e % 11.2e % 9.2e % 9.2e %10s\n', ...
            r, 100*E_run_min(r), iter_run(r), ...
            real(s11E_run_opt(r)), imag(s11E_run_opt(r)), ...
            real(sigma_run_opt(r)), imag(sigma_run_opt(r)), ...
            real(eps33T_run_opt(r)), imag(eps33T_run_opt(r)), ...
            real(d31_run_opt(r)), imag(d31_run_opt(r)), ...
            string(ispossible_run(r)));
    else
        fprintf('%4d %6s%% %6d % 9.2e % 9.2e % 11.2e % 11.2e % 11.2e % 11.2e % 9.2e % 9.2e %10s\n', ...
            r, '> 100', iter_run(r), ...
            real(s11E_run_opt(r)), imag(s11E_run_opt(r)), ...
            real(sigma_run_opt(r)), imag(sigma_run_opt(r)), ...
            real(eps33T_run_opt(r)), imag(eps33T_run_opt(r)), ...
            real(d31_run_opt(r)), imag(d31_run_opt(r)), ...
            string(ispossible_run(r))); 
    end
end
 
% Calculate the global minimum of the average relative error,
% which is the minimum of E_run_min in all N_run runs.
E_global_min = 1e10;
for r = 1 : N_run
    if E_run_min(r) < E_global_min - 1e-6
        E_global_min = E_run_min(r);
        run_opt = r;
        % Globally optimal material constants
        s11E_global_opt = s11E_run_opt(r);
        sigma_global_opt = sigma_run_opt(r);
        eps33T_global_opt = eps33T_run_opt(r);
        d31_global_opt = d31_run_opt(r);
    end
end

if E_global_min > 1
    fprintf('%s\n', 'Cannot find the optimal material constants.');
    fprintf('%s\n', 'Minimum average relative error > 100%');
    return;
end

% Calculate the number of runs where the global minimum of E and 
% the globally optimal material constants are found.
N_run_opt_found = 0;
for r = 1 : N_run
    if E_run_min(r) < E_global_min + 1e-6 ...
            && abs(s11E_run_opt(r) / s11E_global_opt - 1) < 1e-4 ...
            && abs(sigma_run_opt(r) / sigma_global_opt - 1) < 1e-4 ...
            && abs(eps33T_run_opt(r) / eps33T_global_opt - 1) < 1e-4 ...
            && abs(d31_run_opt(r) / d31_global_opt - 1) < 1e-4
        N_run_opt_found = N_run_opt_found + 1;
    end
end

fprintf('\n%s%d%s%d%s\n', 'Optimal material constants found in ', ...
    N_run_opt_found, ' runs among all ', N_run, ' runs:');
fprintf('%s%.3f %+.3f i%s\n', 's11E = (', ...
    1e12 * real(s11E_global_opt), 1e12 * imag(s11E_global_opt), ...
    ') * 10^(-12) m2/N');
fprintf('%s%.4f %+.4f i%s\n', 'sigma = (', ...
    real(sigma_global_opt), imag(sigma_global_opt), ')');
fprintf('%s%.3f %+.3f i%s\n', 'eps33T = (', ...
    1e9 * real(eps33T_global_opt), 1e9 * imag(eps33T_global_opt), ...
    ') * 10^(-9) F/m');
fprintf('%s%.2f %+.2f i%s\n', 'd31 = (', ...
    1e12 * real(d31_global_opt), 1e12 * imag(d31_global_opt), ...
    ') * 10^(-12) C/N');
fprintf('\n%s%.2f%%%s\n', 'The minimum average relative error is ', ...
    100 * E_global_min, '.');

% Calculate the best fitting admittance and impedance
% with the globally optimal material constants.
[Y_mod_opt, Z_mod_opt] = one_dimensional_model(f, rho, a, d, ...
    s11E_global_opt, sigma_global_opt, eps33T_global_opt, d31_global_opt);

% Save the measured and fitted admittances and impedances.
fid = fopen(outputfile, 'wt');
fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', '"f (Hz)"', ...
    '"G_exp (S)"', '"G_mod (S)"', '"B_exp (S)"', '"B_mod (S)"', ...
    '"R_exp (Ohm)"', '"R_mod (Ohm)"', '"X_exp (Ohm)"', '"X_mod (Ohm)"');
for n = 1 : N
    fprintf(fid, '%.4e\t', f(n), real(Y_exp(n)), real(Y_mod_opt(n)), ...
        imag(Y_exp(n)), imag(Y_mod_opt(n)), real(Z_exp(n)), ...
        real(Z_mod_opt(n)), imag(Z_exp(n)), imag(Z_mod_opt(n)));
    fprintf(fid, '\n');
end
fclose(fid);

% Calculate the best fitting admittance and impedance
% with the globally optimal material constants in a detailed
% frequency range.
f_detail = (f(1) : 0.1*(f(2)-f(1)) : f(N))';
[Y_mod_opt_detail, Z_mod_opt_detail] = one_dimensional_model(f_detail, rho, a, d, ...
    s11E_global_opt, sigma_global_opt, eps33T_global_opt, d31_global_opt);

% Plot the conductances, susceptances, resistances, and reactances
% in the experiment and in the one-dimensional model.
figure;
subplot(2,2,1);
semilogy(0.001*f, real(Y_exp), 'o');
hold on;
semilogy(0.001*f_detail, real(Y_mod_opt_detail));
xlabel('Frequency f (kHz)');
ylabel('Conductance G (S)');
legend('Experiment', 'Model');

subplot(2,2,2);
plot(0.001*f, imag(Y_exp), 'o');
hold on;
plot(0.001*f_detail, imag(Y_mod_opt_detail));
xlabel('Frequency f (kHz)');
ylabel('Susceptance B (S)');
legend('Experiment', 'Model');

subplot(2,2,3);
semilogy(0.001*f, real(Z_exp), 'o');
hold on;
semilogy(0.001*f_detail, real(Z_mod_opt_detail));
xlabel('Frequency f (kHz)');
ylabel('Resistance R (Ohm)');
legend('Experiment', 'Model');

subplot(2,2,4);
plot(0.001*f, imag(Z_exp), 'o');
hold on;
plot(0.001*f_detail, imag(Z_mod_opt_detail));
xlabel('Frequency f (kHz)');
ylabel('Reactance (Ohm)');
legend('Experiment', 'Model');