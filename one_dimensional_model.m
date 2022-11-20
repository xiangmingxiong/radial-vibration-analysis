function [Y_mod, Z_mod, dY_mod_dy, d2Y_mod_dy2, dZ_mod_dy, d2Z_mod_dy2] ...
    = one_dimensional_model(f, rho, a, d, s11E, sigma, eps33T, d31)
% Calculate the admittance (Y_mod) and impedance (Z_mod)
% in the one-dimensional vibration model and their
% derivatives and Hessian matrices (with respect to y).

N = size(f, 1); % Number of frequency data points

z = 2 * pi * a * sqrt(rho * s11E * (1 - sigma^2)) * f;
j1 = z .* besselj(0,z) ./ besselj(1,z);

% Admittance (model) (S)
Y_mod = 2i * pi^2 * a^2 / d * f ...
    .* (eps33T + 2 * d31^2 * (2 - j1) ./ (s11E * (1 - sigma) * (j1 + sigma - 1)));

% Impedance (model) (Ohm)
Z_mod = Y_mod .^ (-1); 

if nargout == 2
    return; % Return when only Y_mod and Z_mod are needed.
end

% Partial derivative of Y_mod with respect to s11E
dY_mod_ds11E = 2i * pi^2 * a^2 * d31^2 / (d * (1 - sigma) * s11E^2) * f ...
    .* ((1 + sigma) * z.^2 - 8 * j1 + (3 + sigma) * j1.^2 + 4 * (1 - sigma)) ...
    ./ (j1 + sigma - 1).^2;

% Partial derivative of Y_mod with respect to sigma
dY_mod_dsigma = 4i * pi^2 * a^2 * d31^2 / (d * s11E * (1 - sigma)^2) * f ...
    .* (-sigma * z.^2 + 4 * j1 - (1 + sigma) * j1.^2 - 4 * (1 - sigma)) ...
    ./ (j1 + sigma - 1).^2;

% Partial derivative of Y_mod with respect to eps33T
dY_mod_deps33T = 2i * pi^2 * a^2 / d * f;

% Partial derivative of Y_mod with respect to d31
dY_mod_dd31 = 8i * pi^2 * a^2 * d31 / (d * s11E * (1 - sigma)) * f ...
    .* (2 - j1) ./ (j1 + sigma - 1);

dY_mod_dy = zeros(N,8); % Gradient of the Y_mod with respect to y
dY_mod_dy(:,1) = dY_mod_ds11E;
dY_mod_dy(:,2) = 1i * dY_mod_ds11E;
dY_mod_dy(:,3) = dY_mod_dsigma;
dY_mod_dy(:,4) = 1i * dY_mod_dsigma;
dY_mod_dy(:,5) = dY_mod_deps33T;
dY_mod_dy(:,6) = 1i * dY_mod_deps33T;
dY_mod_dy(:,7) = dY_mod_dd31;
dY_mod_dy(:,8) = 1i * dY_mod_dd31;

% Second-order partial derivative with respect to s11E
d2Y_mod_ds11E2 = 2i * pi^2 * a^2 * d31^2 / (d * (1 - sigma) * s11E^3) * f ...
    .* (8 * (sigma - 1)^2 - (sigma^2 - 1) * z.^2 + (sigma + 1) * z.^4 ...
    + 24 * (sigma - 1) * j1 - (sigma + 1) * (sigma + 4) * z.^2 .* j1 ...
    + 24 * j1.^2 + (sigma + 1) * z.^2 .* j1.^2 - (sigma^2 + 4 * sigma + 7) * j1.^3) ...
    ./ (j1 + sigma - 1).^3;
           
% Second-order partial derivative with respect to s11E and sigma
d2Y_mod_ds11E_dsigma = 4i * pi^2 * a^2 * d31^2 / (d * (1 - sigma)^2 * s11E^2) * f ...
    .* (-4 * (sigma - 1)^2 + 2 * (sigma - 1) * z.^2 - sigma * z.^4 ...
    + 12 * (1 - sigma) * j1 + (sigma + 1)^2 * z.^2 .* j1 ...
    - (sigma^2 - 2 * sigma + 9) * j1.^2 - sigma * z.^2 .* j1.^2 ...
    + (sigma^2 + sigma + 2) * j1.^3) ...
    ./ (j1 + sigma - 1).^3;

% Second-order partial derivative with respect to s11E and eps33T
d2Y_mod_ds11E_deps33T = zeros(N,1);

% Second-order partial derivative with respect to s11E and d31
d2Y_mod_ds11E_dd31 = 4i * pi^2 * a^2 * d31 / (d * (1 - sigma) * s11E^2) * f ...
    .* ((1 + sigma) * z.^2 - 8 * j1 + (3 + sigma) * j1.^2 + 4 * (1 - sigma)) ...
    ./ (j1 + sigma - 1).^2;

% Second-order partial derivative with respect to sigma
d2Y_mod_dsigma2 = 4i * pi^2 * a^2 * d31^2 / (d * s11E * (1 + sigma) * (1 - sigma)^3) * f ...
    .* (12 * (1 - sigma) * (1 - sigma^2) + (1 - sigma) * (sigma^2 + 8 * sigma + 1) * z.^2 ...
    + 2 * sigma^2 * z.^4 - 4 * (1 - sigma) * (7 * sigma + 5) * j1 ...
    - (2 * sigma^3 + 3 * sigma^2 + 4 * sigma + 1) * z.^2 .* j1 ...
    + (sigma^3 - 9 * sigma^2 + 19 * sigma + 13) * j1.^2 + 2 * sigma^2 * z.^2 .* j1.^2 ...
    - (2 * sigma + 1) * (sigma^2 + 3) * j1.^3) ...
    ./ (j1 + sigma - 1).^3;

% Second-order partial derivative with respect to sigma and eps33T
d2Y_mod_dsigma_deps33T = zeros(N,1);

% Second-order partial derivative with respect to sigma and d31
d2Y_mod_dsigma_dd31 = 8i * pi^2 * a^2 * d31 / (d * s11E * (1 - sigma)^2) * f ...
    .* (-sigma * z.^2 + 4 * j1 - (1 + sigma) * j1.^2 - 4 * (1 - sigma)) ...
    ./ (j1 + sigma - 1).^2;

% Second-order partial derivative with respect to eps33T
d2Y_mod_deps33T2 = zeros(N,1);

% Second-order partial derivative with respect to eps33T and d31
d2Y_mod_deps33T_dd31 = zeros(N,1);

% Second-order partial derivative with respect to d31
d2Y_mod_dd312 = 8i * pi^2 * a^2 / (d * s11E * (1 - sigma)) * f ...
    .* (2 - j1) ./ (j1 + sigma - 1);

% Hessian matrix of Y_mod with respect to y
% Note that those d2Y_mod_dy2(:,j,k) with j>k are not used in the
% calculation of d2E_dy2 due to symmetry and are therefore omitted here.
d2Y_mod_dy2 = zeros(N,6,6);
d2Y_mod_dy2(:,1,1) = d2Y_mod_ds11E2;
d2Y_mod_dy2(:,1,2) = 1i * d2Y_mod_ds11E2;
d2Y_mod_dy2(:,1,3) = d2Y_mod_ds11E_dsigma;
d2Y_mod_dy2(:,1,4) = 1i * d2Y_mod_ds11E_dsigma;
d2Y_mod_dy2(:,1,5) = d2Y_mod_ds11E_deps33T;
d2Y_mod_dy2(:,1,6) = 1i * d2Y_mod_ds11E_deps33T;
d2Y_mod_dy2(:,1,7) = d2Y_mod_ds11E_dd31;
d2Y_mod_dy2(:,1,8) = 1i * d2Y_mod_ds11E_dd31;

d2Y_mod_dy2(:,2,2) = -d2Y_mod_ds11E2;
d2Y_mod_dy2(:,2,3) = 1i * d2Y_mod_ds11E_dsigma;
d2Y_mod_dy2(:,2,4) = -d2Y_mod_ds11E_dsigma;
d2Y_mod_dy2(:,2,5) = 1i * d2Y_mod_ds11E_deps33T;
d2Y_mod_dy2(:,2,6) = -d2Y_mod_ds11E_deps33T;
d2Y_mod_dy2(:,2,7) = 1i * d2Y_mod_ds11E_dd31;
d2Y_mod_dy2(:,2,8) = -d2Y_mod_ds11E_dd31;

d2Y_mod_dy2(:,3,3) = d2Y_mod_dsigma2;
d2Y_mod_dy2(:,3,4) = 1i * d2Y_mod_dsigma2;
d2Y_mod_dy2(:,3,5) = d2Y_mod_dsigma_deps33T;
d2Y_mod_dy2(:,3,6) = 1i * d2Y_mod_dsigma_deps33T;
d2Y_mod_dy2(:,3,7) = d2Y_mod_dsigma_dd31;
d2Y_mod_dy2(:,3,8) = 1i * d2Y_mod_dsigma_dd31;

d2Y_mod_dy2(:,4,4) = -d2Y_mod_dsigma2;
d2Y_mod_dy2(:,4,5) = 1i * d2Y_mod_dsigma_deps33T;
d2Y_mod_dy2(:,4,6) = -d2Y_mod_dsigma_deps33T;
d2Y_mod_dy2(:,4,7) = 1i * d2Y_mod_dsigma_dd31;
d2Y_mod_dy2(:,4,8) = -d2Y_mod_dsigma_dd31;

d2Y_mod_dy2(:,5,5) = d2Y_mod_deps33T2;
d2Y_mod_dy2(:,5,6) = 1i * d2Y_mod_deps33T2;
d2Y_mod_dy2(:,5,7) = d2Y_mod_deps33T_dd31;
d2Y_mod_dy2(:,5,8) = 1i * d2Y_mod_deps33T_dd31;

d2Y_mod_dy2(:,6,6) = -d2Y_mod_deps33T2;
d2Y_mod_dy2(:,6,7) = 1i * d2Y_mod_deps33T_dd31;
d2Y_mod_dy2(:,6,8) = -d2Y_mod_deps33T_dd31;

d2Y_mod_dy2(:,7,7) = d2Y_mod_dd312;
d2Y_mod_dy2(:,7,8) = 1i * d2Y_mod_dd312;

d2Y_mod_dy2(:,8,8) = -d2Y_mod_dd312;

dZ_mod_dy = zeros(N,8); % Gradient of Z_mod with respect to y
for j = 1 : 8
    dZ_mod_dy(:,j) = -dY_mod_dy(:,j) ./ Y_mod.^2;
end

% Hessian matrix of Z_mod with respect to y
% Note that those d2Z_mod_dy2(:,j,k) with j>k are not used in the
% calculation of d2E_dy2 due to symmetry and are therefore omitted here.
d2Z_mod_dy2 = zeros(N,8,8);
for k = 1 : 8
    for j = 1 : k
        d2Z_mod_dy2(:,j,k) = -d2Y_mod_dy2(:,j,k) ./ Y_mod.^2 ...
            + 2 * dY_mod_dy(:,j) .* dY_mod_dy(:,k) ./ Y_mod.^3;
    end
end
end

