function [X_final, num_iterations] = quaternion_Hyper19(A, X0, tolerance)
%QUATERNION_HYPER19  Structure-preserving 19th-order hyperpower method 
% for quaternion matrix pseudoinverse.
%
%   [X_final, num_iterations] = quaternion_Hyper19(A, X0, tolerance)
%
%   A          : quaternion matrix
%   X0         : initial guess (e.g. α*A', with α = 1/‖A‖_F²)
%   tolerance  : stopping criterion
%
%   X_final    : approximate pseudoinverse
%   num_iterations : number of iterations performed

% Quaternion identity of size n×n
n = size(A,1);
I = quaternion(eye(n), zeros(n), zeros(n), zeros(n));

% Constants
t1    = 1/4 * (sqrt(27 - 2*sqrt(93)) + 1);
t2    = 1/4 * (1 - sqrt(27 - 2*sqrt(93)));
t3    = 1/496 * (5*sqrt(93) - 93);
s1    = 1/496 * (-5*sqrt(93) - 93);
s2    = -1/4 * sqrt(93);
zeta1 = 3/8;
zeta2 = 321/1984;

% Initial guess
Xj = X0;
num_iterations = 0;

% Iteration loop
while true
    Rj   = I - A*Xj;
    Rj2  = Rj*Rj;
    Rj4  = Rj2*Rj2;

    Uj   = (I + t1*Rj2 + Rj4) * (I + t2*Rj2 + Rj4);
    Vj   = Uj + t3*Rj2;
    Wj   = Uj + s1*Rj2 + s2*Rj4;

    Xnext = Xj * (I + (Rj + Rj2) * ((Vj*Wj) + zeta1*Rj2 + zeta2*Rj4));

    num_iterations = num_iterations + 1;

    if norm(Xnext - Xj,'fro') < tolerance
        Xj = Xnext;
        break;
    end

    Xj = Xnext;
end

X_final = Xj;
end
