function [X_final, num_iterations] = quaternion_RAPID1(A, order, alp, tol)
%QUATERNION_RAPID1  RAPID iteration for quaternion pseudoinverse.
%
%   [X_final,num_iterations] = quaternion_RAPID1(A, order, alp, tol)
%
%   Inputs:
%     A        : quaternion matrix (m×n)
%     order    : RAPID order (nonnegative integer)
%     alp      : scaling parameter (e.g. 1 / ‖A‖_F²)
%     tol      : stopping tolerance
%
%   Output:
%     X_final        : approximate pseudoinverse of A
%     num_iterations : number of iterations performed

if nargin < 4, tol = 1e-10; end

[m,n] = size(A);

% Initial guess: quaternion Hermitian transpose
Xj = alp * A';       % use A' if your toolbox overloads it for conjugate-transpose
num_iterations = 0;

% Preallocate zero blocks for identity
Zm = zeros(m);
Iqm = quaternion(eye(m), Zm, Zm, Zm);   % quaternion identity (m×m)

while true
    % Residual product
    Pj = A * Xj;    % m×m quaternion

    % --- RAPID polynomial (nested Horner form for speed) ---
    t1 = 7 * Iqm - Pj;
    t2 = 15 * Iqm - Pj * t1;
    t3 = 13 * Iqm - Pj * t2;

    Uj = 0.25 * (Xj * t3);
    Vj = Uj + Xj * (Iqm - A * Uj);

    % Higher-order RAPID refinement
    if order <= 0
        Zn = Vj;
    else
        Y = Uj; W = Vj;
        for k = 2:(order+1)
            Z = W + Y * (Iqm - A * W);
            Y = W;
            W = Z;
        end
        Zn = W;
    end

    % Next iterate
    Xnext = Zn + Xj * (Iqm - A * Zn);
    num_iterations = num_iterations + 1;

    % Convergence check
    if norm(Xnext - Xj, 'fro') < tol
        Xj = Xnext;
        break;
    end

    % Update
    Xj = Xnext;
end

X_final = Xj;
end
