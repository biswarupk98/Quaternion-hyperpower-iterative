function [X_final, num_iterations, residuals] = quaternion_HONSchulz(A, tol, maxIter)
% Third-order Newton–Schulz pseudoinverse.
% Iteration: X_{k+1} = 3T - 3TAT + T(AT)^2

if nargin < 3, maxIter = 100; end
if nargin < 2, tol = 1e-12; end

alpha = 1/(norm(A,'fro')^2 + eps);
T = alpha * A';

residuals = [];
num_iterations = 0;

for k = 1:maxIter
    AT   = A*T;
    AT2  = AT*AT;
    TAT  = (T*A)*T;
    Tnew = 3*T - 3*TAT + T*AT2;
    
    res = norm(A*T*A - A,'fro');
    residuals(end+1) = res; %#ok<AGROW>
    
    if res < tol
        break;
    end
    
    T = Tnew;
    num_iterations = k;
end

X_final = T;
end
