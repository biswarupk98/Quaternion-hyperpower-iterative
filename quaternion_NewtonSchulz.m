function [X_final, num_iterations, residuals] = quaternion_NewtonSchulz(A, tol, maxIter, gamma,alpha)
% Damped Newton–Schulz pseudoinverse for quaternion matrices.
%   A: quaternion matrix
%   tol: tolerance
%   maxIter: maximum iterations
%   gamma: damping factor (0 < gamma <= 1)

if nargin < 4, gamma = 0.5; end
if nargin < 3, maxIter = 100; end
if nargin < 2, tol = 1e-8; end

[m,n] = size(A);
if m >= n
    I = quaternion(eye(n));
else
    I = quaternion(eye(m));
end

% alpha = 1/(norm(A,'fro')^2);
X = alpha * A';

residuals = [];
num_iterations = 0;

for k = 1:maxIter
    if m >= n
        cov_dev = X*A - I;
        update  = cov_dev*X;
    else
        cov_dev = A*X - I;
        update  = X*cov_dev;
    end
    
    Xnew = X - gamma*update;
    res  = norm(cov_dev,'fro');
    residuals(end+1) = res; %#ok<AGROW>
    
    if res < tol
        break;
    end
    
    X = Xnew;
    num_iterations = k;
end

X_final = X;
end
