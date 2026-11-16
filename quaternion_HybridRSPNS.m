function [X_final, info] = quaternion_HybridRSPNS(A, r, p, T, tol, maxIter)
% Hybrid RSP + NS: alternate T RSP steps with one hyperpower step
%   A: quaternion matrix
%   r: sketch size
%   p: hyperpower order
%   T: RSP steps per cycle

[m,n] = size(A);
if nargin < 6, maxIter = 500; end
if nargin < 5, tol = 1e-8; end

X = (1/(norm(A,'fro')^2)) * A';
residuals = [];

for it=1:maxIter
    for t=1:T
        Omega = quaternion(randn(n,r),randn(n,r),randn(n,r),randn(n,r));
        Y = A*Omega;
        R = Omega - X*Y;
        Ypinv = pinv(Y);
        X = X + R*Ypinv;
    end
    % hyperpower step
    I = quaternion(eye(n));
    F = I - X*A;
    S = I;
    Fpow = F;
    for j=2:p
        S = S + Fpow;
        Fpow = Fpow*F;
    end
    X = S*X;
    res = norm(A*X*A - A,'fro');
    residuals(end+1) = res; %#ok<AGROW>
    if res < tol, break; end
end
X_final = X;
info.residuals = residuals;
info.iters = it;
end
