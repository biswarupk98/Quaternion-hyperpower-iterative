function Xnext = ns_hyperpower_left(A, X, p)
%NS_HYPERPOWER_LEFT  General left-hyperpower / NS step
%   Xnext = ns_hyperpower_left(A, X, p)
%   Computes G = I - A*X, S = I + G + ... + G^(p-1), then Xnext = X * S
%
%   A : (m x n) quaternion matrix (m >= n)
%   X : (n x m) quaternion matrix (current approximation)
%   p : order (positive integer >= 1). For your requests use p=10 or p=19.
%
% Note: uses MATLAB quaternion arrays (A, X should be quaternion objects).

if nargin < 3
    error('ns_hyperpower_left requires A, X, and p');
end
if p < 1 || floor(p) ~= p
    error('p must be a positive integer');
end

m = size(A,1);   % G is m x m
% quaternion identity of size m
I = quaternion(eye(m), zeros(m), zeros(m), zeros(m));

% left residual
G = I - A * X;

% accumulate S = I + G + G^2 + ... + G^(p-1)
S = I;
if p > 1
    Gpow = G;
    for k = 1:(p-1)
        S = S + Gpow;
        Gpow = Gpow * G;
    end
end

% update on the right
Xnext = X * S;
end

% -------------------------
% Backwards-compatible wrappers
% -------------------------
function Xnext = ns_hyperpower_left_19_simple(A, X)
% Wrapper for legacy name expecting p=19
Xnext = ns_hyperpower_left(A, X, 19);
end

function Xnext = ns_hyperpower_left_10_simple(A, X)
% Wrapper for legacy name expecting p=10
Xnext = ns_hyperpower_left(A, X, 10);
end
