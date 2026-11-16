% test_quaternion_inverse_example_showAdag.m
% Build a 3x3 quaternion matrix, compute generalized inverses via SAI/RAPID/HPI19
% and print A^dag (the approximate pseudoinverse) for each method in several forms.

clear; clc;

%% -------------------------
% 1) build quaternion matrix A (3x3)
% -------------------------
% Adjust entries if you'd like an exact transcription.
A = [ quaternion(6, 3, 5, 2),  quaternion(1, 5, 2, 3),  quaternion(0, 1, 7, 8);
      quaternion(2, 1, 1, 1),  quaternion(3, 3, 1, 1),  quaternion(2, 5, 2, 1);
      quaternion(4, 2, 2, 2),  quaternion(6, 6, 2, 2),  quaternion(4,10, 4, 2) ];

fprintf('Input quaternion matrix A (each entry: s + x i + y j + z k):\n');
for i=1:3
    for j=1:3
        q = A(i,j);
        fprintf(' A(%d,%d) = %g + %g i + %g j + %g k\n', i,j, q.s, q.x, q.y, q.z);
    end
end
fprintf('\n');

%% -------------------------
% 2) compute scaling alpha via complex embedding
%% -------------------------
A0 = A.s; A1 = A.x; A2 = A.y; A3 = A.z;
A11 = complex(A0, A1);
A12 = complex(A2, A3);
CA = [A11, A12; -conj(A12), conj(A11)];
try
    smax = svds(CA,1,'largest');
catch
    svals = svd(CA,'econ');
    smax = max(svals);
end
alp = 1 / (max(smax, eps)^2);
fprintf('Chosen alpha = %.6e\n\n', alp);

tol = 1e-12;   % solver tolerance

%% utility: pretty-print quaternion matrix and component matrices
function print_quat_matrix(X, name)
    [m,n] = size(X);
    fprintf('--- %s  (quaternion entries) ---\n', name);
    for ii=1:m
        for jj=1:n
            q = X(ii,jj);
            fprintf('%8.4g + %8.4g i + %8.4g j + %8.4g k    ', q.s, q.x, q.y, q.z);
        end
        fprintf('\n');
    end
    fprintf('\n');
    % print component blocks
    fprintf(' %s component matrices (S, X, Y, Z):\n', name);
    S = X.s; Xc = X.x; Y = X.y; Z = X.z;
    fprintf('S =\n'); disp(S);
    fprintf('X =\n'); disp(Xc);
    fprintf('Y =\n'); disp(Y);
    fprintf('Z =\n'); disp(Z);
    % complex embedding of X: Xsigma = [X11, X12; -conj(X12), conj(X11)]
    X11 = complex(S, Xc);
    X12 = complex(Y, Z);
    Xsigma = [ X11,        X12;
              -conj(X12), conj(X11) ];
    fprintf('Complex embedding of %s (2m x 2n complex matrix):\n', name);
    disp(Xsigma);
    fprintf('\n');
end

%% helper: MP diagnostics
function print_mp_tests(A, X, methodName)
    AX = A * X;
    XA = X * A;
    AXA = AX * A;
    XAX = X * AX;
    MP1 = norm((AX)'-AX,'fro');
    MP2 = norm((XA)'-XA,'fro');
    MP3 = norm(AXA - A,'fro');
    MP4 = norm(XAX - X,'fro');
    fprintf('--- Diagnostics for %s ---\n', methodName);
    fprintf('|| (AX) - (AX)^H ||_F = %.4e\n', MP1);
    fprintf('|| (XA) - (XA)^H ||_F = %.4e\n', MP2);
    fprintf('|| AXA - A ||_F        = %.4e\n', MP3);
    fprintf('|| XAX - X ||_F        = %.4e\n', MP4);
    fprintf('\n');
end

%% -------------------------
% 3) Run SAI (if available)
%% -------------------------
X_sai = []; it_sai = NaN;
try
    if exist('quaternion_SAI1','file')
        [X_sai, it_sai] = quaternion_SAI1(A, alp, tol);
    elseif exist('quaternion_SAI','file')
        [X_sai, it_sai] = quaternion_SAI(A, alp, tol);
    else
        error('SAI implementation not found on MATLAB path');
    end
    fprintf('SAI finished (iterations = %d). Printing A^dag (SAI):\n', it_sai);
    print_quat_matrix(X_sai, 'A^\dagger (SAI)');
    print_mp_tests(A, X_sai, 'SAI');
catch ME
    fprintf('SAI not run: %s\n\n', ME.message);
end

%% -------------------------
% 4) Run RAPID (if available)
%% -------------------------
X_rapid = []; it_rapid = NaN;
try
    if exist('quaternion_RAPID1','file')
        order = 2;  % choose order (you can change)
        [X_rapid, it_rapid] = quaternion_RAPID1(A, order, alp, tol);
    elseif exist('quaternion_RAPID','file')
        [X_rapid, it_rapid] = quaternion_RAPID(A, 2, alp, tol);
    else
        error('RAPID implementation not found on MATLAB path');
    end
    fprintf('RAPID finished (iterations = %d). Printing A^dag (RAPID):\n', it_rapid);
    print_quat_matrix(X_rapid, 'A^\dagger (RAPID)');
    print_mp_tests(A, X_rapid, 'RAPID');
catch ME
    fprintf('RAPID not run: %s\n\n', ME.message);
end

%% -------------------------
% 5) Run HPI19 / Hyper19 (if available)
%% -------------------------
X_hpi = []; it_hpi = NaN;
try
    X0 = alp * A'; % initial guess
    if exist('quaternion_Hyper19','file')
        [X_hpi, it_hpi] = quaternion_Hyper19(A, X0, tol);
    elseif exist('Hyper19','file')
        [X_hpi, it_hpi] = Hyper19(A, X0, tol);
    else
        error('Hyper19 implementation not found on MATLAB path');
    end
    fprintf('HPI19 finished (iterations = %d). Printing A^dag (HPI19):\n', it_hpi);
    print_quat_matrix(X_hpi, 'A^\dagger (HPI19)');
    print_mp_tests(A, X_hpi, 'HPI19');
catch ME
    fprintf('HPI19 not run: %s\n\n', ME.message);
end

%% -------------------------
% 6) Save results to .mat for later inspection
%% -------------------------
outfn = 'test_quat_adag_results.mat';
save(outfn, 'A', 'X_sai', 'it_sai', 'X_rapid', 'it_rapid', 'X_hpi', 'it_hpi');
fprintf('Saved results to %s\n', outfn);
