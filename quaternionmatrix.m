% clc; clear;
% 
% %% Dimensions
% n1 = 4;
% n2 = 3;
% 
% %% Generate random quaternion matrix components
% Xs = randn(n1,n2);
% Xx = randn(n1,n2);
% Xy = randn(n1,n2);
% Xz = randn(n1,n2);
% 
% %% -------- Quaternion Nuclear Norm via Complex Adjoint --------
% % Complex adjoint representation (Zhang 1997)
% 
% A = Xs + 1i*Xx;
% B = Xy + 1i*Xz;
% 
% X_complex = [ A  B;
%              -conj(B)  conj(A) ];
% 
% % Singular values of complex adjoint
% s_complex = svd(X_complex);
% 
% % Quaternion singular values appear twice in complex adjoint
% s_quat = s_complex(1:2:end);
% 
% nuclear_quat = sum(s_quat);
% 
% %% -------- Full Real Representation --------
% 
% XR = [  Xs   -Xx   -Xy   -Xz;
%         Xx    Xs   -Xz    Xy;
%         Xy    Xz    Xs   -Xx;
%         Xz   -Xy    Xx    Xs ];
% 
% s_real = svd(XR);
% 
% nuclear_real = sum(s_real);
% 
% %% -------- Verification --------
% 
% fprintf('Quaternion nuclear norm        : %.6f\n', nuclear_quat);
% fprintf('Real nuclear norm (full block) : %.6f\n', nuclear_real);
% fprintf('Real/4                         : %.6f\n', nuclear_real/4);
% fprintf('Difference                     : %.6e\n', ...
%         abs(nuclear_quat - nuclear_real/4));
clc; clear;

%% Dimensions
n1 = 4;
n2 = 3;

%% Random quaternion matrix
Xs = randn(n1,n2);
Xx = randn(n1,n2);
Xy = randn(n1,n2);
Xz = randn(n1,n2);

%% ------------------------------------------------------------
%% 1) Quaternion singular values via complex adjoint
%% ------------------------------------------------------------

A = Xs + 1i*Xx;
B = Xy + 1i*Xz;

X_complex = [ A  B;
             -conj(B)  conj(A) ];

s_complex = svd(X_complex);

% Quaternion singular values (appear twice)
s_quat = s_complex(1:2:end);

%% ------------------------------------------------------------
%% 2) Full real embedding singular values
%% ------------------------------------------------------------

XR = [  Xs   -Xx   -Xy   -Xz;
        Xx    Xs   -Xz    Xy;
        Xy    Xz    Xs   -Xx;
        Xz   -Xy    Xx    Xs ];

s_real_full = svd(XR);

%% ------------------------------------------------------------
%% 3) Column representation singular values
%% ------------------------------------------------------------

XcR = [Xs; Xx; Xy; Xz];   % 4n1 × n2

s_real_column = svd(XcR);

%% ------------------------------------------------------------
%% Display results
%% ------------------------------------------------------------

disp('Quaternion singular values:')
disp(s_quat')

disp('Full real embedding singular values:')
disp(s_real_full')

disp('Column representation singular values:')
disp(s_real_column')

%% Nuclear norms
nuclear_quat = sum(s_quat);
nuclear_real_full = sum(s_real_full);
nuclear_real_column = sum(s_real_column);

fprintf('\nNuclear Norm Comparison\n');
fprintf('Quaternion nuclear norm      : %.6f\n', nuclear_quat);
fprintf('Full real nuclear norm /4    : %.6f\n', nuclear_real_full/4);
fprintf('Column representation norm   : %.6f\n', nuclear_real_column);