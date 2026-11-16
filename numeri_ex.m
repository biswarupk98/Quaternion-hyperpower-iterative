%% test_quaternion_inverse_n1500.m
% Compare quaternion pseudoinverse methods for single size n = 1500
clear; clc; close all;

% --------------------
% Parameters
% --------------------
sizes = 1500;         % <-- only run for n = 1500
tolerance = 1e-10;

methods = {'QSVD','SAI','Newton–Schulz','HONSchulz','HPI19','RAPID','HybridRSPNS'};
numM = numel(methods);
numN = numel(sizes);

Times = zeros(numM,numN);
Iters = zeros(numM,numN);
Err_AXherm = zeros(numM,numN);
Err_XAherm = zeros(numM,numN);
Err_AXA = zeros(numM,numN);
Err_XAX = zeros(numM,numN);

rng(0); % reproducible random matrices

for s = 1:numN
    n = sizes(s);
    fprintf('\n===== Matrix size %d =====\n', n);

    % Random quaternion matrix A (components in (0,1))
    A0 = rand(n,n); A1 = rand(n,n);
    A2 = rand(n,n); A3 = rand(n,n);
    A  = quaternion(A0,A1,A2,A3);

    % Compute scaling alp via complex embedding
    A11 = complex(A0,A1); A12 = complex(A2,A3);
    CA  = [A11, A12; -conj(A12), conj(A11)];
    try
        svals = svd(CA,'econ');
        alp = 1/(max(svals)^2);
    catch
        % fallback small alpha if SVD fails for huge matrices in out-of-memory
        alp = 1/ (norm(CA,'fro')^2 + eps);
    end

    % 1) QSVD (pinv)
    try
        tstart = tic;
        X = pinv(A);
        Times(1,s) = toc(tstart);
        Iters(1,s) = 1;
        Err_AXherm(1,s) = norm((A*X)'-A*X,'fro');
        Err_XAherm(1,s) = norm((X*A)'-X*A,'fro');
        Err_AXA(1,s) = norm(A*X*A - A,'fro');
        Err_XAX(1,s) = norm(X*A*X - X,'fro');
    catch ME
        warning('QSVD failed: %s', ME.message);
        Times(1,s)=NaN; Iters(1,s)=NaN; Err_AXherm(1,s)=NaN; Err_XAherm(1,s)=NaN;
        Err_AXA(1,s)=NaN; Err_XAX(1,s)=NaN;
    end

    % 2) Quaternion SAI (toolbox)
    try
        tstart = tic;
        [X,it] = quaternion_SAI1(A,alp,tolerance);
        Times(2,s) = toc(tstart);
        Iters(2,s) = it;
        Err_AXherm(2,s) = norm((A*X)'-A*X,'fro');
        Err_XAherm(2,s) = norm((X*A)'-X*A,'fro');
        Err_AXA(2,s) = norm(A*X*A - A,'fro');
        Err_XAX(2,s) = norm(X*A*X - X,'fro');
    catch ME
        warning('SAI failed: %s', ME.message);
        Times(2,s)=NaN; Iters(2,s)=NaN; Err_AXherm(2,s)=NaN; Err_XAherm(2,s)=NaN;
        Err_AXA(2,s)=NaN; Err_XAX(2,s)=NaN;
    end

    % 3) Newton–Schulz
    try
        tstart = tic;
        [X,it,~] = quaternion_NewtonSchulz(A,tolerance,200,0.5,alp);
        Times(3,s) = toc(tstart);
        Iters(3,s) = it;
        Err_AXherm(3,s) = norm((A*X)'-A*X,'fro');
        Err_XAherm(3,s) = norm((X*A)'-X*A,'fro');
        Err_AXA(3,s) = norm(A*X*A - A,'fro');
        Err_XAX(3,s) = norm(X*A*X - X,'fro');
    catch ME
        warning('NewtonSchulz failed: %s', ME.message);
        Times(3,s)=NaN; Iters(3,s)=NaN; Err_AXherm(3,s)=NaN; Err_XAherm(3,s)=NaN;
        Err_AXA(3,s)=NaN; Err_XAX(3,s)=NaN;
    end

    % 4) HONSchulz
    try
        tstart = tic;
        [X,it,~] = quaternion_HONSchulz(A,tolerance,200);
        Times(4,s) = toc(tstart);
        Iters(4,s) = it;
        Err_AXherm(4,s) = norm((A*X)'-A*X,'fro');
        Err_XAherm(4,s) = norm((X*A)'-X*A,'fro');
        Err_AXA(4,s) = norm(A*X*A - A,'fro');
        Err_XAX(4,s) = norm(X*A*X - X,'fro');
    catch ME
        warning('HONSchulz failed: %s', ME.message);
        Times(4,s)=NaN; Iters(4,s)=NaN; Err_AXherm(4,s)=NaN; Err_XAherm(4,s)=NaN;
        Err_AXA(4,s)=NaN; Err_XAX(4,s)=NaN;
    end

    % 5) HPI19
    try
        X0 = alp * A';
        tstart = tic;
        [X,it] = quaternion_Hyper19(A,X0,tolerance);
        Times(5,s) = toc(tstart);
        Iters(5,s) = it;
        Err_AXherm(5,s) = norm((A*X)'-A*X,'fro');
        Err_XAherm(5,s) = norm((X*A)'-X*A,'fro');
        Err_AXA(5,s) = norm(A*X*A - A,'fro');
        Err_XAX(5,s) = norm(X*A*X - X,'fro');
    catch ME
        warning('HPI19 failed: %s', ME.message);
        Times(5,s)=NaN; Iters(5,s)=NaN; Err_AXherm(5,s)=NaN; Err_XAherm(5,s)=NaN;
        Err_AXA(5,s)=NaN; Err_XAX(5,s)=NaN;
    end

    % 6) RAPID
    try
        tstart = tic;
        [X,it] = quaternion_RAPID1(A,5,alp,tolerance);
        Times(6,s) = toc(tstart);
        Iters(6,s) = it;
        Err_AXherm(6,s) = norm((A*X)'-A*X,'fro');
        Err_XAherm(6,s) = norm((X*A)'-X*A,'fro');
        Err_AXA(6,s) = norm(A*X*A - A,'fro');
        Err_XAX(6,s) = norm(X*A*X - X,'fro');
    catch ME
        warning('RAPID failed: %s', ME.message);
        Times(6,s)=NaN; Iters(6,s)=NaN; Err_AXherm(6,s)=NaN; Err_XAherm(6,s)=NaN;
        Err_AXA(6,s)=NaN; Err_XAX(6,s)=NaN;
    end

    % 7) HybridRSPNS
    try
        tstart = tic;
        [X,info] = quaternion_HybridRSPNS(A,8,4,5,tolerance,200);
        Times(7,s) = toc(tstart);
        Iters(7,s) = info.iters;
        Err_AXherm(7,s) = norm((A*X)'-A*X,'fro');
        Err_XAherm(7,s) = norm((X*A)'-X*A,'fro');
        Err_AXA(7,s) = norm(A*X*A - A,'fro');
        Err_XAX(7,s) = norm(X*A*X - X,'fro');
    catch ME
        warning('HybridRSPNS failed: %s', ME.message);
        Times(7,s)=NaN; Iters(7,s)=NaN; Err_AXherm(7,s)=NaN; Err_XAherm(7,s)=NaN;
        Err_AXA(7,s)=NaN; Err_XAX(7,s)=NaN;
    end

end % end size loop

% --------------------
% Print results for n = 1500
% --------------------
pickSize = 1500;
idx = find(sizes == pickSize,1);

fprintf('\n===== Results for matrix size %d =====\n', sizes(idx));
fprintf('%-15s %-10s %-8s %-18s %-18s %-18s %-18s\n', ...
    'Method','Time(s)','Iters','||AXA - A||_F','||XAX - X||_F','||(AX)^T - AX||_F','||(XA)^T - XA||_F');

for m = 1:numM
    fprintf('%-15s %-10.3e %-8d %-18.2e %-18.2e %-18.2e %-18.2e\n', ...
        methods{m}, Times(m,idx), Iters(m,idx), ...
        Err_AXA(m,idx), Err_XAX(m,idx), Err_AXherm(m,idx), Err_XAherm(m,idx));
end

% --------------------
% Plots (single-point per method)
% --------------------
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% 1) CPU time
figure; hold on; grid on;
for m=1:numM, plot(sizes,Times(m,:),'-o','LineWidth',1.5,'MarkerSize',8); end
xlabel('Matrix size $n$','Interpreter','latex'); ylabel('CPU time (s)','Interpreter','latex');
title('CPU Time (n = 1500)','Interpreter','latex');
legend(methods,'Location','northwest');

% 2) AXA error
figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_AXA(m,:),'-o','LineWidth',1.5,'MarkerSize',8); end
xlabel('Matrix size $n$','Interpreter','latex'); ylabel('$\|AXA - A\|_F$','Interpreter','latex');
title('AXA Error (n = 1500)','Interpreter','latex');
legend(methods,'Location','northeast');

% 3) XAX error
figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_XAX(m,:),'-o','LineWidth',1.5,'MarkerSize',8); end
xlabel('Matrix size $n$','Interpreter','latex'); ylabel('$\|XAX - X\|_F$','Interpreter','latex');
title('XAX Error (n = 1500)','Interpreter','latex');
legend(methods,'Location','northeast');

% 4) AX Hermitian
figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_AXherm(m,:),'-o','LineWidth',1.5,'MarkerSize',8); end
xlabel('Matrix size $n$','Interpreter','latex'); ylabel('$\|AX - (AX)^\ast\|_F$','Interpreter','latex');
title('AX Hermitian Error (n = 1500)','Interpreter','latex');
legend(methods,'Location','northeast');

% 5) XA Hermitian
figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_XAherm(m,:),'-o','LineWidth',1.5,'MarkerSize',8); end
xlabel('Matrix size $n$','Interpreter','latex'); ylabel('$\|(XA)^\ast - XA\|_F$','Interpreter','latex');
title('XA Hermitian Error (n = 1500)','Interpreter','latex');
legend(methods,'Location','northeast');

