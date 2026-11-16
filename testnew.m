%% test_quaternion_inverse.m
% Compare quaternion pseudoinverse methods (QTFM toolbox)
% Matrix sizes from 50 to 1000 with step 50

clear; clc; close all;

sizes = 200:50:1200;
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

for s = 1:numN
    n = sizes(s);
    fprintf('\n===== Matrix size %d =====\n',n);

    % Random quaternion matrix
    A0 = rand(n,n); A1 = rand(n,n);
    A2 = rand(n,n); A3 = rand(n,n);
    A  = quaternion(A0,A1,A2,A3);

    % Scaling factor alpha
    A11 = complex(A0,A1); A12 = complex(A2,A3);
    CA  = [A11, A12; -conj(A12), conj(A11)];
    svals = svd(CA,'econ');
    alp = 1/(max(svals)^2);

    %% --- Method 1: QSVD ---
    tic; X = pinv(A); Times(1,s)=toc;
    Iters(1,s)=1;
    Err_AXherm(1,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(1,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(1,s)=norm(A*X*A-A,'fro');
    Err_XAX(1,s)=norm(X*A*X-X,'fro');

    %% --- Method 2: Quaternion SAI ---
    tic; [X,it] = quaternion_SAI1(A,alp,tolerance); Times(2,s)=toc;
    Iters(2,s)=it;
    Err_AXherm(2,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(2,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(2,s)=norm(A*X*A-A,'fro');
    Err_XAX(2,s)=norm(X*A*X-X,'fro');

    %% --- Method 3: Newton–Schulz ---
    tic; [X,~,~] = quaternion_NewtonSchulz(A,tolerance,200,0.5); Times(3,s)=toc;
    Iters(3,s)=200;
    Err_AXherm(3,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(3,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(3,s)=norm(A*X*A-A,'fro');
    Err_XAX(3,s)=norm(X*A*X-X,'fro');

    %% --- Method 4: HONSchulz ---
    tic; [X,~,~] = quaternion_HONSchulz(A,tolerance,200); Times(4,s)=toc;
    Iters(4,s)=200;
    Err_AXherm(4,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(4,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(4,s)=norm(A*X*A-A,'fro');
    Err_XAX(4,s)=norm(X*A*X-X,'fro');

    %% --- Method 5: HPI19 ---
    X0 = alp*A';
    tic; [X,it] = quaternion_Hyper19(A,X0,tolerance); Times(5,s)=toc;
    Iters(5,s)=it;
    Err_AXherm(5,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(5,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(5,s)=norm(A*X*A-A,'fro');
    Err_XAX(5,s)=norm(X*A*X-X,'fro');

    %% --- Method 6: RAPID ---
    tic; [X,it] = quaternion_RAPID1(A,5,alp,tolerance); Times(6,s)=toc;
    Iters(6,s)=it;
    Err_AXherm(6,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(6,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(6,s)=norm(A*X*A-A,'fro');
    Err_XAX(6,s)=norm(X*A*X-X,'fro');

    %% --- Method 7: HybridRSPNS ---
    tic; [X,info] = quaternion_HybridRSPNS(A,8,4,5,tolerance,200); Times(7,s)=toc;
    Iters(7,s)=info.iters;
    Err_AXherm(7,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(7,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(7,s)=norm(A*X*A-A,'fro');
    Err_XAX(7,s)=norm(X*A*X-X,'fro');
end

%% === Results Plots ===
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% 1. Time vs matrix size
figure; hold on; grid on;
for m=1:numM, plot(sizes,Times(m,:),'-o','LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('CPU time (s)','Interpreter','latex');
title('CPU Time vs Matrix Size','Interpreter','latex');
legend(methods,'Location','northwest');

% 2. Error: ||AX - (AX)'||_F
figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_AXherm(m,:),'-o','LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|AX - (AX)^\ast\|_F$','Interpreter','latex');
title('AX Hermitian Error','Interpreter','latex');
legend(methods,'Location','northeast');

% 3. Error: ||(XA)' - XA||_F
figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_XAherm(m,:),'-o','LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|(XA)^\ast - XA\|_F$','Interpreter','latex');
title('XA Hermitian Error','Interpreter','latex');
legend(methods,'Location','northeast');

% 4. Error: ||AXA - A||_F
figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_AXA(m,:),'-o','LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|AXA - A\|_F$','Interpreter','latex');
title('AXA Consistency Error','Interpreter','latex');
legend(methods,'Location','northeast');

% 5. Error: ||XAX - X||_F
figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_XAX(m,:),'-o','LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|XAX - X\|_F$','Interpreter','latex');
title('XAX Consistency Error','Interpreter','latex');
legend(methods,'Location','northeast');


%% === Print Table for one particular size ===
pickSize = 300;                   % choose matrix size here
idx = find(sizes == pickSize,1);  % index in results

if isempty(idx)
    error('Chosen size not in tested sizes');
end

fprintf('\n===== Results for matrix size %d =====\n', pickSize);
fprintf('%-15s %-10s %-8s %-12s %-12s %-12s %-12s\n', ...
    'Method','Time(s)','Iters','||AX-AX^*||','||(XA)^*-XA||','||AXA-A||','||XAX-X||');

for m = 1:numM
    fprintf('%-15s %-10.3e %-8d %-12.2e %-12.2e %-12.2e %-12.2e\n', ...
        methods{m}, Times(m,idx), Iters(m,idx), ...
        Err_AXherm(m,idx), Err_XAherm(m,idx), ...
        Err_AXA(m,idx), Err_XAX(m,idx));
end
