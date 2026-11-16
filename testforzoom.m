%% test_quaternion_inverse_zoomed.m
% Compare quaternion pseudoinverse methods (QTFM toolbox)
% Matrix sizes from 300 to 1300 step 50

clear; clc; close all;

sizes = 200:50:1200;
tolerance = 1e-10;

% original compute-order methods used in the loop (keep this for indexing)
methods_orig = {'QSVD','SAI','Newton–Schulz','HONSchulz','HPI19','RAPID','HybridRSPNS'};
numM = numel(methods_orig);
numN = numel(sizes);

% preallocate results
Times = zeros(numM,numN);
Iters = zeros(numM,numN);
Err_AXherm = zeros(numM,numN);
Err_XAherm = zeros(numM,numN);
Err_AXA = zeros(numM,numN);
Err_XAX = zeros(numM,numN);

%% ----------------------------
% === COMPUTATION (your loop) ==
% -----------------------------
for s = 1:numN
    n = sizes(s);
    fprintf('\n===== Matrix size %d =====\n',n);

    % Random quaternion matrix
    A0 = rand(n,n); A1 = rand(n,n);
    A2 = rand(n,n); A3 = rand(n,n);
    A  = quaternion(A0,A1,A2,A3);

    % Scaling factor alpha (via complex embedding)
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
    tic; [X,it,~] = quaternion_NewtonSchulz(A,tolerance,200,0.5,alp); Times(3,s)=toc;
    Iters(3,s)=it;
    Err_AXherm(3,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(3,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(3,s)=norm(A*X*A-A,'fro');
    Err_XAX(3,s)=norm(X*A*X-X,'fro');

    %% --- Method 4: HONSchulz ---
    tic; [X,it,~] = quaternion_HONSchulz(A,tolerance,200); Times(4,s)=toc;
    Iters(4,s)=it;
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
    tic; [X,info] = quaternion_HybridRSPNS(A,8,4,5,tolerance,500); Times(7,s)=toc;
    Iters(7,s)=info.iters;
    Err_AXherm(7,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(7,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(7,s)=norm(A*X*A-A,'fro');
    Err_XAX(7,s)=norm(X*A*X-X,'fro');
end

%% === Reorder methods for plotting as requested ===
methods_new = {'SAI','RAPID','HPI19','QSVD','Newton--Schulz','HONSchulz','HybridRSPNS'};
% mapping original -> new index: original = [QSVD(1), SAI(2), NS(3), HON(4), HPI19(5), RAPID(6), Hybrid(7)]
newOrder = [2, 6, 5, 1, 3, 4, 7];   % (SAI,RAPID,HPI19,QSVD,NS,HON,Hybrid)

% reorder data arrays
Times = Times(newOrder, :);
Iters = Iters(newOrder, :);
Err_AXherm = Err_AXherm(newOrder, :);
Err_XAherm = Err_XAherm(newOrder, :);
Err_AXA = Err_AXA(newOrder, :);
Err_XAX = Err_XAX(newOrder, :);

%% === Results Plots (style preserved) ===
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% Use circle marker for all
marker = '-o';
markerList = repmat({marker},1,numM);  % one per method

% ---------- CPU: Zoomed vertical range to emphasize low-cost methods ----------
fig1 = figure('Units','normalized','Position',[0.1 0.1 0.65 0.7]);
ax1 = axes('Position',[0.08 0.11 0.60 0.78]); hold(ax1,'on'); grid(ax1,'on');

numM = numel(methods_new);
h = gobjects(numM,1);
for m=1:numM
    h(m) = plot(ax1, sizes, Times(m,:), markerList{m}, 'LineWidth',1.5, 'MarkerSize',6);
end
xlabel(ax1,'Matrix size $n$','Interpreter','latex');
ylabel(ax1,'CPU time (s)','Interpreter','latex');
title(ax1,'CPU Time vs Matrix Size (zoomed view)','Interpreter','latex');

% compute a vertical zoom limit that highlights the lower cluster while ignoring spikes
% use 92nd percentile of all Times to avoid extreme outliers
allTimes = Times(:);
yZoom = prctile(allTimes,92);
if yZoom <= 0
    yZoom = max(allTimes); 
end
ylim(ax1, [0 yZoom]);
xlim(ax1, [min(sizes) max(sizes)]);
legend(ax1, methods_new, 'Location','northwest');

% annotate that this is a zoomed view
text(ax1, min(sizes)+20, 0.95*yZoom, 'Zoomed vertical range','FontSize',10, 'Interpreter','latex');

% ---------- CPU: Full-range figure (for context) ----------
fig2 = figure('Units','normalized','Position',[0.78 0.1 0.45 0.6]);
axF = axes('Position',[0.1 0.12 0.85 0.80]); hold(axF,'on'); grid(axF,'on');
for m=1:numM
    plot(axF, sizes, Times(m,:), markerList{m}, 'LineWidth',1.5, 'MarkerSize',6);
end
xlabel(axF,'Matrix size $n$','Interpreter','latex');
ylabel(axF,'CPU time (s)','Interpreter','latex');
title(axF,'CPU Time vs Matrix Size','Interpreter','latex');
legend(axF, methods_new, 'Location','northwest');
xlim(axF, [min(sizes) max(sizes)]);

% ---------- Error plots: log-scale (semilogy) - differences visible ----------
% 1. AX Hermitian error
fig3 = figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_AXherm(m,:),markerList{m},'LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|AX - (AX)^\ast\|_F$','Interpreter','latex');
title('AX Hermitian Error','Interpreter','latex');
legend(methods_new,'Location','northeast');

% 2. XA Hermitian error
fig4 = figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_XAherm(m,:),markerList{m},'LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|(XA)^\ast - XA\|_F$','Interpreter','latex');
title('XA Hermitian Error','Interpreter','latex');
legend(methods_new,'Location','northeast');

% 3. AXA consistency
fig5 = figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_AXA(m,:),markerList{m},'LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|AXA - A\|_F$','Interpreter','latex');
title('AXA Consistency Error','Interpreter','latex');
legend(methods_new,'Location','northeast');

% 4. XAX consistency
fig6 = figure; hold on; grid on;
for m=1:numM, semilogy(sizes,Err_XAX(m,:),markerList{m},'LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|XAX - X\|_F$','Interpreter','latex');
title('XAX Consistency Error','Interpreter','latex');
legend(methods_new,'Location','northeast');

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
        methods_new{m}, Times(m,idx), Iters(m,idx), ...
        Err_AXherm(m,idx), Err_XAherm(m,idx), ...
        Err_AXA(m,idx), Err_XAX(m,idx));
end
