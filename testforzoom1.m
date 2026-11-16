%% test_quaternion_inverse_full.m
% Full experiment + enhanced plotting
% Compare quaternion pseudoinverse methods (QTFM toolbox)
% Matrix sizes from 100 to 600 step 50 (adjust sizes as needed)

clear; clc; close all;

%% -------------------- PARAMETERS --------------------
sizes = 100:20:400;
tolerance = 1e-10;

% original compute-order methods used in the loop (indexing must match compute block)
methods_orig = {'QSVD','SAI','Newton--Schulz','HONSchulz','HPI19','RAPID','HybridRSPNS'};
numM = numel(methods_orig);
numN = numel(sizes);

% preallocate results
Times = zeros(numM,numN);
Iters = zeros(numM,numN);
Err_AXherm = zeros(numM,numN);
Err_XAherm = zeros(numM,numN);
Err_AXA = zeros(numM,numN);
Err_XAX = zeros(numM,numN);

%% ----------------- COMPUTATION LOOP -----------------
for s = 1:numN
    n = sizes(s);
    fprintf('\n===== Matrix size %d =====\n',n);

    % Random quaternion matrix (uniform [0,1])
    A0 = rand(n,n); A1 = rand(n,n);
    A2 = rand(n,n); A3 = rand(n,n);
    A  = quaternion(A0,A1,A2,A3);

    % Scaling factor alpha via complex embedding (largest singular value)
    A11 = complex(A0,A1); A12 = complex(A2,A3);
    CA  = [A11, A12; -conj(A12), conj(A11)];
    % compute largest singular value (svds is faster than full svd for bigger n)
    try
        sigma_max = svds(CA,1,'largest');
    catch
        sv = svd(CA,'econ'); sigma_max = max(sv);
    end
    alp = 1/(sigma_max^2);

    %% --- Method 1: QSVD ---
    try
        t0 = tic; X = pinv(A); Times(1,s)=toc(t0);
        Iters(1,s)=1;
        Err_AXherm(1,s)=norm((A*X)'-A*X,'fro');
        Err_XAherm(1,s)=norm((X*A)'-X*A,'fro');
        Err_AXA(1,s)=norm(A*X*A-A,'fro');
        Err_XAX(1,s)=norm(X*A*X-X,'fro');
    catch ME
        warning('QSVD failed at n=%d: %s', n, ME.message);
        Times(1,s)=Inf; Iters(1,s)=0;
        Err_AXherm(1,s)=NaN; Err_XAherm(1,s)=NaN; Err_AXA(1,s)=NaN; Err_XAX(1,s)=NaN;
    end

    %% --- Method 2: Quaternion SAI ---
    try
        t0 = tic; [X,it] = quaternion_SAI1(A,alp,tolerance); Times(2,s)=toc(t0);
        Iters(2,s)=it;
        Err_AXherm(2,s)=norm((A*X)'-A*X,'fro');
        Err_XAherm(2,s)=norm((X*A)'-X*A,'fro');
        Err_AXA(2,s)=norm(A*X*A-A,'fro');
        Err_XAX(2,s)=norm(X*A*X-X,'fro');
    catch ME
        warning('SAI failed at n=%d: %s', n, ME.message);
        Times(2,s)=Inf; Iters(2,s)=0;
        Err_AXherm(2,s)=NaN; Err_XAherm(2,s)=NaN; Err_AXA(2,s)=NaN; Err_XAX(2,s)=NaN;
    end

    %% --- Method 3: Newton–Schulz (ensure alp passed) ---
    try
        t0 = tic; [X,it,~] = quaternion_NewtonSchulz(A,tolerance,200,0.5,alp); Times(3,s)=toc(t0);
        Iters(3,s)=it;
        Err_AXherm(3,s)=norm((A*X)'-A*X,'fro');
        Err_XAherm(3,s)=norm((X*A)'-X*A,'fro');
        Err_AXA(3,s)=norm(A*X*A-A,'fro');
        Err_XAX(3,s)=norm(X*A*X-X,'fro');
    catch ME
        warning('Newton–Schulz failed at n=%d: %s', n, ME.message);
        Times(3,s)=Inf; Iters(3,s)=0;
        Err_AXherm(3,s)=NaN; Err_XAherm(3,s)=NaN; Err_AXA(3,s)=NaN; Err_XAX(3,s)=NaN;
    end

    %% --- Method 4: HONSchulz ---
    try
        t0 = tic; [X,it,~] = quaternion_HONSchulz(A,tolerance,200); Times(4,s)=toc(t0);
        Iters(4,s)=it;
        Err_AXherm(4,s)=norm((A*X)'-A*X,'fro');
        Err_XAherm(4,s)=norm((X*A)'-X*A,'fro');
        Err_AXA(4,s)=norm(A*X*A-A,'fro');
        Err_XAX(4,s)=norm(X*A*X-X,'fro');
    catch ME
        warning('HONSchulz failed at n=%d: %s', n, ME.message);
        Times(4,s)=Inf; Iters(4,s)=0;
        Err_AXherm(4,s)=NaN; Err_XAherm(4,s)=NaN; Err_AXA(4,s)=NaN; Err_XAX(4,s)=NaN;
    end

    %% --- Method 5: HPI19 ---
    try
        X0 = alp*A';
        t0 = tic; [X,it] = quaternion_Hyper19(A,X0,tolerance); Times(5,s)=toc(t0);
        Iters(5,s)=it;
        Err_AXherm(5,s)=norm((A*X)'-A*X,'fro');
        Err_XAherm(5,s)=norm((X*A)'-X*A,'fro');
        Err_AXA(5,s)=norm(A*X*A-A,'fro');
        Err_XAX(5,s)=norm(X*A*X-X,'fro');
    catch ME
        warning('HPI19 failed at n=%d: %s', n, ME.message);
        Times(5,s)=Inf; Iters(5,s)=0;
        Err_AXherm(5,s)=NaN; Err_XAherm(5,s)=NaN; Err_AXA(5,s)=NaN; Err_XAX(5,s)=NaN;
    end

    %% --- Method 6: RAPID ---
    try
        t0 = tic; [X,it] = quaternion_RAPID1(A,5,alp,tolerance); Times(6,s)=toc(t0);
        Iters(6,s)=it;
        Err_AXherm(6,s)=norm((A*X)'-A*X,'fro');
        Err_XAherm(6,s)=norm((X*A)'-X*A,'fro');
        Err_AXA(6,s)=norm(A*X*A-A,'fro');
        Err_XAX(6,s)=norm(X*A*X-X,'fro');
    catch ME
        warning('RAPID failed at n=%d: %s', n, ME.message);
        Times(6,s)=Inf; Iters(6,s)=0;
        Err_AXherm(6,s)=NaN; Err_XAherm(6,s)=NaN; Err_AXA(6,s)=NaN; Err_XAX(6,s)=NaN;
    end

    %% --- Method 7: HybridRSPNS ---
    try
        t0 = tic; [X,info] = quaternion_HybridRSPNS(A,8,4,5,tolerance,500); Times(7,s)=toc(t0);
        Iters(7,s)=info.iters;
        Err_AXherm(7,s)=norm((A*X)'-A*X,'fro');
        Err_XAherm(7,s)=norm((X*A)'-X*A,'fro');
        Err_AXA(7,s)=norm(A*X*A-A,'fro');
        Err_XAX(7,s)=norm(X*A*X-X,'fro');
    catch ME
        warning('HybridRSPNS failed at n=%d: %s', n, ME.message);
        Times(7,s)=Inf; Iters(7,s)=0;
        Err_AXherm(7,s)=NaN; Err_XAherm(7,s)=NaN; Err_AXA(7,s)=NaN; Err_XAX(7,s)=NaN;
    end
end

%% ---------------- REORDER METHODS FOR PLOTTING ----------------
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

%% -------------------- ENHANCED PLOTTING --------------------
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% Use circle marker for all
marker = '-o';
markerList = repmat({marker},1,numM);

% Increase marker size and linewidth for clarity
mksz = 7;
lw = 1.6;

numM = numel(methods_new);

% ---------- CPU: Zoomed vertical range to emphasize low-cost methods ----------
fig1 = figure('Units','normalized','Position',[0.05 0.05 0.55 0.75]);
ax1 = axes('Position',[0.08 0.11 0.60 0.78]); hold(ax1,'on'); grid(ax1,'on');

h = gobjects(numM,1);
for m=1:numM
    h(m) = plot(ax1, sizes, Times(m,:), markerList{m}, 'LineWidth', lw, 'MarkerSize', mksz);
end
xlabel(ax1,'Matrix size $n$','Interpreter','latex');
ylabel(ax1,'CPU time (s)','Interpreter','latex');
title(ax1,'CPU Time vs Matrix Size (zoomed vertical)','Interpreter','latex');

% vertical zoom: 92nd percentile to avoid spikes (tweak if needed)
allTimes = Times(:);
validTimes = allTimes(~isinf(allTimes) & ~isnan(allTimes));
if isempty(validTimes)
    yZoom = 1;
else
    yZoom = prctile(validTimes,92);
end
ylim(ax1, [0 yZoom]);
xlim(ax1, [min(sizes) max(sizes)]);
legend(ax1, methods_new, 'Location','northwest');

% ---------- CPU: Neighborhood zoom (600-800) to show SAI region ----------
xL = 600; xR = 800;
% clamp to sizes
xL = max(xL, min(sizes)); xR = min(xR, max(sizes));

fig2 = figure('Units','normalized','Position',[0.62 0.55 0.35 0.35]);
ax2 = axes('Position',[0.12 0.15 0.82 0.78]); hold(ax2,'on'); grid(ax2,'on');
for m=1:numM
    plot(ax2, sizes, Times(m,:), markerList{m}, 'LineWidth', lw, 'MarkerSize', mksz);
end
xlim(ax2, [xL xR]);

% compute vertical limits around cluster, excluding extreme outliers
ix = (sizes >= xL) & (sizes <= xR);
vals = Times(:, ix);
vals = vals(~isinf(vals(:)) & ~isnan(vals(:)));
if isempty(vals)
    ymin = 0; ymax = 1;
else
    ymin = min(vals(:));
    ymax = prctile(vals(:),85);
end
if ymin == ymax
    ymax = ymin + eps;
end
yPad = 0.08*(ymax - ymin + eps);
ylim(ax2, [max(0,ymin - yPad) ymax + yPad]);
xlabel(ax2,'Matrix size $n$','Interpreter','latex');
ylabel(ax2,'CPU time (s)','Interpreter','latex');
title(ax2,sprintf('CPU time (neighbourhood %d--%d)',xL,xR),'Interpreter','latex');
legend(ax2, methods_new, 'Location','northeastoutside');

% ---------- CPU: Full-range figure (for context) ----------
fig3 = figure('Units','normalized','Position',[0.62 0.05 0.35 0.45]);
ax3 = axes('Position',[0.12 0.12 0.82 0.82]); hold(ax3,'on'); grid(ax3,'on');
for m=1:numM
    plot(ax3, sizes, Times(m,:), markerList{m}, 'LineWidth', lw, 'MarkerSize', mksz);
end
xlabel(ax3,'Matrix size $n$','Interpreter','latex');
ylabel(ax3,'CPU time (s)','Interpreter','latex');
title(ax3,'CPU Time vs Matrix Size (full range)','Interpreter','latex');
legend(ax3, methods_new, 'Location','northwest');

% ---------- Error plots (semilogy) with zoom and bar view ----------
% Representative size for bar comparison (choose in SAI neighbourhood)
repN = 600;
[~, idxRep] = min(abs(sizes - repN));

% AX Hermitian: full semilogy + zoom
figE1 = figure('Units','normalized','Position',[0.02 0.02 0.45 0.38]);
axE1 = subplot(1,2,1); hold(axE1,'on'); grid(axE1,'on');
for m=1:numM, semilogy(axE1, sizes, Err_AXherm(m,:), markerList{m}, 'LineWidth', lw, 'MarkerSize', mksz); end
xlabel(axE1,'Matrix size $n$','Interpreter','latex');
ylabel(axE1,'$\|AX-(AX)^\ast\|_F$','Interpreter','latex');
title(axE1,'AX Hermitian error (full)','Interpreter','latex');
legend(axE1, methods_new, 'Location','northeast');

axE2 = subplot(1,2,2); hold(axE2,'on'); grid(axE2,'on');
% use the same neighbourhood x-range for zoom
yvals = Err_AXherm(:, ix);
yvals = yvals(~isinf(yvals(:)) & ~isnan(yvals(:)));
if isempty(yvals)
    ymin = eps; ymax = 1;
else
    ymin = max(min(yvals(:)), eps);
    ymax = prctile(yvals(:),85);
end
% replot with markers and styles
for m=1:numM, semilogy(axE2, sizes, Err_AXherm(m,:), markerList{m}, 'LineWidth', lw, 'MarkerSize', mksz); end
xlim(axE2, [min(sizes) max(sizes)]);
ylim(axE2, [ymin, max(ymax, 10*ymin)]);
xlabel(axE2,'Matrix size $n$','Interpreter','latex');
title(axE2,'AX Hermitian error (zoom)','Interpreter','latex');

% Bar plots (log10) at representative size
figBar = figure('Units','normalized','Position',[0.5 0.02 0.45 0.38]);
subplot(1,2,1);
barVals = Err_AXherm(:, idxRep);
bar(log10(max(barVals,eps)));
set(gca,'XTick',1:numM,'XTickLabel',methods_new,'XTickLabelRotation',45,'TickLabelInterpreter','latex');
ylabel('log_{10} error','Interpreter','latex');
title(sprintf('AX Hermitian error at n=%d', sizes(idxRep)),'Interpreter','latex');

subplot(1,2,2);
barVals2 = Err_XAX(:, idxRep);
bar(log10(max(barVals2,eps)));
set(gca,'XTick',1:numM,'XTickLabel',methods_new,'XTickLabelRotation',45,'TickLabelInterpreter','latex');
title(sprintf('XAX error at n=%d', sizes(idxRep)),'Interpreter','latex');

% ---------- Heatmap of AX Hermitian (log10) ----------
figHeat = figure('Units','normalized','Position',[0.02 0.42 0.45 0.5]);
imagesc(sizes, 1:numM, log10(max(Err_AXherm, eps)));
colormap(parula); colorbar;
set(gca,'YTick',1:numM,'YTickLabel',methods_new,'TickLabelInterpreter','latex');
xlabel('Matrix size $n$','Interpreter','latex');
title('log_{10}(\|AX-(AX)^* \|_F) across sizes','Interpreter','latex');

% ---------- Improve overall aesthetics ----------
allFigs = findall(0,'Type','figure');
for k = 1:numel(allFigs)
    figk = allFigs(k);
    axesInFig = findall(figk,'type','axes');
    for a = 1:numel(axesInFig)
        set(axesInFig(a),'FontSize',10);
    end
end

%% -------------------- PRINT TABLE (example size) --------------------
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
