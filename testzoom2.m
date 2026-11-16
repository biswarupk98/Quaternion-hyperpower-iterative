%% test_quaternion_inverse_zoomed_with_left_inset_one_arrow.m
% Full test script with left-side inset zooms (one arrow, inset under legend).
clear; clc; close all;

% ----------------------
% Parameters
% ----------------------
sizes = 200:50:1200;         % matrix sizes
tolerance = 1e-10;

methods_orig = {'QSVD','SAI','Newton–Schulz','HONSchulz','HPI19','RAPID','HybridRSPNS'};
numM = numel(methods_orig);
numN = numel(sizes);

% Preallocate results
Times = zeros(numM,numN);
Iters = zeros(numM,numN);
Err_AXherm = zeros(numM,numN);
Err_XAherm = zeros(numM,numN);
Err_AXA = zeros(numM,numN);
Err_XAX = zeros(numM,numN);

% ----------------------
% Main computation loop (kept same as your original)
% ----------------------
for s = 1:numN
    n = sizes(s);
    fprintf('\n===== Matrix size %d =====\n',n);

    % Random quaternion matrix components
    A0 = rand(n,n); A1 = rand(n,n);
    A2 = rand(n,n); A3 = rand(n,n);
    A  = quaternion(A0,A1,A2,A3);

    % Compute scaling alpha using complex embedding
    A11 = complex(A0,A1); A12 = complex(A2,A3);
    CA  = [A11, A12; -conj(A12), conj(A11)];
    svals = svd(CA,'econ');
    alp = 1/(max(svals)^2);

    % --- Method 1: QSVD (toolbox pseudo-inverse) ---
    tic; X = pinv(A); Times(1,s)=toc;
    Iters(1,s)=1;
    Err_AXherm(1,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(1,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(1,s)=norm(A*X*A-A,'fro');
    Err_XAX(1,s)=norm(X*A*X-X,'fro');

    % --- Method 2: Quaternion SAI (toolbox) ---
    tic; [X,it] = quaternion_SAI1(A,alp,tolerance); Times(2,s)=toc;
    Iters(2,s)=it;
    Err_AXherm(2,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(2,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(2,s)=norm(A*X*A-A,'fro');
    Err_XAX(2,s)=norm(X*A*X-X,'fro');

    % --- Method 3: Newton–Schulz ---
    tic; [X,it,~] = quaternion_NewtonSchulz(A,tolerance,200,0.5,alp); Times(3,s)=toc;
    Iters(3,s)=it;
    Err_AXherm(3,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(3,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(3,s)=norm(A*X*A-A,'fro');
    Err_XAX(3,s)=norm(X*A*X-X,'fro');

    % --- Method 4: HONSchulz ---
    tic; [X,it,~] = quaternion_HONSchulz(A,tolerance,200); Times(4,s)=toc;
    Iters(4,s)=it;
    Err_AXherm(4,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(4,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(4,s)=norm(A*X*A-A,'fro');
    Err_XAX(4,s)=norm(X*A*X-X,'fro');

    % --- Method 5: HPI19 ---
    X0 = alp*A';
    tic; [X,it] = quaternion_Hyper19(A,X0,tolerance); Times(5,s)=toc;
    Iters(5,s)=it;
    Err_AXherm(5,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(5,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(5,s)=norm(A*X*A-A,'fro');
    Err_XAX(5,s)=norm(X*A*X-X,'fro');

    % --- Method 6: RAPID ---
    tic; [X,it] = quaternion_RAPID1(A,5,alp,tolerance); Times(6,s)=toc;
    Iters(6,s)=it;
    Err_AXherm(6,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(6,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(6,s)=norm(A*X*A-A,'fro');
    Err_XAX(6,s)=norm(X*A*X-X,'fro');

    % --- Method 7: HybridRSPNS ---
    tic; [X,info] = quaternion_HybridRSPNS(A,8,4,5,tolerance,500); Times(7,s)=toc;
    Iters(7,s)=info.iters;
    Err_AXherm(7,s)=norm((A*X)'-A*X,'fro');
    Err_XAherm(7,s)=norm((X*A)'-X*A,'fro');
    Err_AXA(7,s)=norm(A*X*A-A,'fro');
    Err_XAX(7,s)=norm(X*A*X-X,'fro');
end

% ----------------------
% Reorder methods for plotting (requested order)
% ----------------------
methods_new = {'SAI','RAPID','HPI19','QSVD','Newton--Schulz','HONSchulz','HybridRSPNS'};
newOrder = [2, 6, 5, 1, 3, 4, 7];   % mapping
Times = Times(newOrder, :);
Iters = Iters(newOrder, :);
Err_AXherm = Err_AXherm(newOrder, :);
Err_XAherm = Err_XAherm(newOrder, :);
Err_AXA = Err_AXA(newOrder, :);
Err_XAX = Err_XAX(newOrder, :);

% ----------------------
% Plotting settings
% ----------------------
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
marker = '-o';
numM = numel(methods_new);
markerList = repmat({marker},1,numM);

% ----------------------
% CPU: Zoomed cluster view (main)
% ----------------------
fig1 = figure('Units','normalized','Position',[0.05 0.08 0.55 0.70]);
ax1 = axes('Position',[0.08 0.11 0.60 0.78]); hold(ax1,'on'); grid(ax1,'on');
h = gobjects(numM,1);
for m=1:numM
    h(m) = plot(ax1, sizes, Times(m,:), markerList{m}, 'LineWidth',1.5, 'MarkerSize',6);
end
xlabel(ax1,'Matrix size $n$','Interpreter','latex');
ylabel(ax1,'CPU time (s)','Interpreter','latex');
title(ax1,'CPU Time vs Matrix Size (zoomed view)','Interpreter','latex');

% zoom vertical limit (92nd percentile)
allTimes = Times(:);
yZoom = prctile(allTimes,92);
if ~isfinite(yZoom) || yZoom<=0
    yZoom = max(allTimes);
end
ylim(ax1, [0 yZoom]);
xlim(ax1, [min(sizes) max(sizes)]);
leg = legend(ax1, methods_new, 'Location','northwest');

% ----------------------
% CPU: Full range figure (context)
% ----------------------
fig2 = figure('Units','normalized','Position',[0.70 0.08 0.55 0.70]);
axF = axes('Position',[0.10 0.12 0.85 0.80]); hold(axF,'on'); grid(axF,'on');
for m=1:numM
    plot(axF, sizes, Times(m,:), markerList{m}, 'LineWidth',1.5, 'MarkerSize',6);
end
xlabel(axF,'Matrix size $n$','Interpreter','latex');
ylabel(axF,'CPU time (s)','Interpreter','latex');
title(axF,'CPU Time vs Matrix Size (full range)','Interpreter','latex');
legend(axF, methods_new, 'Location','northwest');
xlim(axF, [min(sizes) max(sizes)]);

% ----------------------
% Error plots (semilogy)
% ----------------------
fig3 = figure('Units','normalized','Position',[0.02 0.02 0.45 0.35]); hold on; grid on;
for m=1:numM, semilogy(sizes,Err_AXherm(m,:),markerList{m},'LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|AX - (AX)^\ast\|_F$','Interpreter','latex');
title('AX Hermitian Error','Interpreter','latex');
legend(methods_new,'Location','northeast');

fig4 = figure('Units','normalized','Position',[0.50 0.02 0.45 0.35]); hold on; grid on;
for m=1:numM, semilogy(sizes,Err_XAherm(m,:),markerList{m},'LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|(XA)^\ast - XA\|_F$','Interpreter','latex');
title('XA Hermitian Error','Interpreter','latex');
legend(methods_new,'Location','northeast');

fig5 = figure('Units','normalized','Position',[0.02 0.40 0.45 0.35]); hold on; grid on;
for m=1:numM, semilogy(sizes,Err_AXA(m,:),markerList{m},'LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|AXA - A\|_F$','Interpreter','latex');
title('AXA Consistency Error','Interpreter','latex');
legend(methods_new,'Location','northeast');

fig6 = figure('Units','normalized','Position',[0.50 0.40 0.45 0.35]); hold on; grid on;
for m=1:numM, semilogy(sizes,Err_XAX(m,:),markerList{m},'LineWidth',1.5); end
xlabel('Matrix size $n$','Interpreter','latex');
ylabel('$\|XAX - X\|_F$','Interpreter','latex');
title('XAX Consistency Error','Interpreter','latex');
legend(methods_new,'Location','northeast');

% ----------------------
% Add inset (left-side) for CPU plot only (one tidy arrow)
% Zoom to highlight 3 methods: indices 1:3 (SAI, RAPID, HPI19)
% ----------------------
cpu_zoom_x = [1100 1200];
% choose zoom y-range automatically from the 3 methods at those x points
idx_zoom = find(sizes >= cpu_zoom_x(1) & sizes <= cpu_zoom_x(2));
if isempty(idx_zoom)
    idx_zoom = max(1,length(sizes)-4):length(sizes);
end
x_zoom = sizes(idx_zoom);

% compute y-range from first three methods in the zoom range
methods_to_show = [1,2,3]; % SAI,RAPID,HPI19 after reorder
yvals = Times(methods_to_show, idx_zoom);
ymin = min(yvals(:)); ymax = max(yvals(:));
pad = 0.08 * max(1, (ymax - ymin));
cpu_zoom_y_auto = [max(0,ymin - pad), ymax + pad];

% determine inset position just below the legend
drawnow; % ensure legend position property is available
legPos = leg.Position;   % legend position in normalized figure units
axPos = get(ax1,'Position'); % axes position (figure normalized)
% compute inset left = ax left + legend left inside axes
inset_w = 0.18; inset_h = 0.18;
inset_left = axPos(1) + max(0, legPos(1)); % place under legend (same left)
inset_bottom = axPos(2) + legPos(2) - (inset_h + 0.02); % slightly below legend
% defensive: ensure inset_bottom within [0,1]
inset_bottom = max(0.05, inset_bottom);
inset_pos_cpu = [inset_left, inset_bottom, inset_w, inset_h];

% draw dashed rectangle (main axes) showing zoom area but tight to the 3-method values
rect_y = cpu_zoom_y_auto(1);
rect_h = cpu_zoom_y_auto(2) - cpu_zoom_y_auto(1);
rectangle(ax1,'Position',[cpu_zoom_x(1), rect_y, diff(cpu_zoom_x), rect_h], ...
    'EdgeColor',[0 0 0], 'LineStyle','--', 'LineWidth',1.0);

% create inset axes
ax_in1 = axes('Position', inset_pos_cpu);
box(ax_in1,'on'); hold(ax_in1,'on'); grid(ax_in1,'on');
colors = get(gca,'ColorOrder');
for k = 1:numel(methods_to_show)
    m = methods_to_show(k);
    plot(ax_in1, x_zoom, Times(m, idx_zoom), '-o', 'LineWidth',1.2, 'MarkerSize',5);
end
xlim(ax_in1, cpu_zoom_x);
ylim(ax_in1, cpu_zoom_y_auto);
title(ax_in1,'Zoom (SAI,RAPID,HPI19)','Interpreter','latex','FontSize',8);
set(ax_in1,'FontSize',8);

% draw a single small arrow from the center-top of the rectangle to the inset
% pick source point (mid of rectangle top)
srcX = cpu_zoom_x(1) + diff(cpu_zoom_x)/2;
srcY = rect_y + rect_h; % top
src_fig = ds2nfu(ax1, srcX, srcY);
% pick target point inside inset (a bit right of inset left, near top)
tgt_fig = [inset_pos_cpu(1) + 0.12*inset_pos_cpu(3), inset_pos_cpu(2) + 0.85*inset_pos_cpu(4)];
annotation('arrow', [src_fig(1) tgt_fig(1)], [src_fig(2) tgt_fig(2)], 'Color','k', 'LineWidth',1.0, 'HeadLength',8, 'HeadWidth',8);

% ----------------------
% Add small left-side insets for error figures (optional; keep default positions)
% Here we add small left insets showing last zoom points (same x range) for visibility
err_zoom_x = cpu_zoom_x;
err_zoom_y = [0.1 0.5];
inset_err_pos = [0.12 0.10 0.18 0.16];
add_error_inset(fig3, Err_AXherm, inset_err_pos, err_zoom_x, err_zoom_y, idx_zoom, x_zoom, markerList);
add_error_inset(fig4, Err_XAherm, inset_err_pos, err_zoom_x, err_zoom_y, idx_zoom, x_zoom, markerList);
add_error_inset(fig5, Err_AXA, inset_err_pos, err_zoom_x, err_zoom_y, idx_zoom, x_zoom, markerList);
add_error_inset(fig6, Err_XAX, inset_err_pos, err_zoom_x, err_zoom_y, idx_zoom, x_zoom, markerList);

% ----------------------
% Print selected summary table (pick a size)
% ----------------------
pickSize = 300;
idx = find(sizes == pickSize,1);
if isempty(idx)
    warning('pickSize not in sizes - showing last size instead.');
    idx = numel(sizes);
end

fprintf('\n===== Results for matrix size %d =====\n', sizes(idx));
fprintf('%-15s %-10s %-8s %-12s %-12s %-12s %-12s\n', ...
    'Method','Time(s)','Iters','||AX-AX^*||','||(XA)^*-XA||','||AXA-A||','||XAX-X||');

for m = 1:numM
    fprintf('%-15s %-10.3e %-8d %-12.2e %-12.2e %-12.2e %-12.2e\n', ...
        methods_new{m}, Times(m,idx), Iters(m,idx), ...
        Err_AXherm(m,idx), Err_XAherm(m,idx), Err_AXA(m,idx), Err_XAX(m,idx));
end

% ----------------------
% Helper: add error inset (explicit inputs)
% ----------------------
function add_error_inset(figHandle, Ydata, insetPos, z_x, z_y, idx_z, xz, markerListLocal)
    figure(figHandle);
    axMain = gca;
    hold(axMain,'on');
    rectangle(axMain,'Position',[z_x(1), z_y(1), diff(z_x), diff(z_y)], ...
        'EdgeColor',[0 0 0], 'LineStyle','--', 'LineWidth',1);
    axIns = axes('Position', insetPos );
    box(axIns,'on'); hold(axIns,'on'); grid(axIns,'on');
    for mm=1:size(Ydata,1)
        y_zoom_vals = Ydata(mm, idx_z);
        plot(axIns, xz, y_zoom_vals, markerListLocal{mm}, 'LineWidth',1, 'MarkerSize',4);
    end
    xlim(axIns, z_x);
    ylim(axIns, z_y);
    set(axIns,'FontSize',8);
    title(axIns,'Zoom 1100--1200','Interpreter','latex','FontSize',8);
    % single small arrow
    src_fig = ds2nfu(axMain, z_x(1), z_y(1));
    tgt_fig = [insetPos(1)+0.08, insetPos(2)+insetPos(4)*0.8];
    annotation('arrow', [src_fig(1) tgt_fig(1)], [src_fig(2) tgt_fig(2)], 'Color','k','LineWidth',0.8);
end

% ----------------------
% Utility: data-space to normalized figure units
% ----------------------
function nf = ds2nfu(axHandle, xdata, ydata)
    axUnits = get(axHandle,'Units');
    set(axHandle,'Units','normalized');
    axPos = get(axHandle,'Position'); % [left bottom width height]
    xLim = get(axHandle,'XLim'); yLim = get(axHandle,'YLim');
    xn = (xdata - xLim(1)) / (xLim(2) - xLim(1));
    yn = (ydata - yLim(1)) / (yLim(2) - yLim(1));
    xf = axPos(1) + xn * axPos(3);
    yf = axPos(2) + yn * axPos(4);
    nf = [xf, yf];
    set(axHandle,'Units',axUnits);
end

% End of script
