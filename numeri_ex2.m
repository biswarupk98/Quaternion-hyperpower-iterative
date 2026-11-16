%% numeri_ex2.m
% Robust test script for quaternion pseudoinverses (ready-to-run)
% Methods order: {'SAI','RAPID','HPI19','QSVD','Newton-Schulz','HONSchulz','HybridRSPNS'}

clear; clc; close all;

% --------------------
% Parameters & plotting defaults
% --------------------
rng(1);                      % reproducible

methods = {'SAI','RAPID','HPI19','QSVD','Newton-Schulz','HONSchulz','HybridRSPNS'};
numM = numel(methods);
tolerance = 1e-10;

% Plotting base sizes (tweak these if you want even larger text)
baseTitleFS  = 24;   % title font size
baseLabelFS  = 20;   % x/y label font size
baseTickFS   = 16;   % tick labels font size
baseLegendFS = 16;   % legend font size
baseLineW    = 1.8;  % plot line width
baseMarkerS  = 10;   % marker size

% Apply safe global defaults for clearer LaTeX text (supported props)
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex'); % usually supported
set(groot, 'defaultAxesFontSize', baseTickFS);
set(groot, 'defaultLineLineWidth', baseLineW);
set(groot, 'defaultLineMarkerSize', baseMarkerS);

% --------------------
% Build testCases robustly (avoid inline cell literal issues)
% --------------------
testCases = cell(3,1);
testCases{1} = struct('n',1000,'m',500,'name','Overdetermined (1000x500)');
testCases{2} = struct('n',500,'m',1000,'name','Underdetermined (500x1000)');
% If you want the 500x500 rank-deficient case uncomment next line:
% testCases{3} = struct('n',500,'m',500,'name','Rank-deficient (500x500)');
testCases{3} = struct('n',1000,'m',1000,'name','Rank-deficient (1000x1000)');

numCases = numel(testCases);

% Preallocate results (methods x cases)
Times      = NaN(numM,numCases);
Iters      = NaN(numM,numCases);
Err_AXherm = NaN(numM,numCases);
Err_XAherm = NaN(numM,numCases);
Err_AXA     = NaN(numM,numCases);
Err_XAX     = NaN(numM,numCases);

% Rank fraction for rank-deficient case (fraction of min(n,m))
rank_fraction = 0.35;

% Precompute readable case names
caseNames = cellfun(@(s) s.name, testCases, 'UniformOutput', false);

% --------------------
% Main loops
% --------------------
for c = 1:numCases
    tc = testCases{c};
    n = tc.n; m = tc.m;
    fprintf('\n===== Case: %s (n=%d, m=%d) =====\n', tc.name, n, m);

    % --- Generate quaternion components A0,A1,A2,A3 ---
    if contains(tc.name,'Rank-deficient')
        r = max(1,floor(rank_fraction * min(n,m)));
        U0 = randn(n,r); V0 = randn(m,r);
        U1 = randn(n,r); V1 = randn(m,r);
        U2 = randn(n,r); V2 = randn(m,r);
        U3 = randn(n,r); V3 = randn(m,r);
        A0 = U0*V0'; A1 = U1*V1'; A2 = U2*V2'; A3 = U3*V3';
        perturb = 1e-12;
        A0 = A0 + perturb*randn(n,m);
        A1 = A1 + perturb*randn(n,m);
        A2 = A2 + perturb*randn(n,m);
        A3 = A3 + perturb*randn(n,m);
    else
        A0 = rand(n,m); A1 = rand(n,m);
        A2 = rand(n,m); A3 = rand(n,m);
    end

    % --- Build quaternion matrix (assumes quaternion constructor supports rectangular) ---
    A = quaternion(A0,A1,A2,A3);

    % --- Compute scaling alpha via complex embedding (rectangular-safe) ---
    A11 = complex(A0,A1); A12 = complex(A2,A3);
    CA = [A11, A12; -conj(A12), conj(A11)]; % size (2n) x (2m)
    try
        svals = svd(CA,'econ');
        alp = 1/(max(svals)^2);
    catch
        alp = 1/(norm(CA,'fro')^2 + eps);
    end

    % --- Loop over methods in the requested order ---
    for mi = 1:numM
        mname = methods{mi};
        try
            switch mname
                case 'SAI'
                    tstart = tic;
                    [X,it] = quaternion_SAI1(A,alp,tolerance);
                    Times(mi,c) = toc(tstart); Iters(mi,c) = it;

                case 'RAPID'
                    tstart = tic;
                    [X,it] = quaternion_RAPID1(A,5,alp,tolerance);
                    Times(mi,c) = toc(tstart); Iters(mi,c) = it;

                case 'HPI19'
                    X0 = alp * A';
                    tstart = tic;
                    [X,it] = quaternion_Hyper19(A,X0,tolerance);
                    Times(mi,c) = toc(tstart); Iters(mi,c) = it;

                case 'QSVD'
                    tstart = tic;
                    X = pinv(A);               % pseudoinverse (m x n)
                    Times(mi,c) = toc(tstart); Iters(mi,c) = 1;

                case 'Newton-Schulz'
                    tstart = tic;
                    [X,it,~] = quaternion_NewtonSchulz(A,tolerance,200,0.5,alp);
                    Times(mi,c) = toc(tstart); Iters(mi,c) = it;

                case 'HONSchulz'
                    tstart = tic;
                    [X,it,~] = quaternion_HONSchulz(A,tolerance,200);
                    Times(mi,c) = toc(tstart); Iters(mi,c) = it;

                case 'HybridRSPNS'
                    tstart = tic;
                    [X,info] = quaternion_HybridRSPNS(A,8,4,5,tolerance,200);
                    Times(mi,c) = toc(tstart);
                    if isstruct(info) && isfield(info,'iters')
                        Iters(mi,c) = info.iters;
                    else
                        Iters(mi,c) = NaN;
                    end

                otherwise
                    error('Unknown method: %s', mname);
            end

            % --- Compute residuals if X exists ---
            if exist('X','var') && ~isempty(X)
                Err_AXherm(mi,c) = norm((A*X)'-A*X,'fro');
                Err_XAherm(mi,c) = norm((X*A)'-X*A,'fro');
                Err_AXA(mi,c)    = norm(A*X*A - A,'fro');
                Err_XAX(mi,c)    = norm(X*A*X - X,'fro');
                clear X;
            end

        catch ME
            % keep NaNs and warn
            warning('Method %s failed for case %s: %s', mname, tc.name, ME.message);
            clear X;
        end
    end % methods loop

end % cases loop

% --------------------
% Print summarized results
% --------------------
fprintf('\n\n===== Summary table (methods x cases) =====\n');
headerLine = 'Method           | Case                                      | Time(s)      | Iters | ||AXA-A||_F   | ||XAX-X||_F   | ||(AX)^* - AX||_F | ||(XA)^* - XA||_F\n';
fprintf('%s', headerLine);
fprintf('%s\n', repmat('-',1,length(headerLine)));

for mi = 1:numM
    for c = 1:numCases
        fprintf('%-16s | %-40s | %-12.3e | %-5g | %-12.2e | %-12.2e | %-16.2e | %-16.2e\n', ...
            methods{mi}, caseNames{c}, Times(mi,c), Iters(mi,c), ...
            Err_AXA(mi,c), Err_XAX(mi,c), Err_AXherm(mi,c), Err_XAherm(mi,c));
    end
end

% --------------------
% Plots (big, clear LaTeX text)
% --------------------

% 1) CPU time (grouped bar)
f1 = figure('Name','CPU time per method & case','Units','normalized','Position',[0.08 0.12 0.72 0.60]);
bar(Times','grouped');
ax = gca;
ax.FontSize = baseTickFS;
ax.LineWidth = 1.2;
xticklabels(caseNames);
xtickangle(30);
ylabel('CPU time (s)','Interpreter','latex','FontSize',baseLabelFS);
title('CPU time across cases','Interpreter','latex','FontSize',baseTitleFS);
legend(methods,'Interpreter','latex','FontSize',baseLegendFS,'Location','northeastoutside','Box','off');
grid on;

% 2) AXA error (log scale)
f2 = figure('Name','AXA error across cases','Units','normalized','Position',[0.12 0.12 0.62 0.50]);
hold on;
for mi = 1:numM
    plot(1:numCases, Err_AXA(mi,:), '-o', 'LineWidth', baseLineW, 'MarkerSize', baseMarkerS);
end
hold off;
ax = gca;
ax.FontSize = baseTickFS;
ax.LineWidth = 1.2;
xticks(1:numCases); xticklabels(caseNames); xtickangle(30);
ylabel('$\|AXA - A\|_F$','Interpreter','latex','FontSize',baseLabelFS);
title('AXA error across cases','Interpreter','latex','FontSize',baseTitleFS);
legend(methods,'Interpreter','latex','FontSize',baseLegendFS,'Location','northeastoutside','Box','off');
set(gca,'YScale','log'); grid on;

% 3) XAX error (log scale)
f3 = figure('Name','XAX error across cases','Units','normalized','Position',[0.14 0.14 0.62 0.50]);
hold on;
for mi = 1:numM
    plot(1:numCases, Err_XAX(mi,:), '-s', 'LineWidth', baseLineW, 'MarkerSize', baseMarkerS);
end
hold off;
ax = gca;
ax.FontSize = baseTickFS;
ax.LineWidth = 1.2;
xticks(1:numCases); xticklabels(caseNames); xtickangle(30);
ylabel('$\|XAX - X\|_F$','Interpreter','latex','FontSize',baseLabelFS);
title('XAX error across cases','Interpreter','latex','FontSize',baseTitleFS);
legend(methods,'Interpreter','latex','FontSize',baseLegendFS,'Location','northeastoutside','Box','off');
set(gca,'YScale','log'); grid on;

fprintf('\nDone.\n');
