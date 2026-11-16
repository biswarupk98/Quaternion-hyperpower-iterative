% compare_q_methods_custom_labels.m
% Sweep using embedding-based alp for initializer; compare factorized vs pure NS.
clear; close all; clc;
rng(0);

%% sweep sizes (you used 10:5:50 in your snippet; change if needed)
sizes = 10:5:50;
numN = numel(sizes);

%% params
tol = 1e-10;
max_outer = 500;

% method mapping (internal slots):
% 2 = SAI, 3 = NS p=19, 4 = Hyper19, 5 = NS p=10
display_order = [4, 2, 3, 5]; % output order: Hyper19, SAI1, NS19, NS10
display_names = {'QHPI19','QSAI','QHONSchultz(order 19)','QHONSchultz(order 10)'};

% allocate arrays
Times      = NaN(5, numN);
Iters      = NaN(5, numN);
Err_AXherm = NaN(5, numN);
Err_XAherm = NaN(5, numN);
Err_AXA    = NaN(5, numN);
Err_XAX    = NaN(5, numN);
Rel_AXA    = NaN(5, numN);
Rel_XAX    = NaN(5, numN);

fprintf('Sweep using embedding-based alp for initializer; sizes %d:%d:%d\n', sizes(1), sizes(2)-sizes(1), sizes(end));

for ss = 1:numN
    n = sizes(ss);
    fprintf('\n--- size n = %d (%d/%d) ---\n', n, ss, numN);

    % build random quaternion matrix A
    A0 = randn(n,n); A1 = randn(n,n);
    A2 = randn(n,n); A3 = randn(n,n);
    A = quaternion(A0, A1, A2, A3);

    % ------- compute alp via complex embedding (as requested) -------
    A11 = complex(A0, A1);
    A12 = complex(A2, A3);
    CA = [A11, A12; -conj(A12), conj(A11)];
    svals = svd(CA, 'econ');
    alp = 1/(max(svals)^2 + eps);
    % ----------------------------------------------------------------

    % initial guesses
    X0_conjAH = alp * (conj(A).');   % used for SAI & NS runs
    X0_Atrans  = alp * A';           % used for Hyper19 per your snippet

    % exact norm for normalization
    normA = norm(A,'fro');

    %% quaternion_Hyper19 (factorized) -> stored in slot 4
    if exist('quaternion_Hyper19','file') == 2
        try
            t0 = tic;
            try
                [Xh, it_h] = quaternion_Hyper19(A, X0_Atrans, tol);
            catch
                [Xh, it_h] = quaternion_Hyper19(A, X0_Atrans);
            end
            Times(4, ss) = toc(t0);
            Iters(4, ss) = it_h;
            Err_AXherm(4, ss) = norm((A*Xh)'-A*Xh,'fro');
            Err_XAherm(4, ss) = norm((Xh*A)'-Xh*A,'fro');
            Err_AXA(4, ss) = norm(A*Xh*A - A,'fro');
            Err_XAX(4, ss) = norm(Xh*A*Xh - Xh,'fro');
            Rel_AXA(4, ss) = Err_AXA(4, ss) / max(eps, normA);
            Rel_XAX(4, ss) = Err_XAX(4, ss) / max(eps, norm(Xh,'fro'));
            fprintf('Hyper19 (slot4): time=%.3fs, iters=%d, rel_AXA=%.3e\n', Times(4,ss), Iters(4,ss), Rel_AXA(4,ss));
        catch ME
            warning('quaternion_Hyper19 failed at n=%d: %s', n, ME.message);
        end
    else
        fprintf('quaternion_Hyper19 not found -> skipped for n=%d\n', n);
    end

    %% quaternion_SAI1 (factorized) -> stored in slot 2
    if exist('quaternion_SAI1','file') == 2
        try
            t0 = tic;
            try
                [Xs, it_s] = quaternion_SAI1(A, alp, tol);
            catch
                try
                    [Xs, it_s] = quaternion_SAI1(A, X0_conjAH, tol);
                catch
                    [Xs, it_s] = quaternion_SAI1(A, X0_conjAH);
                end
            end
            Times(2, ss) = toc(t0);
            Iters(2, ss) = it_s;
            Err_AXherm(2, ss) = norm((A*Xs)'-A*Xs,'fro');
            Err_XAherm(2, ss) = norm((Xs*A)'-Xs*A,'fro');
            Err_AXA(2, ss) = norm(A*Xs*A - A,'fro');
            Err_XAX(2, ss) = norm(Xs*A*Xs - Xs,'fro');
            Rel_AXA(2, ss) = Err_AXA(2, ss) / max(eps, normA);
            Rel_XAX(2, ss) = Err_XAX(2, ss) / max(eps, norm(Xs,'fro'));
            fprintf('SAI1 (slot2): time=%.3fs, iters=%d, rel_AXA=%.3e\n', Times(2,ss), Iters(2,ss), Rel_AXA(2,ss));
        catch ME
            warning('quaternion_SAI1 failed at n=%d: %s', n, ME.message);
        end
    else
        fprintf('quaternion_SAI1 not found -> skipped for n=%d\n', n);
    end

    %% NS left p = 19 (no factorization) -> stored in slot 3
    try
        X = X0_conjAH;
        t0 = tic;
        it_ns = max_outer;
        for k = 1:max_outer
            Xnext = ns_hyperpower_left(A, X, 19);    % quaternion-based
            delta = norm(Xnext - X, 'fro');
            if delta < tol
                X = Xnext; it_ns = k; break;
            end
            X = Xnext;
        end
        Times(3, ss) = toc(t0);
        Iters(3, ss) = it_ns;
        Err_AXherm(3, ss) = norm((A*X)'-A*X,'fro');
        Err_XAherm(3, ss) = norm((X*A)'-X*A,'fro');
        Err_AXA(3, ss) = norm(A*X*A - A,'fro');
        Err_XAX(3, ss) = norm(X*A*X - X,'fro');
        Rel_AXA(3, ss) = Err_AXA(3, ss) / max(eps, normA);
        Rel_XAX(3, ss) = Err_XAX(3, ss) / max(eps, norm(X,'fro'));
        fprintf('NS p=19 (slot3): time=%.3fs, outer_iters=%d, rel_AXA=%.3e\n', Times(3,ss), Iters(3,ss), Rel_AXA(3,ss));
    catch ME
        warning('ns_hyperpower_left p=19 failed at n=%d: %s', n, ME.message);
    end

    %% NS left p = 10 (no factorization) -> stored in slot 5
    try
        X = X0_conjAH;
        t0 = tic;
        it_ns = max_outer;
        for k = 1:max_outer
            Xnext = ns_hyperpower_left(A, X, 10);
            delta = norm(Xnext - X, 'fro');
            if delta < tol
                X = Xnext; it_ns = k; break;
            end
            X = Xnext;
        end
        Times(5, ss) = toc(t0);
        Iters(5, ss) = it_ns;
        Err_AXherm(5, ss) = norm((A*X)'-A*X,'fro');
        Err_XAherm(5, ss) = norm((X*A)'-X*A,'fro');
        Err_AXA(5, ss) = norm(A*X*A - A,'fro');
        Err_XAX(5, ss) = norm(X*A*X - X,'fro');
        Rel_AXA(5, ss) = Err_AXA(5, ss) / max(eps, normA);
        Rel_XAX(5, ss) = Err_XAX(5, ss) / max(eps, norm(X,'fro'));
        fprintf('NS p=10 (slot5): time=%.3fs, outer_iters=%d, rel_AXA=%.3e\n', Times(5,ss), Iters(5,ss), Rel_AXA(5,ss));
    catch ME
        warning('ns_hyperpower_left p=10 failed at n=%d: %s', n, ME.message);
    end
end

%% Prepare display arrays (order: Hyper19, SAI1, NS p=19, NS p=10)
idx = display_order;
Times_disp   = Times(idx, :);
Rel_AXA_disp = Rel_AXA(idx, :);
Rel_XAX_disp = Rel_XAX(idx, :);

%% Plot CPU time (paper-friendly style)
fs = 24; lineW = 3.5;
set(groot, 'defaultTextInterpreter','latex', 'defaultAxesTickLabelInterpreter','latex', ...
    'defaultLegendInterpreter','latex', 'defaultAxesFontSize', fs, 'defaultLineLineWidth', lineW);

figure('Name','CPU times','Color','w');
colors = lines(numel(idx));
markers = {'-o','-s','-^','-d'};
hold on; grid on;
for i = 1:numel(idx)
    plot(sizes, Times_disp(i,:), markers{i}, 'LineWidth', 1.6, 'Color', colors(i,:), 'DisplayName', display_names{i});
end
xlabel('Matrix size $n$','Interpreter','latex'); ylabel('CPU time (s)','Interpreter','latex');
legend(display_names,'Interpreter','latex','Location','northwest'); title('Timing comparison','Interpreter','latex');

%% Plot Rel ||AXA-A|| / ||A||
figure('Name','Relative AXA error','Color','w');
hold on; grid on; set(gca,'YScale','log');
for i = 1:numel(idx)
    semilogy(sizes, Rel_AXA_disp(i,:), markers{i}, 'LineWidth', 1.6, 'Color', colors(i,:), 'DisplayName', display_names{i});
end
xlabel('Matrix size $n$','Interpreter','latex'); ylabel('$\|AXA-A\|_F / \|A\|_F$','Interpreter','latex');
legend(display_names,'Interpreter','latex','Location','northeast'); title('Relative AXA error','Interpreter','latex');

%% Plot Rel ||XAX-X|| / ||X||
figure('Name','Relative XAX error','Color','w');
hold on; grid on; set(gca,'YScale','log');
for i = 1:numel(idx)
    semilogy(sizes, Rel_XAX_disp(i,:), markers{i}, 'LineWidth', 1.6, 'Color', colors(i,:), 'DisplayName', display_names{i});
end
xlabel('Matrix size $n$','Interpreter','latex'); ylabel('$\|XAX-X\|_F / \|X\|_F$','Interpreter','latex');
legend(display_names,'Interpreter','latex','Location','northeast'); title('Relative XAX error','Interpreter','latex');

%% Print compact table of timings (display order)
fprintf('\nSummary table (time(s)) for methods: %s\n', strjoin(display_names, ', '));
fprintf('%-28s', 'Method');
for ss=1:numN, fprintf('%10d', sizes(ss)); end
fprintf('\n');
for i=1:numel(idx)
    fprintf('%-28s', display_names{i});
    for ss=1:numN
        tv = Times_disp(i,ss);
        if isnan(tv), fprintf('%10s','NA'); else fprintf('%10.3f', tv); end
    end
    fprintf('\n');
end

fprintf('\nDone.\n');
