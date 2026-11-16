%% ---------------- Paper-style timings bar chart (LEFT) ----------------
% set paper-friendly LaTeX style & fonts (from user)
paperFontName = 'Times New Roman';
fs = 30;           % base font size
lineW = 1.5;
markerSz = 6;

set(groot, 'defaultTextInterpreter','latex', ...
           'defaultAxesTickLabelInterpreter','latex', ...
           'defaultLegendInterpreter','latex', ...
           'defaultAxesFontName', paperFontName, ...
           'defaultTextFontName', paperFontName, ...
           'defaultAxesFontSize', fs, ...
           'defaultAxesLineWidth', 1.2, ...
           'defaultLineLineWidth', lineW, ...
           'defaultLineMarkerSize', markerSz, ...
           'defaultFigureColor','w', ...
           'defaultFigurePaperPositionMode','auto');

set(groot,'defaultAxesLabelFontSizeMultiplier',1);
set(groot,'defaultAxesTitleFontSizeMultiplier',1);

% build consistent label/time vectors (preserve method order)
methodNames = {'Hyper19 (fact)', 'NS left p=19', 'SAI1 (fact)', 'NS left p=10'};
methodTimes = nan(1,4);

if ~isempty(X_hyper19) && ~isnan(t_hyper19)
    methodTimes(1) = t_hyper19;
end
methodTimes(2) = t_ns19_left;  % always present from your run_left_iter
if ~isempty(X_sai1) && ~isnan(t_sai1)
    methodTimes(3) = t_sai1;
end
methodTimes(4) = t_ns10_left;  % always present from your run_left_iter

% For plotting: replace NaNs with 0 so bar positions remain; mark missing with hatch
plotTimes = methodTimes;
plotTimes(isnan(plotTimes)) = 0;

figure('Name','Timings (LEFT) - Paper style','NumberTitle','off','Units','normalized','Position',[0.25 0.35 0.5 0.45]);
hBar = bar(plotTimes,'BarWidth',0.6);
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.XTick = 1:numel(methodNames);
ax.XTickLabel = methodNames;
ax.XTickLabelRotation = 25;
ylabel('Wall-clock time (s)','Interpreter','latex','FontSize',fs);
title('Timing comparison — LEFT hyperpower / NS / SAI1','Interpreter','latex','FontSize',fs);

% visually de-emphasize missing methods (hatch + light face)
for ii = 1:length(plotTimes)
    if methodTimes(ii) == 0  % missing
        hBar.FaceColor = 'flat';                       % allow coloring per-bar
        hBar.CData(ii,:) = [0.85 0.85 0.85];          % light gray
        % add text "skipped"
        text(ii,  max(plotTimes)*0.05, 'skipped', 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom','Interpreter','latex','FontSize',round(fs*0.6));
    else
        % keep default color for present methods; optionally highlight fastest
        hBar.CData(ii,:) = [0.2 0.4470 0.7410];  % default MATLAB blue-ish (one color for all)
    end
end

% annotate each bar with numeric timing (if present)
yl = ylim;
for ii = 1:numel(methodTimes)
    if ~isnan(methodTimes(ii))
        txt = sprintf('%.3f s', methodTimes(ii));
        text(ii, plotTimes(ii) + (yl(2)-yl(1))*0.02, txt, 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom','Interpreter','latex','FontSize',round(fs*0.6));
    end
end

grid on;
box on;
set(ax,'FontName',paperFontName,'FontSize',fs,'LineWidth',1.2);

% improve readability for paper: enlarge margins and tick lengths
ax.TickLength = [0.02 0.02];
ax.Position = [0.11 0.15 0.86 0.78];

% Save high-resolution outputs suitable for papers
outdir = fullfile(pwd,'figures');
if ~exist(outdir,'dir'), mkdir(outdir); end
figname = fullfile(outdir,'timings_left_paper');

% Use exportgraphics (modern MATLAB) for vector PDF/EPS and high-res PNG
try
    exportgraphics(gcf,[figname '.pdf'],'ContentType','vector');
    exportgraphics(gcf,[figname '.eps'],'ContentType','vector');
    exportgraphics(gcf,[figname '.png'],'Resolution',300);
catch
    % Fallback to print if exportgraphics not available
    print(gcf,[figname '.pdf'],'-dpdf','-r300');
    print(gcf,[figname '.eps'],'-depsc2','-r300');
    print(gcf,[figname '.png'],'-dpng','-r300');
end

fprintf('Paper-style timings figure saved to %s (pdf, eps, png).\n', outdir);
