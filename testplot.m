%% quick_plot_psnr_ssim_no_renderer.m
% PSNR/SSIM plotting with paper-friendly LaTeX fonts, no explicit renderer settings.
clear; close all; clc;

%% === Global Style Defaults (paper style) ===
fs = 24;          % base font size for axes/labels
lineW = 3.5;      % line width
markerSz = 10;    % marker size

set(groot, 'defaultTextInterpreter','latex', ...
           'defaultAxesTickLabelInterpreter','latex', ...
           'defaultLegendInterpreter','latex', ...
           'defaultAxesFontSize', fs, ...
           'defaultAxesLineWidth', 3.0, ...
           'defaultLineLineWidth', lineW, ...
           'defaultLineMarkerSize', markerSz, ...
           'defaultFigureColor','w', ...
           'defaultFigurePaperPositionMode','auto');
set(groot,'defaultAxesLabelFontSizeMultiplier',1.5);
set(groot,'defaultAxesTitleFontSizeMultiplier',1);

outdir = fullfile(pwd,'image','resultcomparison');
if ~exist(outdir,'dir'), mkdir(outdir); end

%% ---- pick files (multi-select) ----
[f,p] = uigetfile('*.mat','Select .mat files','MultiSelect','on');
if isequal(f,0), disp('No file selected.'); return; end
if ischar(f), files = {fullfile(p,f)}; else files = fullfile(p,f); end

%% ---- loop over files ----
for k = 1:numel(files)
    S = load(files{k});
    % find PSNR & SSIM fields
    if isfield(S,'psnr_history'), psnr = S.psnr_history(:);
    elseif isfield(S,'PSNR'), psnr = S.PSNR(:);
    else error('No PSNR found in %s', files{k}); end

    if isfield(S,'ssim_history'), ssim = S.ssim_history(:);
    elseif isfield(S,'SSIM'), ssim = S.SSIM(:);
    else error('No SSIM found in %s', files{k}); end

    n = min(numel(psnr), numel(ssim));
    psnr = psnr(1:n); ssim = ssim(1:n);
    it = (1:n)';

    %% ---- plotting ----
    cPSNR = [0 0.45 0.74];   % blue
    cSSIM = [0.6 0 0.13];    % maroon

    % Create figure (no explicit Renderer argument)
    fig = figure('Color','w','Units','pixels','Position',[100 100 1200 520]);
    ax = axes('Parent',fig,'Position',[0.15 0.18 0.70 0.72],'Box','on');
    grid(ax,'on'); hold(ax,'on');

    % left axis: PSNR
    yyaxis(ax,'left');
    h1 = plot(it, psnr, '-o', 'Color', cPSNR, 'LineWidth', lineW, 'MarkerSize', markerSz);
    ylabel(ax, 'PSNR (dB)', 'Interpreter','latex', 'FontSize', fs*1.2);

    % right axis: SSIM
    yyaxis(ax,'right');
    h2 = plot(it, ssim, '-s', 'Color', cSSIM, 'LineWidth', lineW, 'MarkerSize', markerSz);
    ylabel(ax, 'SSIM', 'Interpreter','latex', 'FontSize', fs*1.2);

    % x-label
    xlabel(ax, 'Iteration', 'Interpreter','latex', 'FontSize', fs*1.2);

    % legend (font size slightly smaller)
    legend(ax, [h1 h2], {'PSNR','SSIM'}, 'Location','best', ...
           'Interpreter','latex', 'FontSize', round(fs*1.0));

    %% ---- colored border (decorative) ----
    pos = ax.Position;
    annotation('line',[pos(1) pos(1)+pos(3)],[pos(2) pos(2)],'Color','k','LineWidth',1.6);
    annotation('line',[pos(1) pos(1)+pos(3)],[pos(2)+pos(4) pos(2)+pos(4)],'Color','k','LineWidth',1.6);
    annotation('line',[pos(1) pos(1)],[pos(2) pos(2)+pos(4)],'Color',cPSNR,'LineWidth',1.6);
    annotation('line',[pos(1)+pos(3) pos(1)+pos(3)],[pos(2) pos(2)+pos(4)],'Color',cSSIM,'LineWidth',1.6);

    %% ---- save (no renderer switching) ----
    [~,name] = fileparts(files{k});
    outbase = fullfile(outdir, [name '_psnrssim']);

    % Try exportgraphics if available (preferred). We do not change renderer.
    if exist('exportgraphics','file')
        try
            exportgraphics(fig, [outbase '.eps'], 'ContentType','vector');
        catch ME
            warning('exportgraphics EPS failed for %s: %s', outbase, ME.message);
        end
        try
            exportgraphics(fig, [outbase '.pdf'], 'ContentType','vector');
        catch ME
            warning('exportgraphics PDF failed for %s: %s', outbase, ME.message);
        end
    else
        % fallback: use print for EPS/PDF if exportgraphics not available
        try
            print(fig, [outbase '.eps'], '-depsc2', '-r300');
        catch ME
            warning('print EPS failed for %s: %s', outbase, ME.message);
        end
        try
            print(fig, [outbase '.pdf'], '-dpdf', '-r300');
        catch ME
            warning('print PDF failed for %s: %s', outbase, ME.message);
        end
    end

    % Always save MATLAB .fig so you can reopen/edit later
    try
        savefig(fig, [outbase '.fig']);
    catch ME
        warning('savefig failed for %s: %s', outbase, ME.message);
    end

    close(fig);
    fprintf('Saved: %s.{eps,pdf,fig}\n', outbase);
end

fprintf('Finished. Results in: %s\n', outdir);
