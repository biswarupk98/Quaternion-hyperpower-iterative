function plot_psnr_ssim_simple(matfile)
% PSNR (blue, left) / SSIM (maroon, right) with colored box borders
% with extra margin so nothing is cropped in PDF/LaTeX
% Now uses bold serif font for labels, ticks, and legend.

S = load(matfile);
psnr = S.psnr_history(:);
ssim = S.ssim_history(:);
it   = 1:numel(psnr);

% Style
fs_label = 30; fs_tick = 24; fs_leg = 24; fnt = 'Times'; % serif font
cPSNR = [0 0.45 0.74];   % blue
cSSIM = [0.6 0 0.13];    % maroon

% Bigger canvas (extra wide for SSIM side)
fig = figure('Units','pixels','Position',[100 100 1400 600],'Color','w','Renderer','painters');

% Shift axes left & reduce width → leaves room on right for SSIM labels
left = 0.15; bottom = 0.18; width = 0.70; height = 0.72;
ax = axes('Position',[left bottom width height]); hold(ax,'on'); grid(ax,'on');

% PSNR (left)
yyaxis(ax,'left');
plot(it, psnr,'-o','Color',cPSNR,'LineWidth',1.6,'MarkerSize',6);
ylabel('PSNR (dB)','FontSize',fs_label,'FontName',fnt,'FontWeight','bold','Color',cPSNR);
ax.YAxis(1).Color = cPSNR;

% SSIM (right)
yyaxis(ax,'right');
plot(it, ssim,'-s','Color',cSSIM,'LineWidth',1.6,'MarkerSize',6);
ylabel('SSIM','FontSize',fs_label,'FontName',fnt,'FontWeight','bold','Color',cSSIM);
ax.YAxis(2).Color = cSSIM;

% X axis
xlabel('Iteration','FontSize',fs_label,'FontName',fnt,'FontWeight','bold','Color','k');
set(ax,'FontSize',fs_tick,'FontName',fnt,'FontWeight','bold','XColor','k');

% Legend boxed (bold text)
hLeg = legend({'PSNR','SSIM'}, 'Location','best', ...
    'FontSize', fs_leg, 'FontName', fnt, 'FontWeight','bold');
set(hLeg,'Box','on','EdgeColor','k','LineWidth',1.2);

% ---- Colored borders (black top/bottom, left blue, right maroon) ----
origUnits = ax.Units; ax.Units = 'normalized';
axpos = ax.Position; ax.Units = origUnits;
x0=axpos(1); y0=axpos(2); w=axpos(3); h=axpos(4);

annotation('line',[x0 x0+w],[y0 y0],'Color','k','LineWidth',1.6);         % bottom
annotation('line',[x0 x0+w],[y0+h y0+h],'Color','k','LineWidth',1.6);     % top
annotation('line',[x0 x0],[y0 y0+h],'Color',cPSNR,'LineWidth',1.6);       % left
annotation('line',[x0+w x0+w],[y0 y0+h],'Color',cSSIM,'LineWidth',1.6);   % right

% Save
[folder,name] = fileparts(matfile);
if isempty(folder), folder = pwd; end
outbase = fullfile(folder,[name '_psnrssim_coloredbox']);
exportgraphics(fig,[outbase '.pdf'],'ContentType','vector');
exportgraphics(fig,[outbase '.eps'],'ContentType','vector');
exportgraphics(fig,[outbase '.png'],'Resolution',600);

fprintf('Saved: %s.{pdf,eps,png}\n', outbase);
end
