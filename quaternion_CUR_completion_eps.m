function quaternion_CUR_completion_eps(imgPath, method, missingRate, rank_col, rank_row, n_iter, tol, varargin)
% quaternion_CUR_completion  CUR-based quaternion image completion and save outputs.
%
% Usage:
% quaternion_CUR_completion(imgPath, method, missingRate, rank_col, rank_row, n_iter, tol)
%
% Inputs:
%   imgPath    - path to image file (RGB)
%   method     - 'SAI','RAPID','Hyper19','QSVD','NewtonSchulz'
%   missingRate- fraction of pixels missing (0..1)
%   rank_col   - number of sampled columns in CUR
%   rank_row   - number of sampled rows in CUR
%   n_iter     - number of CUR iterations
%   tol        - tolerance passed to solvers (if used)
%
% Optional name-value:
%   'outdir'   - true/false (default true) create results folder next to image
%   'denoise'  - true/false (default true) apply light Gaussian denoising
%   'savePNG'  - true/false (default true) save PNG as well as EPS
%
% The function saves 4 files:
%   <imagename>_<method>_original.eps
%   <imagename>_<method>_masked.eps
%   <imagename>_<method>_reconstructed.eps
%   <imagename>_<method>_psnrssim.eps

% parse options
p = inputParser;
addParameter(p,'outdir',true,@islogical);
addParameter(p,'denoise',true,@islogical);
addParameter(p,'savePNG',true,@islogical);
parse(p,varargin{:});
opts = p.Results;

% read image
img = imread(imgPath);
if size(img,3) ~= 3
    error('Input image must be RGB (H x W x 3).');
end
img = im2double(img);
[h,w,~] = size(img);

% quaternion lift (RGB channels as i,j,k)
A = rgb2quat(img);

% create mask: 1 observed, 0 missing
rng(0);            % reproducible mask
mask = double(rand(h,w) > missingRate);
maskQ = quaternion(mask, zeros(h,w), zeros(h,w), zeros(h,w));
A_miss = A .* maskQ;
% --- PSNR for masked image ---
A_miss_rgb = quat2rgb(A_miss);
ppsnr = psnr(A_miss_rgb, img);
fprintf('PPSNR (masked image vs original) = %.3f dB\n', ppsnr);
% initial reconstruction X = observed only
X = A_miss;

% compute alpha via complex embedding (largest singular value)
A11 = complex(img(:,:,1), img(:,:,2));
A12 = complex(img(:,:,3), zeros(h,w));
CA  = [A11, A12; -conj(A12), conj(A11)];
try
    sigma_max = svds(CA,1,'largest');
catch
    svals = svd(CA,'econ'); sigma_max = max(svals);
end
alp = 1/(max(sigma_max, eps)^2);

psnr_history = nan(1,n_iter);
ssim_history = nan(1,n_iter);

fprintf('Running %s, missingRate=%.2f, ranks=(%d,%d), iter=%d\n', method, missingRate, rank_col, rank_row, n_iter);
tstart = tic;
for it = 1:n_iter
    % compute CUR approximation of current X
    Y = CURT_quat(X, rank_col, rank_row, alp, tol, method);
    % update only missing entries
    X = A_miss + (1 - maskQ).*Y;
    % convert to RGB for optional denoise and metric
    X_rgb = quat2rgb(X);
    if opts.denoise
        X_rgb = imgaussfilt(X_rgb, 0.6);
    end
    % re-lift
    X = rgb2quat(X_rgb);
    % metrics
    psnr_history(it) = psnr(X_rgb, img);
    ssim_history(it) = ssim(X_rgb, img);
    fprintf(' it %2d/%d  PSNR=%.3f  SSIM=%.4f\n', it, n_iter, psnr_history(it), ssim_history(it));
end
total_time = toc(tstart);
X_final_rgb = quat2rgb(X);
final_psnr = psnr(X_final_rgb, img);
final_ssim = ssim(X_final_rgb, img);
fprintf('Done: PSNR=%.3f, SSIM=%.4f, time=%.2fs\n', final_psnr, final_ssim, total_time);

% prepare output folder inside image folder: <imageFolder>/results
[imgDir,imgName,~] = fileparts(imgPath);
if isempty(imgDir), imgDir = pwd; end
outFolder = fullfile(imgDir,'results');
if opts.outdir && ~exist(outFolder,'dir'), mkdir(outFolder); end

% safe base filename
methodTag = regexprep(method,'\W','');
basefn = sprintf('%s_%s', imgName, methodTag);

% save images (EPS + optional PNG), WITHOUT titles or axes (for paper)
% A) original
fig = figure('Visible','off','Units','pixels','Position',[50 50 800 800]);
imshow(img,'Border','tight'); axis off; set(gca,'position',[0 0 1 1]);
if opts.outdir
    print(fig, fullfile(outFolder,[basefn '_original.eps']), '-depsc2','-r300');
    if opts.savePNG, imwrite(img, fullfile(outFolder,[basefn '_original.png'])); end
end
close(fig);

% B) masked
fig = figure('Visible','off','Units','pixels','Position',[50 50 800 800]);
imshow(quat2rgb(A_miss),'Border','tight'); axis off; set(gca,'position',[0 0 1 1]);
if opts.outdir
    print(fig, fullfile(outFolder,[basefn '_masked.eps']), '-depsc2','-r300');
    if opts.savePNG, imwrite(quat2rgb(A_miss), fullfile(outFolder,[basefn '_masked.png'])); end
end
close(fig);

% C) reconstructed
fig = figure('Visible','off','Units','pixels','Position',[50 50 800 800]);
imshow(X_final_rgb,'Border','tight'); axis off; set(gca,'position',[0 0 1 1]);
if opts.outdir
    print(fig, fullfile(outFolder,[basefn '_reconstructed.eps']), '-depsc2','-r300');
    if opts.savePNG, imwrite(X_final_rgb, fullfile(outFolder,[basefn '_reconstructed.png'])); end
end
close(fig);

% D) PSNR/SSIM plot
fig = figure('Visible','off','Units','pixels','Position',[50 50 900 500]);
ax = axes('Position',[0.12 0.13 0.82 0.78]); hold on; grid on;
yyaxis left;
plot(1:n_iter, psnr_history, '-o','LineWidth',1.6,'MarkerSize',6,'Color',[0 0.45 0.74]);
ylabel('PSNR (dB)','Interpreter','latex','FontSize',11);
yyaxis right;
plot(1:n_iter, ssim_history, '-s','LineWidth',1.6,'MarkerSize',6,'Color',[0.85 0.33 0.10]);
ylabel('SSIM','Interpreter','latex','FontSize',11);
xlabel('Iteration','Interpreter','latex','FontSize',11);
legend({'PSNR','SSIM'},'Location','best','Interpreter','latex');
set(gca,'FontSize',10);
if opts.outdir
    print(fig, fullfile(outFolder,[basefn '_psnrssim.eps']), '-depsc2','-r300');
    if opts.savePNG, exportgraphics(fig, fullfile(outFolder,[basefn '_psnrssim.png']),'Resolution',300); end
end
close(fig);

% save mat results
save(fullfile(outFolder,[basefn '_results.mat']), 'psnr_history','ssim_history','final_psnr','final_ssim','total_time');

end

%% -------------------- Helper: CURT_quat --------------------
function output = CURT_quat(X, R1, R2, alp, tol, method)
    [m,n] = size(X);
    R1 = min(R1,n); R2 = min(R2,m);
    rng('shuffle');
    I = randperm(n,R1);
    J = randperm(m,R2);
    C = X(:,I);
    R = X(J,:);
    % compute pseudoinverses via chosen method
    switch lower(method)
        case 'sai'
            [C_pinv,~] = quaternion_SAI1(C, alp, tol);
            [R_pinv,~] = quaternion_SAI1(R, alp, tol);
        case 'rapid'
            [C_pinv,~] = quaternion_RAPID1(C,5,alp,tol);
            [R_pinv,~] = quaternion_RAPID1(R,5,alp,tol);
        case 'hyper19'
            X0c = alp*C'; X0r = alp*R';
            [C_pinv,~] = quaternion_Hyper19(C, X0c, tol);
            [R_pinv,~] = quaternion_Hyper19(R, X0r, tol);
        case 'qsvd'
            C_pinv = pinv(C); R_pinv = pinv(R);
        case 'newtonschulz'
            [C_pinv,~,~] = quaternion_NewtonSchulz(C,tol,200,0.5,alp);
            [R_pinv,~,~] = quaternion_NewtonSchulz(R,tol,200,0.5,alp);
        otherwise
            error('Unknown method %s', method);
    end
    % fallback if solver failed
    try
        if isempty(C_pinv) || any(isnan(double(C_pinv(1,1))))
            C_pinv = pinv(C);
        end
    catch
        C_pinv = pinv(C);
    end
    try
        if isempty(R_pinv) || any(isnan(double(R_pinv(1,1))))
            R_pinv = pinv(R);
        end
    catch
        R_pinv = pinv(R);
    end
    U = C_pinv * X * R_pinv;
    output = C * U * R;
end

%% -------------------- Quaternion <-> RGB helpers --------------------
function img = quat2rgb(Q)
    R = Q.x; G = Q.y; B = Q.z;
    img = cat(3, R, G, B);
    img = min(max(im2double(img),0),1);
end

function Q = rgb2quat(img)
    [h,w,~] = size(img);
    Q = quaternion(zeros(h,w), img(:,:,1), img(:,:,2), img(:,:,3));
end
