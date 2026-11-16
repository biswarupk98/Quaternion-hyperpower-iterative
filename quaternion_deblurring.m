%% Quaternion Video Deblurring: SAI vs PINV — FULLY CORRECTED
% ═══════════════════════════════════════════════════════════════════════
% Based on the PROVEN WORKING image deblurring approach:
%   - Image: 512×512 house1, sigma=3, r=3, s=3, p=32, q=16
%   - Bi = kron(B1, B2), Bj = -0.5*Bi, Bk = -0.5*Bi
%   - Aq = quaternion(0, Bi, Bj, Bk)
%   - Bq = Aq * Xq  (blur)
%   - Xn = pinv(Aq) * Bq  (restore)
%   - Result: PSNR 38.42 dB, SSIM 0.9644, RR 0.025761
%
% Extension to video:
%   - Each frame is treated as a color image
%   - Frame is resized to (p*q) rows to match Aq dimensions
%   - Aq inverse computed ONCE via SAI and PINV, reused for all frames
%   - Compares SAI (hyperpower iteration) vs PINV (QTFM pseudoinverse)
%
% Requires: QTFM (Quaternion Toolbox for MATLAB)
% ═══════════════════════════════════════════════════════════════════════
clc; clear all; close all;
format long;

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║   Quaternion Video Deblurring: SAI vs PINV                  ║\n');
fprintf('║   Frame-by-Frame (Proven Kronecker Approach)                ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

if ~exist('results_figures', 'dir'), mkdir('results_figures'); end
if ~exist('results_data', 'dir'), mkdir('results_data'); end
if ~exist('Output_images', 'dir'), mkdir('Output_images'); end

%% ═══════════════════════════════════════════════════════════════════════
%% 1. LOAD VIDEO
%% ═══════════════════════════════════════════════════════════════════════
fprintf('[1] Loading video...\n');
max_frames_to_load = 15;

try
    [video_data, vid_info] = load_y4m_video('data/carphone_qcif.y4m', max_frames_to_load);
    fprintf('    ✓ Loaded Y4M video: %d×%d, %d frames\n', ...
        size(video_data,1), size(video_data,2), size(video_data,3));
catch ME
    fprintf('    Y4M not found (%s). Generating synthetic video...\n', ME.message);
    video_data = generate_test_video(144, 176, 10);
    fprintf('    ✓ Generated synthetic video: %d×%d, %d frames\n', ...
        size(video_data,1), size(video_data,2), size(video_data,3));
end

% video_data: [height, width, num_frames, 3] in range [0, 1]
[orig_height, orig_width, num_frames, ~] = size(video_data);
fprintf('    Original frame size: %d × %d\n\n', orig_height, orig_width);

%% ═══════════════════════════════════════════════════════════════════════
%% 2. BLUR PARAMETERS — EXACT SAME AS PROVEN WORKING IMAGE CODE
%% ═══════════════════════════════════════════════════════════════════════
fprintf('[2] Setting blur parameters (proven working: sigma=3, r=3, s=3, p=32, q=16)...\n');

sigma = 3;      % Gaussian blur spread
r = 3;          % Gaussian blur radius (bandwidth of B1)
s = 3;          % Uniform blur radius (bandwidth of B2)
p = 32;         % Size of B1 (Gaussian Toeplitz)
q = 16;         % Size of B2 (Uniform Toeplitz)

% Generate Toeplitz blur matrices
B1 = generateB1(p, sigma, r);   % p×p Gaussian Toeplitz
B2 = generateB2(q, s);          % q×q Uniform Toeplitz

% Kronecker product: Bi = kron(B1, B2) → (p*q) × (p*q) = 512 × 512
Bi = kron(B1, B2);
Bj = -0.5 * Bi;
Bk = -0.5 * Bi;

% Quaternion blur operator: Aq = quaternion(0, Bi, Bj, Bk)
Aq = quaternion(zeros(size(Bi)), Bi, Bj, Bk);

blur_dim = p * q;   % = 512: Aq is 512×512, so image column must have 512 rows
fprintf('    B1: %d×%d, B2: %d×%d\n', p, p, q, q);
fprintf('    Bi = kron(B1,B2): %d×%d\n', blur_dim, blur_dim);
fprintf('    Aq: %d×%d quaternion matrix\n\n', blur_dim, blur_dim);

%% ═══════════════════════════════════════════════════════════════════════
%% 3. COMPUTE INVERSES (once, reused for all frames)
%% ═══════════════════════════════════════════════════════════════════════

% --- Compute alpha for SAI ---
fprintf('[3] Computing generalized inverses of Aq (%d×%d)...\n', blur_dim, blur_dim);
fprintf('    Computing alpha via complex adjoint SVD...\n');
alp = compute_alpha_for_SAI(Aq);
fprintf('    alpha = %.6e\n\n', alp);

tolerance = 1e-6;

% --- METHOD 1: SAI (Hyperpower Iteration with golden ratio) ---
fprintf('    ┌──────────────────────────────────────────────┐\n');
fprintf('    │  SAI: Hyperpower Iteration (9th order)       │\n');
fprintf('    └──────────────────────────────────────────────┘\n');
tic;
[Aq_inv_SAI, sai_iters] = quaternion_SAI(Aq, alp, tolerance);
time_SAI_inv = toc;
fprintf('    ✓ SAI converged in %d iterations, %.4f seconds\n\n', sai_iters, time_SAI_inv);

% --- METHOD 2: PINV (QTFM pseudoinverse) ---
fprintf('    ┌──────────────────────────────────────────────┐\n');
fprintf('    │  PINV: QTFM Pseudoinverse                    │\n');
fprintf('    └──────────────────────────────────────────────┘\n');
tic;
Aq_inv_PINV = QTFM_pinv_compute(Aq);
time_PINV_inv = toc;
fprintf('    ✓ PINV computed in %.4f seconds\n\n', time_PINV_inv);

%% ═══════════════════════════════════════════════════════════════════════
%% 4. PROCESS EACH FRAME: BLUR → RESTORE (SAI & PINV)
%% ═══════════════════════════════════════════════════════════════════════
%
% For each frame:
%   1. Resize frame from (orig_height × orig_width) to (blur_dim × m_cols)
%      where blur_dim = p*q = 512 (rows must match Aq dimension)
%      m_cols = number of columns (we keep aspect ratio or use orig_width)
%   2. Form quaternion matrix: Xq = quaternion(0, R, G, B)
%   3. Blur: Bq = Aq * Xq
%   4. Restore SAI: Xn_sai = Aq_inv_SAI * Bq
%   5. Restore PINV: Xn_pinv = Aq_inv_PINV * Bq
%   6. Resize back to original dimensions for display and metrics
%
fprintf('[4] Processing frames: blur + restore...\n');

% Determine working dimensions
% Aq is blur_dim × blur_dim, so Xq must be blur_dim × m_cols
% We resize each frame to blur_dim rows, keeping the column count
m_cols = orig_width;  % Keep original width as number of columns

% Pre-allocate storage
video_blurred   = zeros(orig_height, orig_width, num_frames, 3);
video_restored_SAI  = zeros(orig_height, orig_width, num_frames, 3);
video_restored_PINV = zeros(orig_height, orig_width, num_frames, 3);

% Per-frame metrics
metrics = struct();
metrics.psnr_blur = zeros(num_frames, 1);
metrics.psnr_sai  = zeros(num_frames, 1);
metrics.psnr_pinv = zeros(num_frames, 1);
metrics.ssim_blur = zeros(num_frames, 1);
metrics.ssim_sai  = zeros(num_frames, 1);
metrics.ssim_pinv = zeros(num_frames, 1);
metrics.rr_sai    = zeros(num_frames, 1);
metrics.rr_pinv   = zeros(num_frames, 1);

tic;
for k = 1:num_frames
    fprintf('    Frame %d/%d: ', k, num_frames);
    
    % ── Extract frame and resize to working dimensions ──
    frame_orig = squeeze(video_data(:, :, k, :));  % [orig_h, orig_w, 3] in [0,1]
    
    % Resize to blur_dim rows (Aq requires blur_dim × blur_dim acting on
    % blur_dim × m_cols quaternion matrix)
    frame_resized = imresize(frame_orig, [blur_dim, m_cols]);
    
    % Scale to [0, 255] for quaternion operations (matching working code)
    R = frame_resized(:,:,1) * 255;
    G = frame_resized(:,:,2) * 255;
    B = frame_resized(:,:,3) * 255;
    
    % ── Form quaternion image matrix ──
    Xq = quaternion(zeros(blur_dim, m_cols), R, G, B);
    
    % ── Apply blur: Bq = Aq * Xq ──
    Bq = Aq * Xq;
    
    % ── Extract blurred image ──
    Blurred_R = Bq.x;  % i-component
    Blurred_G = Bq.y;  % j-component
    Blurred_B = Bq.z;  % k-component
    
    % ── Restore via SAI: Xn = Aq_inv_SAI * Bq ──
    Xn_sai = Aq_inv_SAI * Bq;
    SAI_R = Xn_sai.x;
    SAI_G = Xn_sai.y;
    SAI_B = Xn_sai.z;
    
    % ── Restore via PINV: Xn = Aq_inv_PINV * Bq ──
    Xn_pinv = Aq_inv_PINV * Bq;
    PINV_R = Xn_pinv.x;
    PINV_G = Xn_pinv.y;
    PINV_B = Xn_pinv.z;
    
    % ── Convert to uint8 images at working resolution ──
    orig_uint8   = uint8(cat(3, R, G, B));
    blur_uint8   = uint8(max(0, min(255, cat(3, Blurred_R, Blurred_G, Blurred_B))));
    sai_uint8    = uint8(max(0, min(255, cat(3, SAI_R, SAI_G, SAI_B))));
    pinv_uint8   = uint8(max(0, min(255, cat(3, PINV_R, PINV_G, PINV_B))));
    
    % ── Compute metrics at working resolution (before resize back) ──
    orig_gray = rgb2gray(orig_uint8);
    blur_gray = rgb2gray(uint8(255 * mat2gray(blur_uint8)));
    sai_gray  = rgb2gray(uint8(255 * mat2gray(sai_uint8)));
    pinv_gray = rgb2gray(uint8(255 * mat2gray(pinv_uint8)));
    
    metrics.psnr_blur(k) = psnr(blur_gray, orig_gray);
    metrics.psnr_sai(k)  = psnr(sai_gray, orig_gray);
    metrics.psnr_pinv(k) = psnr(pinv_gray, orig_gray);
    
    metrics.ssim_blur(k) = ssim(blur_gray, orig_gray);
    metrics.ssim_sai(k)  = ssim(sai_gray, orig_gray);
    metrics.ssim_pinv(k) = ssim(pinv_gray, orig_gray);
    
    metrics.rr_sai(k)  = norm(double(orig_uint8(:)) - double(sai_uint8(:))) / ...
                          norm(double(orig_uint8(:)));
    metrics.rr_pinv(k) = norm(double(orig_uint8(:)) - double(pinv_uint8(:))) / ...
                          norm(double(orig_uint8(:)));
    
    % ── Resize back to original dimensions and store ──
    video_blurred(:,:,k,:)       = imresize(double(blur_uint8)/255, [orig_height, orig_width]);
    video_restored_SAI(:,:,k,:)  = imresize(double(sai_uint8)/255,  [orig_height, orig_width]);
    video_restored_PINV(:,:,k,:) = imresize(double(pinv_uint8)/255, [orig_height, orig_width]);
    
    fprintf('SAI PSNR=%.2f dB, PINV PSNR=%.2f dB, RR_SAI=%.4f\n', ...
        metrics.psnr_sai(k), metrics.psnr_pinv(k), metrics.rr_sai(k));
end
time_processing = toc;
fprintf('    ✓ All frames processed in %.2f seconds\n\n', time_processing);

%% ═══════════════════════════════════════════════════════════════════════
%% 5. AGGREGATE METRICS AND REPORT
%% ═══════════════════════════════════════════════════════════════════════
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('                    QUALITY METRICS REPORT                    \n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

avg_psnr_blur = mean(metrics.psnr_blur);
avg_psnr_sai  = mean(metrics.psnr_sai);
avg_psnr_pinv = mean(metrics.psnr_pinv);
avg_ssim_blur = mean(metrics.ssim_blur);
avg_ssim_sai  = mean(metrics.ssim_sai);
avg_ssim_pinv = mean(metrics.ssim_pinv);
avg_rr_sai    = mean(metrics.rr_sai);
avg_rr_pinv   = mean(metrics.rr_pinv);

fprintf('Per-Frame Results:\n');
fprintf('─────────────────────────────────────────────────────────────\n');
fprintf('  Frame  |  PSNR_Blur |  PSNR_SAI  |  PSNR_PINV | RR_SAI   | RR_PINV\n');
fprintf('─────────────────────────────────────────────────────────────\n');
for k = 1:num_frames
    fprintf('  %3d    |  %7.2f   |  %7.2f   |  %7.2f   | %.6f | %.6f\n', ...
        k, metrics.psnr_blur(k), metrics.psnr_sai(k), metrics.psnr_pinv(k), ...
        metrics.rr_sai(k), metrics.rr_pinv(k));
end
fprintf('─────────────────────────────────────────────────────────────\n\n');

fprintf('Blurred:\n');
fprintf('  Avg PSNR: %.2f dB\n', avg_psnr_blur);
fprintf('  Avg SSIM: %.4f\n\n', avg_ssim_blur);

fprintf('SAI (Hyperpower Iteration, 9th order):\n');
fprintf('  Avg PSNR: %.2f dB  (gain: +%.2f dB over blurred)\n', avg_psnr_sai, avg_psnr_sai - avg_psnr_blur);
fprintf('  Avg SSIM: %.4f   (gain: +%.4f)\n', avg_ssim_sai, avg_ssim_sai - avg_ssim_blur);
fprintf('  Avg RR:   %.6f\n', avg_rr_sai);
fprintf('  Inverse computation: %.4f seconds (%d iterations)\n', time_SAI_inv, sai_iters);
fprintf('  Total time (inverse + all frames): %.4f seconds\n\n', time_SAI_inv + time_processing);

fprintf('PINV (QTFM Pseudoinverse):\n');
fprintf('  Avg PSNR: %.2f dB  (gain: +%.2f dB over blurred)\n', avg_psnr_pinv, avg_psnr_pinv - avg_psnr_blur);
fprintf('  Avg SSIM: %.4f   (gain: +%.4f)\n', avg_ssim_pinv, avg_ssim_pinv - avg_ssim_blur);
fprintf('  Avg RR:   %.6f\n', avg_rr_pinv);
fprintf('  Inverse computation: %.4f seconds\n', time_PINV_inv);
fprintf('  Total time (inverse + all frames): %.4f seconds\n\n', time_PINV_inv + time_processing);

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('DIRECT COMPARISON:\n');
fprintf('  PSNR difference (SAI − PINV):  %+.4f dB\n', avg_psnr_sai - avg_psnr_pinv);
fprintf('  SSIM difference (SAI − PINV):  %+.6f\n', avg_ssim_sai - avg_ssim_pinv);
fprintf('  RR   difference (SAI − PINV):  %+.6f\n', avg_rr_sai - avg_rr_pinv);
fprintf('  Inverse speed ratio (PINV time / SAI time): %.2fx\n', time_PINV_inv / time_SAI_inv);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

if avg_psnr_sai >= avg_psnr_pinv
    fprintf('🏆 SAI WINS: +%.4f dB better PSNR, %.2fx faster inverse!\n\n', ...
        avg_psnr_sai - avg_psnr_pinv, time_PINV_inv / time_SAI_inv);
else
    fprintf('🏆 PINV WINS: +%.4f dB better PSNR\n', avg_psnr_pinv - avg_psnr_sai);
    fprintf('   But SAI is %.2fx faster for inverse computation\n\n', time_PINV_inv / time_SAI_inv);
end

%% ═══════════════════════════════════════════════════════════════════════
%% 6. SAVE SAMPLE FRAMES
%% ═══════════════════════════════════════════════════════════════════════
fprintf('Saving sample frames...\n');
for k = [1, ceil(num_frames/2), num_frames]
    % Save at working resolution
    frame_orig_resized = imresize(squeeze(video_data(:,:,k,:)), [blur_dim, m_cols]);
    R = frame_orig_resized(:,:,1)*255;
    G = frame_orig_resized(:,:,2)*255;
    B = frame_orig_resized(:,:,3)*255;
    
    Xq_k = quaternion(zeros(blur_dim, m_cols), R, G, B);
    Bq_k = Aq * Xq_k;
    Xn_sai_k = Aq_inv_SAI * Bq_k;
    Xn_pinv_k = Aq_inv_PINV * Bq_k;
    
    imwrite(uint8(cat(3, R, G, B)), ...
        sprintf('Output_images/frame%d_original.png', k));
    imwrite(uint8(max(0, min(255, cat(3, Bq_k.x, Bq_k.y, Bq_k.z)))), ...
        sprintf('Output_images/frame%d_blurred.png', k));
    imwrite(uint8(max(0, min(255, cat(3, Xn_sai_k.x, Xn_sai_k.y, Xn_sai_k.z)))), ...
        sprintf('Output_images/frame%d_restored_SAI.png', k));
    imwrite(uint8(max(0, min(255, cat(3, Xn_pinv_k.x, Xn_pinv_k.y, Xn_pinv_k.z)))), ...
        sprintf('Output_images/frame%d_restored_PINV.png', k));
end
fprintf('    ✓ Saved to Output_images/\n\n');

%% ═══════════════════════════════════════════════════════════════════════
%% 7. VISUALIZATIONS
%% ═══════════════════════════════════════════════════════════════════════
fprintf('Creating visualizations...\n');

create_frame_comparison_figure(video_data, video_blurred, ...
    video_restored_SAI, video_restored_PINV, metrics, num_frames);

create_metrics_plots(metrics, num_frames, ...
    time_SAI_inv, time_PINV_inv, sai_iters);

fprintf('    ✓ Saved to results_figures/\n\n');

%% Save all results
save('results_data/video_deblur_SAI_vs_PINV.mat', ...
    'metrics', 'time_SAI_inv', 'time_PINV_inv', 'time_processing', ...
    'sai_iters', 'sigma', 'r', 's', 'p', 'q', 'tolerance', ...
    'blur_dim', 'num_frames');

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║                   PROCESSING COMPLETE                       ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n');


%% ═══════════════════════════════════════════════════════════════════════
%% ═══════════════════════════════════════════════════════════════════════
%%                         CORE ALGORITHMS
%% ═══════════════════════════════════════════════════════════════════════
%% ═══════════════════════════════════════════════════════════════════════

%% SAI: Hyperpower Iteration (9th order) with Golden Ratio Constants
function [X_final, num_iterations] = quaternion_SAI(A, alp, tolerance)
    % Compute structure-preserving generalized inverse for quaternion matrix
    % using quaternion algebra directly (no complex embedding).
    %
    % This is the EXACT SAI method from the working code.
    %
    % Input:
    %   A         - quaternion matrix (m × n)
    %   alp       - initial guess scaling: X0 = alp * A'
    %   tolerance - convergence tolerance on ||X_{j+1} - X_j||_F
    %
    % Output:
    %   X_final        - quaternion generalized inverse
    %   num_iterations - number of iterations to converge

    [m, ~] = size(A);

    %% Initial guess
    Xj = alp * A';   % quaternion conjugate transpose

    % Quaternion identity
    I = quaternion(eye(m), zeros(m), zeros(m), zeros(m));

    % Golden ratio constants for 9th-order hyperpower method
    phi = (1 + sqrt(5)) / 2;
    psi = (1 - sqrt(5)) / 2;

    num_iterations = 0;
    max_iter = 100;

    %% Iteration loop
    while num_iterations < max_iter
        % Quaternion residual
        Rj  = I - A * Xj;
        Rj2 = Rj * Rj;
        Rj4 = Rj2 * Rj2;

        % 9th-order hyperpower update with golden ratio
        Qj = (I + phi * Rj2 + Rj4) * (I + psi * Rj2 + Rj4);
        Xj_next = Xj * (I + Rj) * Qj;

        num_iterations = num_iterations + 1;

        % Convergence check
        diff_norm = norm(Xj_next - Xj, 'fro');
        fprintf('        Iter %2d: ||ΔX|| = %.4e\n', num_iterations, diff_norm);

        if diff_norm < tolerance
            fprintf('        ✓ Converged!\n');
            break;
        end

        Xj = Xj_next;
    end

    if num_iterations >= max_iter
        fprintf('        ⚠ Max iterations reached (||ΔX|| = %.4e)\n', diff_norm);
    end

    X_final = Xj_next;
end

%% Compute alpha for SAI initial guess via Complex Adjoint SVD
function alp = compute_alpha_for_SAI(Aq)
    % alpha = 1 / max(svd(CA))^2
    % where CA is the complex adjoint representation of Aq
    %
    % CA = [A11, A12; -conj(A12), conj(A11)]
    % A11 = S + i*Ai, A12 = Aj + i*Ak

    S   = Aq.w;   % scalar part (zeros for purely imaginary)
    A_i = Aq.x;   % i-component
    A_j = Aq.y;   % j-component
    A_k = Aq.z;   % k-component

    A11 = complex(S, A_i);
    A12 = complex(A_j, A_k);

    CA = [A11, A12; -conj(A12), conj(A11)];

    svals = svd(CA, 'econ');
    alp = 1 / (max(svals)^2);
end

%% QTFM Pseudoinverse Computation
function X_pinv = QTFM_pinv_compute(Aq)
    % Compute pseudoinverse of quaternion matrix
    % First tries QTFM's built-in pinv, then falls back to
    % complex adjoint method

    % Try QTFM built-in
    try
        X_pinv = pinv(Aq);
        fprintf('    (Used QTFM built-in pinv)\n');
        return;
    catch
        fprintf('    (QTFM pinv not available, using complex adjoint method)\n');
    end

    % Fallback: complex adjoint pseudoinverse
    S   = Aq.w;
    A_i = Aq.x;
    A_j = Aq.y;
    A_k = Aq.z;

    A11 = complex(S, A_i);
    A12 = complex(A_j, A_k);

    [m, n] = size(A11);

    CA = [A11, A12; -conj(A12), conj(A11)];

    % Compute pseudoinverse of complex adjoint
    CA_pinv = pinv(CA);

    % Extract blocks: CA_pinv has size [2n × 2m]
    % Top-left block: n×m, Top-right block: n×m
    P11 = CA_pinv(1:n, 1:m);
    P12 = CA_pinv(1:n, m+1:2*m);

    % Reconstruct quaternion matrix
    X_pinv = quaternion(real(P11), imag(P11), real(P12), imag(P12));
end


%% ═══════════════════════════════════════════════════════════════════════
%%                     BLUR MATRIX GENERATORS
%% ═══════════════════════════════════════════════════════════════════════

%% Generate Toeplitz matrix B1 (Gaussian blur)
function B1 = generateB1(n, sigma, r)
    % Generates the Toeplitz matrix B^(1) using a Gaussian kernel.
    % Inputs:
    %   n     - Matrix size (n × n)
    %   sigma - Standard deviation for Gaussian kernel
    %   r     - Radius for nonzero entries
    % Output:
    %   B1    - Toeplitz matrix B^(1)

    B1 = zeros(n);
    for i = 1:n
        for j = 1:n
            if abs(i - j) <= r
                B1(i, j) = (1 / (sigma * sqrt(2 * pi))) * ...
                    exp(-((i - j)^2) / (2 * sigma^2));
            end
        end
    end
end

%% Generate Toeplitz matrix B2 (Uniform weights)
function B2 = generateB2(n, s)
    % Generates the Toeplitz matrix B^(2) using uniform weight.
    % Inputs:
    %   n - Matrix size (n × n)
    %   s - Radius for nonzero entries
    % Output:
    %   B2 - Toeplitz matrix B^(2)

    B2 = zeros(n);
    for i = 1:n
        for j = 1:n
            if abs(i - j) <= s
                B2(i, j) = 1 / (2 * s - 1);
            end
        end
    end
end


%% ═══════════════════════════════════════════════════════════════════════
%%                         VIDEO I/O
%% ═══════════════════════════════════════════════════════════════════════

%% Load Y4M Video
function [video_data, info] = load_y4m_video(filename, max_frames)
    fid = fopen(filename, 'r');
    if fid == -1, error('Cannot open file: %s', filename); end

    header = fgetl(fid);
    tokens = strsplit(header);
    width = 0; height = 0;

    for i = 1:length(tokens)
        if startsWith(tokens{i}, 'W'), width = str2double(tokens{i}(2:end)); end
        if startsWith(tokens{i}, 'H'), height = str2double(tokens{i}(2:end)); end
    end

    if width == 0 || height == 0
        fclose(fid);
        error('Could not parse Y4M header');
    end

    info.width = width;
    info.height = height;

    video_data = zeros(height, width, max_frames, 3);
    count = 0;

    while ~feof(fid) && count < max_frames
        line = fgetl(fid);
        if ~ischar(line), break; end

        if strcmp(line, 'FRAME')
            Y = fread(fid, [width, height], 'uint8')';
            U = fread(fid, [width/2, height/2], 'uint8')';
            V = fread(fid, [width/2, height/2], 'uint8')';

            if isempty(Y), break; end

            U = imresize(U, [height, width], 'bilinear');
            V = imresize(V, [height, width], 'bilinear');

            % YUV to RGB
            R = double(Y) + 1.402 * (double(V) - 128);
            G = double(Y) - 0.344136 * (double(U) - 128) - 0.714136 * (double(V) - 128);
            B_ch = double(Y) + 1.772 * (double(U) - 128);

            count = count + 1;
            video_data(:, :, count, 1) = max(0, min(255, R)) / 255;
            video_data(:, :, count, 2) = max(0, min(255, G)) / 255;
            video_data(:, :, count, 3) = max(0, min(255, B_ch)) / 255;
        end
    end

    fclose(fid);
    video_data = video_data(:, :, 1:count, :);
    info.num_frames = count;
end

%% Generate Synthetic Test Video
function video = generate_test_video(height, width, num_frames)
    video = zeros(height, width, num_frames, 3);

    for k = 1:num_frames
        [X, Y] = meshgrid(1:width, 1:height);

        % Colorful background
        R = 0.5 + 0.3 * sin(X/20 + k/5) .* cos(Y/20);
        G = 0.5 + 0.3 * cos(X/20 - k/5) .* sin(Y/20);
        B_ch = 0.5 + 0.3 * sin((X+Y)/25 + k/4);

        % Moving bright circle
        cx = width/2 + 30*sin(2*pi*k/num_frames);
        cy = height/2 + 25*cos(2*pi*k/num_frames);
        dist = sqrt((X-cx).^2 + (Y-cy).^2);
        mask = dist < 18;
        R(mask) = 0.9; G(mask) = 0.2; B_ch(mask) = 0.1;

        % Sharp edges (checkerboard)
        check = mod(floor(X/12) + floor(Y/12), 2);
        region = (X > width*0.65) & (Y > height*0.65);
        R(region) = 0.3 + 0.5*check(region);
        G(region) = 0.3 + 0.5*check(region);
        B_ch(region) = 0.3 + 0.5*check(region);

        % Text-like horizontal lines
        for row_idx = 30:8:min(80, height)
            line_mask = (Y >= row_idx) & (Y <= row_idx+2) & (X >= 10) & (X <= 80);
            R(line_mask) = 0.1; G(line_mask) = 0.1; B_ch(line_mask) = 0.1;
        end

        video(:,:,k,1) = max(0, min(1, R));
        video(:,:,k,2) = max(0, min(1, G));
        video(:,:,k,3) = max(0, min(1, B_ch));
    end
end


%% ═══════════════════════════════════════════════════════════════════════
%%                        VISUALIZATION
%% ═══════════════════════════════════════════════════════════════════════

%% Frame Comparison Figure
function create_frame_comparison_figure(orig, blur, sai, pinv, metrics, nf)
    show_frames = unique([1, ceil(nf/2), nf]);
    num_show = length(show_frames);

    fig = figure('Position', [30, 30, 1800, 300*num_show], 'Visible', 'off');

    for i = 1:num_show
        f = show_frames(i);

        % Original
        subplot(num_show, 5, (i-1)*5 + 1);
        imshow(squeeze(orig(:,:,f,:)));
        title(sprintf('Original F%d', f), 'FontSize', 10);

        % Blurred
        subplot(num_show, 5, (i-1)*5 + 2);
        imshow(squeeze(blur(:,:,f,:)));
        title(sprintf('Blurred (%.1f dB)', metrics.psnr_blur(f)), 'FontSize', 10);

        % SAI Restored
        subplot(num_show, 5, (i-1)*5 + 3);
        imshow(squeeze(sai(:,:,f,:)));
        title(sprintf('SAI (%.1f dB)', metrics.psnr_sai(f)), 'FontSize', 10);

        % PINV Restored
        subplot(num_show, 5, (i-1)*5 + 4);
        imshow(squeeze(pinv(:,:,f,:)));
        title(sprintf('PINV (%.1f dB)', metrics.psnr_pinv(f)), 'FontSize', 10);

        % Error maps (SAI | PINV)
        subplot(num_show, 5, (i-1)*5 + 5);
        err_sai  = abs(squeeze(orig(:,:,f,:)) - squeeze(sai(:,:,f,:)));
        err_pinv = abs(squeeze(orig(:,:,f,:)) - squeeze(pinv(:,:,f,:)));
        imshow([err_sai, err_pinv] * 5);
        title('Error×5 (SAI|PINV)', 'FontSize', 9);
    end

    sgtitle('Quaternion Video Deblurring: SAI vs PINV (Kronecker Approach)', ...
        'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig, 'results_figures/frame_comparison.png');
    close(fig);
end

%% Metrics Plots
function create_metrics_plots(metrics, nf, time_sai, time_pinv, sai_iters)
    fig = figure('Position', [50, 50, 1600, 800], 'Visible', 'off');
    frames = 1:nf;

    % PSNR per frame
    subplot(2,3,1);
    plot(frames, metrics.psnr_blur, 'k--', 'LineWidth', 1.5); hold on;
    plot(frames, metrics.psnr_sai, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
    plot(frames, metrics.psnr_pinv, 'r-s', 'LineWidth', 2, 'MarkerSize', 6);
    legend('Blurred', 'SAI', 'PINV', 'Location', 'best');
    xlabel('Frame'); ylabel('PSNR (dB)');
    title('PSNR per Frame', 'FontWeight', 'bold'); grid on;

    % SSIM per frame
    subplot(2,3,2);
    plot(frames, metrics.ssim_blur, 'k--', 'LineWidth', 1.5); hold on;
    plot(frames, metrics.ssim_sai, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
    plot(frames, metrics.ssim_pinv, 'r-s', 'LineWidth', 2, 'MarkerSize', 6);
    legend('Blurred', 'SAI', 'PINV', 'Location', 'best');
    xlabel('Frame'); ylabel('SSIM');
    title('SSIM per Frame', 'FontWeight', 'bold'); grid on;

    % RR per frame
    subplot(2,3,3);
    plot(frames, metrics.rr_sai, 'b-o', 'LineWidth', 2, 'MarkerSize', 6); hold on;
    plot(frames, metrics.rr_pinv, 'r-s', 'LineWidth', 2, 'MarkerSize', 6);
    legend('SAI', 'PINV', 'Location', 'best');
    xlabel('Frame'); ylabel('Relative Residual');
    title('RR per Frame', 'FontWeight', 'bold'); grid on;

    % Bar: Avg PSNR
    subplot(2,3,4);
    methods = categorical({'Blurred', 'SAI', 'PINV'});
    methods = reordercats(methods, {'Blurred', 'SAI', 'PINV'});
    b = bar(methods, [mean(metrics.psnr_blur), mean(metrics.psnr_sai), mean(metrics.psnr_pinv)]);
    b.FaceColor = 'flat';
    b.CData = [0.5 0.5 0.5; 0.2 0.4 0.8; 0.8 0.2 0.2];
    title('Avg PSNR', 'FontWeight', 'bold');
    ylabel('dB'); grid on;

    % Bar: Avg SSIM
    subplot(2,3,5);
    b = bar(methods, [mean(metrics.ssim_blur), mean(metrics.ssim_sai), mean(metrics.ssim_pinv)]);
    b.FaceColor = 'flat';
    b.CData = [0.5 0.5 0.5; 0.2 0.4 0.8; 0.8 0.2 0.2];
    title('Avg SSIM', 'FontWeight', 'bold');
    ylim([0 1]); grid on;

    % Bar: Inverse computation time
    subplot(2,3,6);
    inv_methods = categorical({'SAI', 'PINV'});
    inv_methods = reordercats(inv_methods, {'SAI', 'PINV'});
    b = bar(inv_methods, [time_sai, time_pinv]);
    b.FaceColor = 'flat';
    b.CData = [0.2 0.4 0.8; 0.8 0.2 0.2];
    title(sprintf('Inverse Time (SAI: %d iters)', sai_iters), 'FontWeight', 'bold');
    ylabel('Seconds'); grid on;

    sgtitle('SAI vs PINV: Video Deblurring Performance', ...
        'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig, 'results_figures/metrics_plots.png');
    close(fig);
end