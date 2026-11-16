%% Quaternion Color Image Deblurring using SAI1 Method
% Based on quaternion linear system: AX = B
% Where X is original image, B is blurred image, A is blur operator
% Solution: X = A†B (using SAI1 to compute Moore-Penrose inverse A†)

clc; clear all; close all;
format long

%% 1. Load Color Images
fprintf('=== Quaternion Color Image Deblurring ===\n\n');
fprintf('Loading images...\n');

% Load first image (you can change to any color image)
X1_rgb = imread('peppers.png');
X1_rgb = imresize(X1_rgb, [256, 256]); % Resize for computational efficiency
X1_rgb = im2double(X1_rgb);

% Optionally load second image for comparison
X2_rgb = imread('autumn.tif');
X2_rgb = imresize(X2_rgb, [256, 256]);
X2_rgb = im2double(X2_rgb);

[height, width, ~] = size(X1_rgb);
n = height * width;
fprintf('Image dimensions: %d x %d (n = %d)\n', height, width, n);

%% 2. Convert RGB to Quaternion Representation
fprintf('\nConverting images to quaternion representation...\n');
fprintf('Quaternion form: X = 0 + R*i + G*j + B*k\n');

% Image 1
R1 = X1_rgb(:,:,1);
G1 = X1_rgb(:,:,2);
B1 = X1_rgb(:,:,3);
Q1 = quaternion(zeros(height, width), R1, G1, B1);
X1_vec = reshape(Q1, [n, 1]);

% Image 2
R2 = X2_rgb(:,:,1);
G2 = X2_rgb(:,:,2);
B2 = X2_rgb(:,:,3);
Q2 = quaternion(zeros(height, width), R2, G2, B2);
X2_vec = reshape(Q2, [n, 1]);

%% 3. Define Blurring Matrices (Toeplitz-based)
fprintf('\n=== Creating Blurring Matrices ===\n');

% Parameters (as in the paper)
sigma = 3;      % Gaussian blur parameter
tau = 10;       % Motion blur length
r = 20;         % Band radius for sparse matrix

% Create different blur types following paper's approach
fprintf('\n1. Motion Blur Matrix\n');
PSF_motion = fspecial('motion', tau, 45);
A_motion = create_toeplitz_blur_matrix(PSF_motion, height, width, r);
fprintf('   Matrix size: %d x %d\n', size(A_motion, 1), size(A_motion, 2));
fprintf('   Sparsity: %.2f%%\n', 100*nnz(A_motion)/(n*n));

fprintf('\n2. Gaussian Blur Matrix\n');
PSF_gaussian = fspecial('gaussian', 15, sigma);
A_gaussian = create_toeplitz_blur_matrix(PSF_gaussian, height, width, r);
fprintf('   Matrix size: %d x %d\n', size(A_gaussian, 1), size(A_gaussian, 2));
fprintf('   Sparsity: %.2f%%\n', 100*nnz(A_gaussian)/(n*n));

fprintf('\n3. Average Blur Matrix\n');
PSF_average = fspecial('average', 9);
A_average = create_toeplitz_blur_matrix(PSF_average, height, width, r);
fprintf('   Matrix size: %d x %d\n', size(A_average, 1), size(A_average, 2));
fprintf('   Sparsity: %.2f%%\n', 100*nnz(A_average)/(n*n));

% Convert to quaternion matrices (real-valued blur operator)
A_motion_q = quaternion(A_motion, zeros(size(A_motion)), ...
                        zeros(size(A_motion)), zeros(size(A_motion)));
A_gaussian_q = quaternion(A_gaussian, zeros(size(A_gaussian)), ...
                          zeros(size(A_gaussian)), zeros(size(A_gaussian)));
A_average_q = quaternion(A_average, zeros(size(A_average)), ...
                         zeros(size(A_average)), zeros(size(A_average)));

%% 4. Generate Blurred Images: B = AX
fprintf('\n=== Generating Blurred Images (B = AX) ===\n');

% Add noise level
noise_level = 0.001;

% Image 1 - Motion blur
fprintf('\nImage 1 - Motion Blur:\n');
B1_motion = A_motion_q * X1_vec;
B1_motion = add_noise(B1_motion, noise_level);
fprintf('  Applied blur and noise (level = %.4f)\n', noise_level);

% Image 1 - Gaussian blur
fprintf('Image 1 - Gaussian Blur:\n');
B1_gaussian = A_gaussian_q * X1_vec;
B1_gaussian = add_noise(B1_gaussian, noise_level);
fprintf('  Applied blur and noise (level = %.4f)\n', noise_level);

% Image 2 - Average blur
fprintf('Image 2 - Average Blur:\n');
B2_average = A_average_q * X2_vec;
B2_average = add_noise(B2_average, noise_level);
fprintf('  Applied blur and noise (level = %.4f)\n', noise_level);

% Visualize blurred images
img1_motion_blurred = qvec_to_rgb(B1_motion, height, width);
img1_gaussian_blurred = qvec_to_rgb(B1_gaussian, height, width);
img2_average_blurred = qvec_to_rgb(B2_average, height, width);

figure('Position', [100, 100, 1200, 400]);
subplot(1,3,1); imshow(img1_motion_blurred); title('Image 1 - Motion Blurred');
subplot(1,3,2); imshow(img1_gaussian_blurred); title('Image 1 - Gaussian Blurred');
subplot(1,3,3); imshow(img2_average_blurred); title('Image 2 - Average Blurred');
sgtitle('Blurred Images');
saveas(gcf, 'blurred_images.png');

%% 5. Compute Moore-Penrose Inverse using SAI1 Method
fprintf('\n=== Computing Moore-Penrose Inverse A† using SAI1 ===\n');

tolerance = 1e-10;

% Motion blur inverse
fprintf('\n1. Motion Blur Matrix Inverse:\n');
tic;
alpha_motion = 1 / (norm(A_motion_q, 'fro')^2);
fprintf('   Initial alpha = %.6e\n', alpha_motion);
[A_motion_pinv, iters_motion] = quaternion_SAI1(A_motion_q, alpha_motion, tolerance);
time_motion = toc;
fprintf('   Converged in %d iterations\n', iters_motion);
fprintf('   Computation time: %.4f seconds\n', time_motion);

% Gaussian blur inverse
fprintf('\n2. Gaussian Blur Matrix Inverse:\n');
tic;
alpha_gaussian = 1 / (norm(A_gaussian_q, 'fro')^2);
fprintf('   Initial alpha = %.6e\n', alpha_gaussian);
[A_gaussian_pinv, iters_gaussian] = quaternion_SAI1(A_gaussian_q, alpha_gaussian, tolerance);
time_gaussian = toc;
fprintf('   Converged in %d iterations\n', iters_gaussian);
fprintf('   Computation time: %.4f seconds\n', time_gaussian);

% Average blur inverse
fprintf('\n3. Average Blur Matrix Inverse:\n');
tic;
alpha_average = 1 / (norm(A_average_q, 'fro')^2);
fprintf('   Initial alpha = %.6e\n', alpha_average);
[A_average_pinv, iters_average] = quaternion_SAI1(A_average_q, alpha_average, tolerance);
time_average = toc;
fprintf('   Converged in %d iterations\n', iters_average);
fprintf('   Computation time: %.4f seconds\n', time_average);

%% 6. Restore Images: X_restored = A†B
fprintf('\n=== Restoring Images (X = A†B) ===\n');

% Image 1 - Motion blur
fprintf('\nRestoring Image 1 from motion blur...\n');
X1_motion_restored = A_motion_pinv * B1_motion;
img1_motion_restored = qvec_to_rgb(X1_motion_restored, height, width);

% Image 1 - Gaussian blur
fprintf('Restoring Image 1 from gaussian blur...\n');
X1_gaussian_restored = A_gaussian_pinv * B1_gaussian;
img1_gaussian_restored = qvec_to_rgb(X1_gaussian_restored, height, width);

% Image 2 - Average blur
fprintf('Restoring Image 2 from average blur...\n');
X2_average_restored = A_average_pinv * B2_average;
img2_average_restored = qvec_to_rgb(X2_average_restored, height, width);

%% 7. Display Results
fprintf('\n=== Displaying Results ===\n');

% Image 1 - Motion blur
figure('Position', [100, 100, 1600, 400]);
subplot(1,4,1); imshow(X1_rgb); title('Original Image');
subplot(1,4,2); imshow(img1_motion_blurred); title('Motion Blurred + Noise');
subplot(1,4,3); imshow(img1_motion_restored); title('Restored (SAI1)');
subplot(1,4,4); imshow(abs(X1_rgb - img1_motion_restored)*5); title('Error (5x)');
sgtitle('Image 1 - Motion Blur Restoration');
saveas(gcf, 'result_motion_blur.png');

% Image 1 - Gaussian blur
figure('Position', [100, 100, 1600, 400]);
subplot(1,4,1); imshow(X1_rgb); title('Original Image');
subplot(1,4,2); imshow(img1_gaussian_blurred); title('Gaussian Blurred + Noise');
subplot(1,4,3); imshow(img1_gaussian_restored); title('Restored (SAI1)');
subplot(1,4,4); imshow(abs(X1_rgb - img1_gaussian_restored)*5); title('Error (5x)');
sgtitle('Image 1 - Gaussian Blur Restoration');
saveas(gcf, 'result_gaussian_blur.png');

% Image 2 - Average blur
figure('Position', [100, 100, 1600, 400]);
subplot(1,4,1); imshow(X2_rgb); title('Original Image');
subplot(1,4,2); imshow(img2_average_blurred); title('Average Blurred + Noise');
subplot(1,4,3); imshow(img2_average_restored); title('Restored (SAI1)');
subplot(1,4,4); imshow(abs(X2_rgb - img2_average_restored)*5); title('Error (5x)');
sgtitle('Image 2 - Average Blur Restoration');
saveas(gcf, 'result_average_blur.png');

%% 8. Compute Quality Metrics
fprintf('\n=== Quality Metrics ===\n');

fprintf('\n--- Image 1: Motion Blur ---\n');
[mse1_m, psnr1_m, ssim1_m] = compute_metrics(X1_rgb, img1_motion_restored);
fprintf('MSE: %.6f, PSNR: %.4f dB, SSIM: %.6f\n', mse1_m, psnr1_m, ssim1_m);

fprintf('\n--- Image 1: Gaussian Blur ---\n');
[mse1_g, psnr1_g, ssim1_g] = compute_metrics(X1_rgb, img1_gaussian_restored);
fprintf('MSE: %.6f, PSNR: %.4f dB, SSIM: %.6f\n', mse1_g, psnr1_g, ssim1_g);

fprintf('\n--- Image 2: Average Blur ---\n');
[mse2_a, psnr2_a, ssim2_a] = compute_metrics(X2_rgb, img2_average_restored);
fprintf('MSE: %.6f, PSNR: %.4f dB, SSIM: %.6f\n', mse2_a, psnr2_a, ssim2_a);

%% 9. Create Summary Table
fprintf('\n=== Summary Table ===\n');
fprintf('%-20s %-12s %-12s %-12s %-12s\n', 'Blur Type', 'Iterations', 'Time (s)', 'PSNR (dB)', 'SSIM');
fprintf('%s\n', repmat('-', 1, 70));
fprintf('%-20s %-12d %-12.4f %-12.4f %-12.6f\n', 'Motion (Image 1)', iters_motion, time_motion, psnr1_m, ssim1_m);
fprintf('%-20s %-12d %-12.4f %-12.4f %-12.6f\n', 'Gaussian (Image 1)', iters_gaussian, time_gaussian, psnr1_g, ssim1_g);
fprintf('%-20s %-12d %-12.4f %-12.4f %-12.6f\n', 'Average (Image 2)', iters_average, time_average, psnr2_a, ssim2_a);

%% 10. Create Comparison Chart
figure('Position', [100, 100, 1200, 400]);

subplot(1,3,1);
bar([iters_motion, iters_gaussian, iters_average]);
set(gca, 'XTickLabel', {'Motion', 'Gaussian', 'Average'});
title('SAI1 Iterations');
ylabel('Number of Iterations');
grid on;

subplot(1,3,2);
bar([psnr1_m, psnr1_g, psnr2_a]);
set(gca, 'XTickLabel', {'Motion', 'Gaussian', 'Average'});
title('PSNR (Higher is Better)');
ylabel('PSNR (dB)');
grid on;

subplot(1,3,3);
bar([ssim1_m, ssim1_g, ssim2_a]);
set(gca, 'XTickLabel', {'Motion', 'Gaussian', 'Average'});
title('SSIM (Higher is Better)');
ylabel('SSIM');
ylim([0, 1]);
grid on;

sgtitle('SAI1 Method Performance Comparison');
saveas(gcf, 'performance_comparison.png');

%% 11. Save Results
fprintf('\nSaving results...\n');
save('quaternion_deblurring_sai1_results.mat', ...
    'mse1_m', 'psnr1_m', 'ssim1_m', 'iters_motion', 'time_motion', ...
    'mse1_g', 'psnr1_g', 'ssim1_g', 'iters_gaussian', 'time_gaussian', ...
    'mse2_a', 'psnr2_a', 'ssim2_a', 'iters_average', 'time_average');

fprintf('\n=== Processing Complete! ===\n');

%% ======================== HELPER FUNCTIONS ========================

%% Create Toeplitz-based Blur Matrix with Bandwidth Constraint
function A = create_toeplitz_blur_matrix(PSF, height, width, r)
    % Creates a sparse Toeplitz convolution matrix with bandwidth r
    % Following the paper's approach: |i-j| < r
    
    [p_h, p_w] = size(PSF);
    pad_h = floor(p_h/2);
    pad_w = floor(p_w/2);
    
    n = height * width;
    
    % Use sparse matrix for efficiency
    row_idx = [];
    col_idx = [];
    values = [];
    
    for i = 1:height
        for j = 1:width
            idx = (i-1)*width + j;
            
            for pi = 1:p_h
                for pj = 1:p_w
                    i_img = i + pi - pad_h - 1;
                    j_img = j + pj - pad_w - 1;
                    
                    % Periodic boundary conditions
                    i_img = mod(i_img - 1, height) + 1;
                    j_img = mod(j_img - 1, width) + 1;
                    
                    idx_img = (i_img-1)*width + j_img;
                    
                    % Apply bandwidth constraint |i-j| < r
                    if abs(idx - idx_img) < r
                        row_idx = [row_idx; idx];
                        col_idx = [col_idx; idx_img];
                        values = [values; PSF(pi, pj)];
                    end
                end
            end
        end
    end
    
    % Create sparse matrix
    A = sparse(row_idx, col_idx, values, n, n);
    A = full(A);
    
    % Normalize to ensure proper blur
    row_sums = sum(abs(A), 2);
    row_sums(row_sums == 0) = 1; % Avoid division by zero
    A = A ./ row_sums;
end

%% Add Quaternion Noise
function Q_noisy = add_noise(Q_vec, noise_level)
    % Add Gaussian noise to quaternion vector
    n = length(Q_vec);
    
    noise_i = noise_level * randn(n, 1);
    noise_j = noise_level * randn(n, 1);
    noise_k = noise_level * randn(n, 1);
    
    noise_q = quaternion(zeros(n, 1), noise_i, noise_j, noise_k);
    Q_noisy = Q_vec + noise_q;
end

%% Convert Quaternion Vector to RGB Image
function rgb_img = qvec_to_rgb(Q_vec, height, width)
    % Reshape and extract RGB components
    Q_img = reshape(Q_vec, [height, width]);
    [~, R, G, B] = parts(Q_img);
    rgb_img = cat(3, R, G, B);
    rgb_img = max(0, min(1, rgb_img));
end

%% Compute Quality Metrics
function [mse, psnr_val, ssim_val] = compute_metrics(original, reconstructed)
    mse = immse(original, reconstructed);
    [psnr_val, ~] = psnr(reconstructed, original);
    ssim_val = ssim(reconstructed, original);
end

%% Quaternion SAI1 Method (Hyperpower Iterative Method)
function [X_final, num_iterations] = quaternion_SAI1(A, alp, tolerance)
    % Structure-preserving generalized inverse using hyperpower method
    % Based on golden ratio constants
    
    [m, n] = size(A);
    
    % Initial guess: X₀ = α·A'
    Xj = alp * A';
    
    % Identity matrix (quaternion)
    I = quaternion(eye(m), zeros(m), zeros(m), zeros(m));
    
    % Golden ratio-based constants
    phi = (1 + sqrt(5)) / 2;  % φ ≈ 1.618
    psi = (1 - sqrt(5)) / 2;  % ψ ≈ -0.618
    
    num_iterations = 0;
    max_iterations = 1000;
    
    fprintf('   Iteration: ');
    while num_iterations < max_iterations
        % Compute residual: Rⱼ = I - A·Xⱼ
        Rj = I - A * Xj;
        
        % Compute powers
        Rj2 = Rj * Rj;
        Rj4 = Rj2 * Rj2;
        
        % Compute Qⱼ = (I + φ·Rⱼ² + Rⱼ⁴)(I + ψ·Rⱼ² + Rⱼ⁴)
        Qj = (I + phi * Rj2 + Rj4) * (I + psi * Rj2 + Rj4);
        
        % Update: Xⱼ₊₁ = Xⱼ(I + Rⱼ)Qⱼ
        Xj_next = Xj * (I + Rj) * Qj;
        
        num_iterations = num_iterations + 1;
        
        % Print progress
        if mod(num_iterations, 10) == 0
            fprintf('%d...', num_iterations);
        end
        
        % Convergence check
        if norm(Xj_next - Xj, 'fro') < tolerance
            fprintf('%d (converged)\n', num_iterations);
            break;
        end
        
        Xj = Xj_next;
    end
    
    if num_iterations >= max_iterations
        fprintf('(max iterations reached)\n');
        warning('SAI1 did not converge within maximum iterations');
    end
    
    X_final = Xj;
end