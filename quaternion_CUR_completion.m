function quaternion_CUR_completion(imgPath, method, missingRate, rank_col, rank_row, n_iter, tol)
% Quaternion CUR-based image completion (SAI, RAPID, Hyper19, QSVD baseline)
% method ∈ {'SAI','RAPID','Hyper19','QSVD'}

     % ---------- Load image (no resize) ----------
    img = imread(imgPath);
    if size(img,3) ~= 3
        error('Input image must be RGB (H x W x 3).');
    end
    img = im2double(img);
    [h,w,~] = size(img);

    % ---------- Quaternion lift (RGB -> i,j,k) ----------
    A = rgb2quat(img);

    % Convert to quaternion (RGB in i,j,k)
    % A = quaternion(zeros(h,w), img(:,:,1), img(:,:,2), img(:,:,3));

    % --- Generate mask ---
    mask = double(rand(h,w) > missingRate);
    maskQ = quaternion(mask, zeros(h,w), zeros(h,w), zeros(h,w));
    A_miss = A .* maskQ;

    % --- Initialize ---
    X = A_miss;
    psnr_history = zeros(1,n_iter);
    ssim_history = zeros(1,n_iter);

    % alpha scaling
    A11 = complex(img(:,:,1), img(:,:,2));
    A12 = complex(img(:,:,3), zeros(h,w));
    CA  = [A11, A12; -conj(A12), conj(A11)];
    svals = svd(CA,'econ');
    alp = 1/(max(svals)^2);

    fprintf("=== Running %s Method ===\n",method);
    tic;
    for it = 1:n_iter
        fprintf("[%s] Iteration %d/%d\n",method,it,n_iter);

        % CUR approximation
        Y = CURT_quat(X, rank_col, rank_row, alp, tol, method);

        % Update missing entries only
        X = A_miss + (1 - maskQ).*Y;

        % Convert to RGB
        X_rgb = quat2rgb(X);

        % --- Gaussian denoising ---
        X_rgb = imgaussfilt(X_rgb, 0.5);
        X = rgb2quat(X_rgb);

        % Compute PSNR + SSIM
        psnr_history(it) = psnr(X_rgb, img);
        ssim_history(it) = ssim(X_rgb, img);
    end
    total_time = toc;

    % Final reconstruction
    X_final_rgb = quat2rgb(X);
    final_psnr = psnr(X_final_rgb,img);
    final_ssim = ssim(X_final_rgb,img);

    % --- Display Results ---
    figure;
    subplot(2,2,1); imshow(img); title('Original');
    subplot(2,2,2); imshow(quat2rgb(A_miss)); 
    title(sprintf('Masked (%.0f%% missing)',missingRate*100));
    subplot(2,2,3); imshow(X_final_rgb); 
    title(sprintf('%s Reconstructed\nPSNR %.2f dB, SSIM %.3f',method,final_psnr,final_ssim));
    subplot(2,2,4); 
    yyaxis left; plot(1:n_iter, psnr_history,'b-o','LineWidth',1); ylabel('PSNR (dB)');
    yyaxis right; plot(1:n_iter, ssim_history,'r-s','LineWidth',1); ylabel('SSIM');
    xlabel('Iteration'); grid on;
    title(sprintf('%s: PSNR/SSIM Evolution',method));
    sgtitle(sprintf('CUR + %s, Time: %.2f s', method, total_time));

    % --- Print Summary ---
    fprintf('\n=== Summary for %s ===\n',method);
    fprintf('Final PSNR: %.2f dB\n',final_psnr);
    fprintf('Final SSIM: %.3f\n',final_ssim);
    fprintf('Total Time: %.2f s\n',total_time);

end


%% === CUR step with selected solver ===
function output = CURT_quat(X, R1, R2, alp, tol, method)
    [m,n] = size(X);

    % Randomly choose row/col indices
    I = randperm(n, R1);
    J = randperm(m, R2);

    C = X(:,I);   % selected columns
    R = X(J,:);   % selected rows

    % Pseudoinverse via chosen method
    switch method
        case 'SAI'
            [C_pinv, ~] = quaternion_SAI1(C, alp, tol);
            [R_pinv, ~] = quaternion_SAI1(R, alp, tol);
        case 'RAPID'
            [C_pinv, ~] = quaternion_RAPID(C, 2, alp, tol); % order=2
            [R_pinv, ~] = quaternion_RAPID(R, 2, alp, tol);
        case 'Hyper19'
            X0c = alp * C'; X0r = alp * R';
            [C_pinv, ~] = quaternion_Hyper19(C, X0c, tol);
            [R_pinv, ~] = quaternion_Hyper19(R, X0r, tol);
        case 'QSVD'
            C_pinv = pinv(C); % MATLAB QSVD pseudoinverse
            R_pinv = pinv(R);
        case 'NewtonSchulz' 
            [C_pinv,~,~] = quaternion_NewtonSchulz(C,tol,200,0.5,alp);
            [R_pinv,~,~] = quaternion_NewtonSchulz(R,tol,200,0.5,alp);
        otherwise
            error('Unknown method: %s',method);
    end

    % CUR approximation
    U = C_pinv * X * R_pinv;
    output = C * U * R;
end


%% === Quaternion to RGB ===
function img = quat2rgb(Q)
    R = Q.x; G = Q.y; B = Q.z;
    img = cat(3, R, G, B);
    img = im2double(mat2gray(img));
end

%% === RGB to Quaternion ===
function Q = rgb2quat(img)
    [h,w,~] = size(img);
    Q = quaternion(zeros(h,w), img(:,:,1), img(:,:,2), img(:,:,3));
end
