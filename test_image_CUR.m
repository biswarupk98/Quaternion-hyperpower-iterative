% test_image_CUR.m
% Put image files under ./image/ subfolder
clear; clc;

imgFolder = fullfile(pwd,'image');
if ~exist(imgFolder,'dir')
    error('Place images in the folder: %s', imgFolder);
end

% pick image file (adjust name)
imgName = 'kodim16.png';            % change to your name inside image folder
imgPath = fullfile(imgFolder, imgName);
if ~exist(imgPath,'file')
    error('Image not found: %s', imgPath);
end

% parameters
missingRate = 0.7;     % fraction missing
rank_col = 60;
rank_row = 60;
n_iter = 25;
tol = 1e-10;

% methods to run (subset or full list)
methods={'NewtonSchulz'};
% methods = {'RAPID','Hyper19','QSVD','NewtonSchulz'};
% 'SAI',

% create results folder inside image folder
resultsFolder = fullfile(imgFolder,'results');
if ~exist(resultsFolder,'dir'), mkdir(resultsFolder); end

% run each method and save outputs
for k = 1:length(methods)
    method = methods{k};
    fprintf('\n=== Running method: %s ===\n', method);
    quaternion_CUR_completion_eps1(imgPath, method, missingRate, rank_col, rank_row, n_iter, tol, ...
        'outdir', true, 'denoise', true, 'savePNG', true);
end

fprintf('All done. Results saved under %s\n', resultsFolder);
