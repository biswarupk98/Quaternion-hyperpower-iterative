% Test script for CUR + Quaternion pseudoinverse methods (with QSVD baseline)

imgPath = fullfile('image','kodim16.png');   % your fruit.jpg path
missingRate = 0.7;   % 70% missing
rank_col = 60; 
rank_row = 60;
n_iter = 35;
tol = 1e-10;

methods = {'SAI','RAPID','Hyper19','QSVD','NewtonSchulz'};

for k = 1:length(methods)
    quaternion_CUR_completion(imgPath, methods{k}, missingRate, rank_col, rank_row, n_iter, tol);
end
