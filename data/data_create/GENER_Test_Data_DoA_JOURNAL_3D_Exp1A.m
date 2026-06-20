% DoA estimation via CNN: Testing DATA generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 8-element ULA, 1 Source, (-90, 90) degrees, T=2000 Snapshots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
tic;
rng(14); % 固定随机种子以保证测试集一致性

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 保存路径 (请根据你的实际路径修改 base_path)
base_data_path = 'D:\Python\Project\doa_estimation\Graduation\data\CNN\CNN_M8_K1';
base_result_path = 'D:\Python\Project\doa_estimation\Graduation\result\CNN';

filename = fullfile(base_data_path,'TEST_DATA_8ULA_K1_min10dBSNR_T2000_3D_90deg(1).h5');
filename2 = fullfile(base_result_path, 'RMSE_l1SVD_8ULA_K1_min10dBSNR_T2000_3D_90deg.h5');
filename3 = fullfile(base_result_path, 'RMSE_UnESPRIT_8ULA_K1_min10dBSNR_T2000_3D_90deg.h5');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 2000; % 快拍数
SNR_dB = 0; % 测试信噪比 (低信噪比挑战)
SOURCE.K = 1; % 1 个信源
ULA.N = 8;    % 8 阵元
SOURCE.interval = 90; % -90 到 90 度
res = 1;      
Nsim = (2 * SOURCE.interval / res) + 1; % 181 个测试点

% ESPRIT 参数适配 8 阵元
ds = 1; 
ms = floor(ULA.N / 2); 
w = min(ms, ULA.N-ds-ms+1);  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 

SOURCE.power = ones(1,SOURCE.K).^2;
noise_power = min(SOURCE.power)*10^(-SNR_dB/10);
THETA_angles = -SOURCE.interval:res:SOURCE.interval;

threshold = 550;
l1_SVD_doa_est = zeros(SOURCE.K, Nsim);
UnESPRIT_doa_est = zeros(SOURCE.K, Nsim);

theta_current = -SOURCE.interval;

disp('开始生成带有快拍的测试集...');
for i = 1:Nsim
    A_ula = zeros(ULA.N, SOURCE.K);
    for k = 1:SOURCE.K 
        A_ula(:,k) = ULA_steer_vec(theta_current(k), ULA.N);
    end  
    
    Ry_the = A_ula * diag(ones(SOURCE.K,1)) * A_ula' + noise_power * eye(ULA.N);
    
    % 生成信号与噪声 (引入快拍)
    S = (randn(SOURCE.K, T) + 1j*randn(SOURCE.K, T)) / sqrt(2); 
    X = A_ula * S;
    Eta = sqrt(noise_power) * (randn(ULA.N, T) + 1j*randn(ULA.N, T)) / sqrt(2);
    Y = X + Eta;
    
    % 传统算法对比
    [ang_est_l1svd, sp_val_l1svd] = l1_SVD_DoA_est(Y, ULA.N, threshold, SOURCE.K, THETA_angles);
    l1_SVD_doa_est(:,i) = sort(ang_est_l1svd)';
    
    doas_unit_ESPRIT_sam = unit_ESPRIT(Y, T, ds, SOURCE.K, w);
    UnESPRIT_doa_est(:,i) = sort(doas_unit_ESPRIT_sam);
    
    % 计算采样协方差矩阵 (Sample Covariance Matrix)
    Ry_sam = Y * Y' / T;
    
    r.sam(:,:,1,i) = real(Ry_sam); 
    r.sam(:,:,2,i) = imag(Ry_sam);
    r.sam(:,:,3,i) = angle(Ry_sam);
    
    r.the(:,:,1,i) = real(Ry_the); 
    r.the(:,:,2,i) = imag(Ry_the);
    r.the(:,:,3,i) = angle(Ry_the);
    
    r.angles(:,i) = theta_current';
    theta_current = theta_current + res;
    
    if mod(i, 20) == 0 || i == Nsim
        disp(['已完成: ', num2str(i), '/', num2str(Nsim)]);
    end
end

disp('清理旧文件...');
files_to_check = {filename, filename2, filename3};
for f = 1:length(files_to_check)
    if exist(files_to_check{f}, 'file') == 2
        delete(files_to_check{f});
    end
end

disp('保存 .h5 测试文件...');
h5create(filename,'/sam', size(r.sam));
h5write(filename, '/sam', r.sam);
h5create(filename,'/theor',size(r.the));
h5write(filename, '/theor', r.the);
h5create(filename,'/angles',size(r.angles));
h5write(filename, '/angles', r.angles);

h5create(filename2,'/l1_SVD_ang',size(l1_SVD_doa_est));
h5write(filename2, '/l1_SVD_ang', l1_SVD_doa_est);

h5create(filename3,'/UnESPRIT_ang',size(UnESPRIT_doa_est));
h5write(filename3, '/UnESPRIT_ang', UnESPRIT_doa_est);

disp('>>> 测试集数据生成并保存完毕！');