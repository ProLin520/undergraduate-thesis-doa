% DoA estimation via CNN: Training DATA generation (Snapshot Edition)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modified for 8-element ULA, 1 Source, Regression, with Snapshots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the DATA (更新了文件名以突出 Snapshots)
filename = fullfile('D:\Python\Project\doa_estimation\Graduation\data\CNN\CNN_M8_K1',...
    'TRAIN_DATA_8ULA_K1_low_SNR_res1_3D_90deg_Snapshots.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SNR_dB_vec = -20:5:0; % SNR values
SOURCE_K = 1; % 1 点信源
ULA_N = 8;    % 8 阵元
SOURCE.interval = 90;
G_res = 1; % degrees
T = 2000;  % 【新增】引入与测试集相同的快拍数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ang_d0 = -SOURCE.interval:G_res:SOURCE.interval;
ang_d = ang_d0'; 
S = length(SNR_dB_vec);

L = size(ang_d,1);
r_sam = zeros(ULA_N, ULA_N, 3, L);
R_sam_all = zeros([size(r_sam) S]); % 用于存储带快拍的采样协方差矩阵
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parfor i=1:S
    SNR_dB = SNR_dB_vec(i);
    noise_power = 10^(-SNR_dB/10);
    
    r_sam = zeros(ULA_N, ULA_N, 3, L);
    for ii=1:L
        SOURCE_angles = ang_d(ii,:);
        A_ula = zeros(ULA_N,SOURCE_K);
        for k=1:SOURCE_K
            A_ula(:,k) = ULA_steer_vec(SOURCE_angles(k),ULA_N);
        end
        
        % 【核心修改】加入快拍，模拟真实的采样协方差矩阵
        % 生成信号
        S_sig = (randn(SOURCE_K, T) + 1j*randn(SOURCE_K, T)) / sqrt(2); 
        X = A_ula * S_sig;
        % 生成噪声
        Eta = sqrt(noise_power) * (randn(ULA_N, T) + 1j*randn(ULA_N, T)) / sqrt(2);
        % 接收信号
        Y = X + Eta;
        % 计算采样协方差矩阵 (Sample Covariance Matrix)
        Ry_sam = Y * Y' / T;
        
        % 提取实部、虚部和相位作为 CNN 输入
        r_sam(:,:,1,ii) = real(Ry_sam); 
        r_sam(:,:,2,ii) = imag(Ry_sam);
        r_sam(:,:,3,ii) = angle(Ry_sam);
    end
    disp(['SNR = ', num2str(SNR_dB), ' dB (Snapshots) 循环完成。']);
    R_sam_all(:,:,:,:,i) = r_sam;
end

angles = ang_d;
time_tot = toc/60; 
disp(['带快拍数据生成耗时: ', num2str(time_tot), ' 分钟']);

if exist(filename, 'file') == 2
    delete(filename);
    disp('已清理旧文件...');
end

% 保存数据 (注意这里存的数据集名称改为了 /sam)
h5create(filename,'/sam',size(R_sam_all));
h5write(filename, '/sam', R_sam_all);
h5create(filename,'/angles',size(angles));
h5write(filename, '/angles', angles);

disp('>>> Snapshot 训练数据已成功保存！');