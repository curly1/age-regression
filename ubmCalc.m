function [ubm,T] = ubmCalc(mfcc_ubm, mfcc_T)
%% Computes UBM and saves to file ubm.mat

%% Training the UBM
nmix        = 1024;
final_niter = 10;
ds_factor   = 1;
nworkers = 12;

ubm = gmm_em(mfcc_ubm, nmix, final_niter, ds_factor, nworkers);

stats_T = cell(size(mfcc_T,1),1);
for i=1:size(mfcc_T,1)
  [N,F] = compute_bw_stats(mfcc_T{i}, ubm);
  stats_T{i} = [N; F];
end

%% Learning the total variability subspace
tvDim = 400*8;
niter = 1;
T = train_tv_space(stats_T, ubm, tvDim, niter, nworkers);

try
save(['/storage/dane/jgrzybowska/MATLAB/ivectors/age_regression/data/ubm' num2str(nmix) '_T' num2str(tvDim) '_agender_dev_WEKA_stand.mat'], 'ubm', 'T');
catch
end

end
