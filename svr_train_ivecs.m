clear, close all;

addpath([cd '/LSSVM/LSSVMlabv1_8_R2009b_R2011a'])
dataFolderPath = '/storage/dane/jgrzybowska/MATLAB/ivectors/age_regression/data/';
addpath(dataFolderPath)

data = load([dataFolderPath 'aGender_ivec_TUBMzAgenderAndGerman_MFCC60DPartitioned07.mat']);
%devdata = load([dataFolderPath 'agender_test_WEKA.mat']);
load([dataFolderPath '/fixed_folds_aGender_K15.mat'],'folds');
%load('agender_test_WEKA_mean_std.mat');
%load('hyperparams.mat');
%devdata = devdata.all_stand;
%cols = randi(size(devdata,2),1000,1);
%devdata = devdata(:,clos);

% SETTINGS
minAge = 6;
maxAge = 80;
%group = (data.females);
%group = (data.males | data.females | data.children);
%group = logical(data.labels);
%folds = f;
%gam = gam3;
%sig2 = sig23;
%who = w;
%gam = ones(15,1).*gam;
%sig2 = ones(15,1).*sig2;

%folds = folds(group);
group = logical(data.labels);          % all data
kernel = 'RBF_kernel';
whiten = 0;
wcc = 1;                                  % wccn

%X = data.features(group,:);
Y = data.labels(group);
%X = data.all;
%Y = data.alllabels;

K = 15;
type = 'function estimation';
%[gam,sig2] = tunelssvm({X,Y,type,[],[], kernel},...
%   'simplex', 'leaveoneoutlssvm',{'mse'});
%folds = crossvalind('Kfold', size(X,1), K);
%folds = f;

% same speaker not in train and test
%allfolds = 0; 
%[~,b] = unique(data.allidx);
%for i = 1:size(b,1)
%    if i ~=  size(b,1)
%        n = b(i+1)-b(i);
%    else
%        n = size(X,1)-b(i)+1;
%    end
%        
%    fol = repmat(folds(i),n,1); 
%    allfolds = [allfolds;fol];
%end
%allfolds = allfolds(2:end,1);
%folds = allfolds;
%folds = data.folds;

%X_stand = bsxfun(@minus,X,mean_test_aGender');
%X_stand = bsxfun(@times,X_stand,1./std_test_aGender');
%X = X_stand;

%%
allTestsScores = [];
for k = 1:K
    test_idx = (folds == k);
    train_idx = (folds ~= k);
    
    %testSpeakers = data.allidx(test_idx);
    
    %X_tr = X(train_idx,:);
    %Y_tr = Y(train_idx);
    
    %rows = randi(size(X_tr,1),5000,1);
    %devX = X_tr(rows,:);
    %devY = Y_tr(rows,:);
    
    if whiten == 1
        m = mean(X(train_idx,:));
        X = bsxfun(@minus,X,m);
        W = calc_white_mat(cov(X(train_idx,:)));
        X = X*W;
    end
    
    features_wccn = zeros(size(data.features));
    Lf = 0;
    Lm = 0;
    Lc = 0;
    if wcc == 1
     Lf=wccn(data.features(data.females&train_idx',:), data.who(data.females&train_idx')');
     Lm=wccn(data.features(data.males&train_idx',:), data.who(data.males&train_idx')');
     Lc=wccn(data.features(data.children&train_idx',:), data.who(data.children&train_idx')');
      %L=wccn(data.features(train_idx',:), who(train_idx')');
      %FeaWCCN = data.features*L;
     features_wccn(data.females,:) = data.features(data.females,:)*Lf;
     features_wccn(data.males,:) = data.features(data.males,:)*Lm;
     features_wccn(data.children,:) = data.features(data.children,:)*Lc;
     % X = FeaWCCN(group,:);
      %m = mean(X(train_idx,:));
      %X = bsxfun(@minus,X,m);
    end
    
    X = features_wccn;
    
    [gam(k), sig2(k)] = tunelssvm({X(train_idx,:),Y(train_idx),type,[],[], kernel}, 'simplex', 'crossvalidatelssvm', {10,'mse'});          
    
    %model = initlssvm(X(train_idx,:),Y(train_idx),'f',[],[],'RBF_kernel');
    %L_fold = 3;                                                 % 3 fold CV
    %model = tunelssvm(model,'simplex', 'rcrossvalidatelssvm',{L_fold,'mae'},'wmyriad');
    %model = robustlssvm(model);
    %Y_pred1 = simlssvm(model,X(test_idx,:));
    
    %gam = 20.53258;     % RBF
    %sig2 = 322.6538;
    %gam = 10580.058;     % RBF
    %sig2 = 322.668;
    %gam = 0.002755;     % linear
    %sig2 = 128.8632;
    %gam = 25301.0366;   % poly
    %sig2 = [208.7688 3];
    
    [alpha,b] = trainlssvm({X(train_idx,:),Y(train_idx),type,gam(k),sig2(k), kernel});
    
    Y_pred = simlssvm({X(train_idx,:),Y(train_idx),type,gam(k),sig2(k), kernel ,'preprocess'},...
        {alpha,b},X(test_idx,:));
    Y_pred(Y_pred < minAge) = minAge;
    Y_pred(Y_pred > maxAge) = maxAge;
    
    Y_true = Y(test_idx);
    
    %MAE1 = 1/(length(Y_pred1))*sum(abs(Y_pred1-Y_true));
    %MAE_std1 = std(abs(Y_pred1-Y_true));
    
    testScores = zeros(size(data.labels,1),1);
    testScores(test_idx) = Y_pred;
    allTestsScores = [allTestsScores, testScores];
    
    %[Y_true_sorted, idx] = sort(Y_true);
    %Y_pred_sorted = Y_pred(idx);
    
    Y_prior = ones(size(Y_true)).*mean(Y(train_idx));
    MAE_prior(k) = 1/(length(Y_prior))*sum(abs(Y_prior-Y_true));
    MAE(k) = 1/(length(Y_pred))*sum(abs(Y_pred-Y_true));
    MAE_std = std(abs(Y_pred-Y_true));
    
    p(k) = 1/(length(Y_pred)-1)*((Y_true-mean(Y_true))/std(Y_true))'*((Y_pred-mean(Y_pred))/std(Y_pred));
end
%%
speaker_id = [1:770]';
allScoresPredicted = sum(allTestsScores,2);
Y_pred_per_speaker = zeros(size(speaker_id,1),1);
Y_true_per_speaker = zeros(size(speaker_id,1),1);
for i = 1:size(speaker_id,1)
    m = (data.who == i);
    Y_pred_per_speaker(i,1) = mean(allScoresPredicted(m));
    Y_true_per_speaker(i,1) = mean(data.labels(m));
end
%%
%Y_true_per_speaker = data.labels;
%Y_true_per_speaker(194,1) = 26;


ivecs400scores = table(speaker_id,Y_true_per_speaker,Y_pred_per_speaker);

%save('arff450meanPerSpeaker.mat', 'arff450scores');

MAE_per_speaker = 1/(length(Y_true_per_speaker))*sum(abs(Y_pred_per_speaker-Y_true_per_speaker));
p_per_speaker = 1/(length(Y_pred_per_speaker)-1)*...
    ((Y_true_per_speaker-mean(Y_true_per_speaker))/std(Y_true_per_speaker))'*...
    ((Y_pred_per_speaker-mean(Y_pred_per_speaker))/std(Y_pred_per_speaker));
%%
mean(MAE_prior)
mean(MAE_per_speaker)
mean(p_per_speaker)

[Y_true_sorted, idx] = sort(Y_true_per_speaker);
Y_pred_sorted = Y_pred_per_speaker(idx);

figure()
plot(Y_true_sorted, '-o'); hold on
plot(Y_pred_sorted, 'or')
legend('True', 'Predicted')
ylabel('age'), xlabel('# speaker')

save('ivecs400scores_allData.mat')

rmpath(dataFolderPath)
rmpath([cd '/LSSVM/LSSVMlabv1_8_R2009b_R2011a'])