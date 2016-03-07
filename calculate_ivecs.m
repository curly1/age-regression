%% calculate i-vectors and save to ivec.mat
clear
addpath([cd '/MSR Identity Toolkit v1.0/code'])
addpath('/storage/dane/jgrzybowska/MATLAB/ivectors/age_regression/data')
%% SETTINGS
cv          = 0;            % 1 - perform crossvalidation on train data, 0 - use test data for test and train data for models
load_ubm    = 0;            % 1 - load ubm from file, 0 - create ubm
K           = 1;            % k - fold cross-validation
plot        = 1;
MFCCPart    = 0;

%% Train data
train_database = load('agender_train_WEKA.mat');
train_database = train_database.database_stand;
train_database = sortrows(train_database,'file_id','ascend');

%% Test data
if cv == 0
    K = 1;
    test_database = load('agender_train_WEKA.mat');
    test_database = test_database.database_stand;
    test_database = sortrows(test_database,'file_id','ascend');
end

%% osobna baza do utworzenia UBM'a i macierzy T
if load_ubm == 0,
    %ubm_database = train_database;
    %ubm_database = load('/storage/dane/jgrzybowska/MATLAB/ivectors/kroswalidacja_ivectors/parameterization_and_data_prep/_database_YT_agender_german_ubm.mat');
    ubm_database = load('agender_dev_WEKA.mat');
    %ubm_database = ubm_database.all_60p_cell;
    ubm_database = ubm_database.database_stand;
    %% dla all
    %ubm_database = num2cell(ubm_database, 1);
    %ubm_database = ubm_database';
    %%
    %ubm_database = num2cell(ubm_database, [1 2]);
    %ubm_database = ubm_database.database;
    %T_database = train_database;
    %T_database = load('/storage/dane/jgrzybowska/MATLAB/ivectors/kroswalidacja_ivectors/parameterization_and_data_prep/_database_YT_agender_german_ubm.mat');
    T_database = load('agender_dev_WEKA.mat'); 
    %T_database = T_database.all_60p_cell;
    T_database = T_database.database_stand;
    %% dla all
    %T_database = num2cell(T_database, 1);
    %T_database = T_database';
    %%
    %T_database = num2cell(T_database, [1 2]);
    %T_database = T_database.database;
    %ubm_database = checkDatabaseNaN(ubm_database);
    %T_database = checkDatabaseNaN(T_database);
end
%% also check for empty parameters (MFCC_delta_cms) cell array
[train_database, idx] = checkDatabaseNaN(train_database);
if cv == 0, [test_database, idx] = checkDatabaseNaN(test_database); end;

%% Data Partition
%if cv == 1, folds = databasePartition(train_database, K); end % proportional database partition
eer = zeros(1,K);
accuracy = zeros(1,K);
eer_conf = zeros(1,K);
accuracy_conf = zeros(1,K);
tr = zeros(1,K);
classes = unique(train_database.age_class);
NClass =  size(unique(train_database.age_class),1);

%% Train/load ubm
if load_ubm == 1, for_ubm = load('ubm1024_T400_agender_dev_WEKA_stand_60p.mat'); ubm = for_ubm.ubm; T = for_ubm.T;
else %[ubm,T] = ubmCalc(ubm_database.MFCC_delta_cms, T_database.MFCC_delta_cms);
    [ubm,T] = ubmCalc(ubm_database.MFCC_delta_cms, T_database.MFCC_delta_cms);
end
%% MFCC Partition
if MFCCPart == 1
    [newMFCCs, train_database] = utterancePartition2(train_database);
    train_ivec_matrix = ivectorsCalc2(newMFCCs, ubm, T);
else    
%% Calculate i-vectors
train_ivec_matrix = ivectorsCalc2(train_database.MFCC_delta_cms, ubm, T);
end
if cv == 0, test_ivec_matrix = ivectorsCalc2(test_database.MFCC_delta_cms, ubm, T); end  
%% Cross-validation
for k = 1:K
  %% 
  if k == 1, folds = crossvalind('Kfold', size(train_database,1), K); end
  if cv == 1, train_idx = (folds ~= k); test_idx = (folds == k);
  else train_idx = logical((1:size(train_database))');                    % train with all training data
  end
  %% Train models
  model_ivecs = zeros(NClass,size(train_ivec_matrix,2));
  age_idx = zeros(size(train_database,1),1);
  for ii = 1:NClass
    for n = 1:size(train_database,1)
      age_idx(n,1) = isequal(train_database.age_class(n), classes(ii));
    end
    rows = age_idx & train_idx;
    model_ivecs(ii,:) = mean(train_ivec_matrix(rows,:));
  end
  %% Test
  if cv == 1,
    test_ivecs = train_ivec_matrix(test_idx,:);
    test_labels = train_database.age_class(test_idx);
  else
    test_ivecs = test_ivec_matrix;
    test_labels = test_database.age_class;
  end
  reference_ids = test_labels;
%%
  scores = zeros(size(test_ivecs,1),NClass);
  for ii = 1:NClass
    scores(:,ii) = dot(repmat(model_ivecs(ii,:),size(test_ivecs,1),1), test_ivecs, 2);
  end

  %% normalizacja
  new_scores = scores./(norm(model_ivecs)*norm(test_ivecs));
  
  %% Post-processing
  scores_t = zeros(size(new_scores));
  for col=1:NClass
      scores_t(:,col) = ((new_scores(:,col))./mean((new_scores(:,classes~=col)),2));
  end
  %% EER, plots
  [max_scores, predicted_ids] = max(new_scores,[],2);
  try
      [accuracy(k), eer(k), stats(k), tr(k)] = get_results([reference_ids, predicted_ids, max_scores],plot);
  catch
      stats(k) = confusionmatStats(reference_ids,predicted_ids);
  end
  %% Confusion matrix
  [sorted_test_labels,i] = sort(test_labels); sorted_scores = new_scores(i,:); 
  figure()
  imagesc(sorted_scores); 
  title('Age Class Verification Likelihood (iVector Model)');
  ylabel('Test #'); xlabel('Model #');
  colorbar; axis xy;
end

%% save ivecs
labels = train_database.age;
features = train_ivec_matrix;

gen = train_database.gender;
ascii = zeros(size(train_database,1),1);
for i = 1:size(train_database,1)
  ascii(i) = double(gen{i});
end
ascii = ascii';
females = (ascii == 102);
children = (ascii == 120);
males = (ascii == 109);
%%
save('/storage/dane/jgrzybowska/MATLAB/ivectors/age_regression/data/aGender_ivec_3200_TUBMz_agender_dev_WEKA_WEKAParams_stand.mat', 'features', 'labels', 'stats', 'model_ivecs', 'females', 'males', 'children');

rmpath([cd '/MSR Identity Toolkit v1.0/code'])
rmpath('/storage/dane/jgrzybowska/MATLAB/ivectors/age_regression/data')