%% Load training data
X_train = readtable("X_train_smote.csv");
y_train = readtable("y_train_smote.csv");
X_train = normalize(X_train,"range"); %scale values to [0,1] interval

%% Feature selection
rng('default')  %for reproducibility
[idx, weights] = relieff(table2array(X_train),table2array(y_train), 10);
% Plot a bar graph showing ranks
figure()
bar(weights(idx));
xlabel('Predictor Rank');
ylabel('Predictor Importance weight')
xticklabels(strrep(X_train.Properties.VariableNames(idx),'_','\_'))
xtickangle(90)

%% Testing if feature selection improves performance
rng('default') % for reproducebility
template = templateKNN("NumNeighbors",3);

X_train_new = X_train(:,X_train.Properties.VariableNames(idx(1:8)));
[hipotesis, pvalue] = testckfold(template, template, X_train, X_train_new, y_train.TenYearCHD, 'Alternative','greater', 'Test', '10x10t')

%% Fitting model with optimizing hyperparametrs
rng('default') % for reproducebility
knn_model_final = fitcknn(X_train_new,y_train, 'NumNeighbors', 3,'OptimizeHyperparameters',{'Distance'},...
                'HyperparameterOptimizationOptions', struct('Optimizer','gridsearch'));

%% Save model
save('knn_model_final.mat', 'knn_model_final');

%% Kfold Crosvalidation of model
crosval = crossval(knn_model_final,'Kfold',5);
KnnError = kfoldLoss(crosval);
fprintf('Crosvalidation error is %f\n',KnnError);

%% Analising performance
yfit = predict(knn_model_final,table2array(X_train_new));
figure;
[cm, order] = confusionmat(knn_model_final.Y,yfit);
c = confusionchart(cm,order);
c.RowSummary = 'row-normalized';
c.ColumnSummary = 'column-normalized';

% Confusion chart and accuracy, precision, recall, F1 score
TP = cm(2,2); TN = cm(1,1); FP =cm(1,2); FN = cm(2,1);
accuracy = (TP + TN)/(sum(sum(cm)));
fprintf('Accuracy is %f\n',accuracy);
prec = TP/(TP + FP);
fprintf('Precision is %f\n',prec);
rec = TP/(TP+FN);
fprintf('Recall is %f\n',rec);
F1 = 2*TP/(2*TP+FP+FN);
fprintf('F score is %f\n',F1);

%% Loding test sets
X_test = readtable("X_test_smote.csv");
y_test = readtable("y_test_smote.csv");
X_test = normalize(X_test,"range"); %scale values to [0,1] interval
X_test_new = X_test(:,X_test.Properties.VariableNames(idx(1:8))); % test set with selected predictors
save('X_test_feature_selected.mat',"X_test_new");

%% Performance statistic of predicting test set
fprintf('pPredicting test set:  ')
tic
yfitT = predict(knn_model_final,table2array(X_test_new));
toc

figure;
[cmT, orderT] = confusionmat(y_test.TenYearCHD,yfitT);
cT = confusionchart(cmT,orderT);
cT.RowSummary = 'row-normalized';
cT.ColumnSummary = 'column-normalized';

% Confusion chart and accuracy, precision, recall, F1 score
TPt = cmT(2,2); TNt = cmT(1,1); FPt =cmT(1,2); FNt = cmT(2,1);
accuracyT = (TPt + TNt)/(sum(sum(cmT)));
fprintf('Accuracy is %f\n',accuracyT);
precT = TPt/(TPt + FPt);
fprintf('Precision is %f\n',precT);
recT = TPt/(TPt+FNt);
fprintf('Recall is %f\n',recT);
F1t = 2*TPt/(2*TPt+FPt+FNt);
fprintf('F score is %f\n',F1t);
