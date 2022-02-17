close all; clear global; clc;
%% load data
X_train = readtable("X_train.csv");
X_train = normalize(X_train,"range"); %scale values to [0,1] interval
y_train = readtable("y_train.csv");

X_test = readtable("X_test.csv");
X_test = normalize(X_test,"range"); %scale values to [0,1] interval
y_test = readtable("y_test.csv");


%% Fitting model
knn_model = fitcknn(X_train,y_train, 'OptimizeHyperparameters',{'Distance','NumNeighbors'},...
                'HyperparameterOptimizationOptions', struct('Optimizer','gridsearch'));

%% performance check
% predict labels for train data
yfit = predict(knn_model,X_train);
% Confusion chart and accuracy, precision, recall, F1 score
figure;
[cm, order] = confusionmat(knn_model.Y,yfit);
c = confusionchart(cm,order);
c.RowSummary = 'row-normalized';
c.ColumnSummary = 'column-normalized';

TP = cm(2,2); TN = cm(1,1); FP =cm(1,2); FN = cm(2,1);
accuracy = (TP + TN)/(sum(sum(cm)));
fprintf('Accuracy is %f\n',accuracy);
prec = TP/(TP + FP);
fprintf('Precision is %f\n',prec);
rec = TP/(TP+FN);
fprintf('Recall is %f\n',rec);
F1 = 2*TP/(2*TP+FP+FN);
fprintf('F score is %f\n',F1);

%% statistics of testing set
% predict labels for test set
yfit = predict(knn_model,table2array(X_test));
% Confusion chart and accuracy, precision, recall, F1 score
figure;
[cm, order] = confusionmat(y_test.TenYearCHD,yfit);
c = confusionchart(cm,order);
c.RowSummary = 'row-normalized';
c.ColumnSummary = 'column-normalized';

TP = cm(2,2); TN = cm(1,1); FP =cm(1,2); FN = cm(2,1);
accuracy = (TP + TN)/(sum(sum(cm)));
fprintf('Accuracy is %f\n',accuracy);
prec = TP/(TP + FP);
fprintf('Precision is %f\n',prec);
rec = TP/(TP+FN);
fprintf('Recall is %f\n',rec);
F1 = 2*TP/(2*TP+FP+FN);
fprintf('F score is %f\n',F1);