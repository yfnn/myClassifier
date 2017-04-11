clear all;

%% load training data
addpath ./data/mnist/
train_images = loadMNISTImages('train-images-idx3-ubyte');
train_labels = loadMNISTLabels('train-labels-idx1-ubyte');
train_labels(train_labels==0) = 10; % Remap 0 to 10

trainData = train_images;
trainlabel = train_labels;
[rows,cols] = size(trainData);

% load trainDatamnist10000;
% load trainlabelmnist10000;

%% PCA 
mean_inputData = mean(trainData,2);
mean_inputData = repmat(mean_inputData,1,60000);
diff_inputData = trainData - mean_inputData;
% % % for r=1:rows
% % %     diff_inputData(r,:) = trainData(r,:) - mean_inputData(r);
% % % end
sigma = diff_inputData'*diff_inputData;
[V,D] = eig(sigma);

% load V10000mnist;
% load D10000mnist;
d = diag(D);
[d1,index] = sort(d,1,'descend');
sV = V(:,index);%特征向量降序排列
sumd = sum(d1);
d2 = d1/sumd;
p=0;
for i=1:size(d2)
    if(sum(d2(1:i))>0.99)%能量取90%
        p=i;
        break;
    end
end
pV = sV(:,1:p);
base_num = size(pV,2);
for i=1:base_num
    base(:,i) = d1(i)^(-1/2)*diff_inputData*pV(:,i);
end
pcaTrainData = trainData'*base;

%% softmax
% load pcatrainData10000;
% load trainlabelmnist10000;
% load base;
inputSize = p; % Size of input vector 
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)

lambda = 1e-4; % Weight decay parameter

theta = 0.005 * randn(numClasses * inputSize, 1);
[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, pcaTrainData', trainlabel);

%% Learning parameters
addpath minFunc
options.maxIter = 100;                                  
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            pcaTrainData', trainlabel, options);
                        
%% STEP 5: Testing
%
%  You should now test your model against the test images.
%  To do this, you will first need to write softmaxPredict
%  (in softmaxPredict.m), which should return predictions
%  given a softmax model and the input data.
images = loadMNISTImages('t10k-images-idx3-ubyte');
labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

inputData = images;
inputData = (inputData'*base)';

% You will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);