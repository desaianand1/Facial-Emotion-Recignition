%% Load data to classify and filepaths
clear variables
load('testingData.mat');
directory = 'G:\My Drive\Image Rec Project\Code and results\oneEmotion\';
load([directory 'oneEmotionProbMat.mat']);

singleEmotionScores = zeros(length(testLabels), 11);

categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face'};
load('HOGNetwork.mat');

%% Classify
testVals = zeros(26244, 1, 1, size(testVector, 4));
for k = 1:size(testVector, 4)
    [featureVector,hogVisualization] = extractHOGFeatures(testVector(:, :, :, k));
    testVals(:, 1, 1, k) = featureVector;
end
[Ypred, scores] = classify(net, testVals);

%% Compute Conf Mat

confMat = zeros(11, 11);
for expect = 1:11
    for ident = 1:11
        val = length(find((categories(expect) == testLabels) & (categories(ident) == Ypred')));
        confMat(expect, ident) = val;
    end
end
