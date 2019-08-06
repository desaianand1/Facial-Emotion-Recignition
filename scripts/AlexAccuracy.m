%% Load data to classify and filepaths
clear variables
load('testingData.mat');
directory = 'G:\My Drive\Image Rec Project\Code and results\oneEmotion\';
load([directory 'oneEmotionProbMat.mat']);

singleEmotionScores = zeros(length(testLabels), 11);

categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face'};
load('AlexNetTrainedFull15.mat');

%% Classify
[Ypred, scores] = classify(net, testVector);

%% Compute Conf Mat

confMat = zeros(11, 11);
for expect = 1:11
    for ident = 1:11
        val = length(find((categories(expect) == testLabels) & (categories(ident) == Ypred')));
        confMat(expect, ident) = val;
    end
end
